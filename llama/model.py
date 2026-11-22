"""LLaMA 2-style Transformer from the notebook, packaged as a module, plus a KV cache.

The only architectural knob compared here is `n_kv_heads`:
- n_kv_heads == n_heads  -> multi-head attention (MHA)
- 1 < n_kv_heads < n_heads -> grouped-query attention (GQA)
- n_kv_heads == 1        -> multi-query attention (MQA)

The KV cache stores keys/values *before* they are repeated across query heads, which is exactly
where GQA saves memory during generation: the cache shrinks by n_heads / n_kv_heads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    vocab_size: int = 65
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 2
    hidden_dim: int = 1024
    max_seq_len: int = 512
    norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def kv_cache_bytes_per_token(self, bytes_per_value: int = 4) -> int:
        """Keys + values, for every layer, per generated token."""
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * bytes_per_value


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


def precompute_rope(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    indices = torch.arange(0, head_dim, 2).float()
    freqs = 1.0 / (theta ** (indices / head_dim))
    angles = torch.outer(torch.arange(max_seq_len).float(), freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    # x: [B, H, S, Hd]; cos/sin: [S, Hd/2]
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    out = torch.stack([x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1)
    return out.flatten(-2)


class KVCache:
    """Per-layer cache of keys and values with shape [B, n_kv_heads, S, head_dim]."""

    def __init__(self, n_layers: int):
        self.k: list[torch.Tensor | None] = [None] * n_layers
        self.v: list[torch.Tensor | None] = [None] * n_layers

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        if self.k[layer] is not None:
            k = torch.cat([self.k[layer], k], dim=2)
            v = torch.cat([self.v[layer], v], dim=2)
        self.k[layer], self.v[layer] = k, v
        return k, v

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v if t is not None)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        if args.n_heads % args.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.layer_id = layer_id
        self.n_heads, self.n_kv_heads, self.head_dim = args.n_heads, args.n_kv_heads, args.head_dim
        self.n_rep = args.n_heads // args.n_kv_heads  # query heads sharing one K/V head
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

    def forward(self, x, cos, sin, cache: KVCache | None = None):
        B, S, D = x.shape
        q = self.wq(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if cache is not None:
            k, v = cache.update(self.layer_id, k, v)  # cached BEFORE repeating: the GQA saving

        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        T = k.shape[2]
        if S > 1:  # causal mask; with a cache, query i sits at absolute position T - S + i
            mask = torch.triu(torch.ones(S, T, dtype=torch.bool, device=x.device), diagonal=T - S + 1)
            scores = scores.masked_fill(mask, float("-inf"))
        out = F.softmax(scores, dim=-1) @ v
        return self.wo(out.transpose(1, 2).contiguous().view(B, S, D))


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))  # SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.attention = Attention(args, layer_id)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, x, cos, sin, cache=None):
        h = x + self.attention(self.attention_norm(x), cos, sin, cache)
        return h + self.feed_forward(self.ffn_norm(h))


class LlamaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args, i) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        cos, sin = precompute_rope(args.head_dim, args.max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, tokens, start_pos: int = 0, cache: KVCache | None = None):
        B, S = tokens.shape
        if start_pos + S > self.args.max_seq_len:
            raise ValueError(f"sequence length {start_pos + S} exceeds max_seq_len={self.args.max_seq_len}")
        x = self.tok_embeddings(tokens)
        cos, sin = self.cos[start_pos:start_pos + S], self.sin[start_pos:start_pos + S]
        for layer in self.layers:
            x = layer(x, cos, sin, cache)
        return self.output(self.norm(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def generate(model: LlamaModel, tokens: torch.Tensor, max_new_tokens: int, use_cache: bool = True,
             temperature: float = 1.0, generator: torch.Generator | None = None):
    """Sample tokens. Returns (tokens, kv_cache_bytes). With the cache, each step feeds one token."""
    model.eval()
    cache = KVCache(model.args.n_layers) if use_cache else None
    logits = model(tokens, 0, cache) if use_cache else model(tokens)
    pos = tokens.shape[1]
    for i in range(max_new_tokens):
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=generator)
        tokens = torch.cat([tokens, nxt], dim=1)
        if i == max_new_tokens - 1:
            break  # the last sampled token is never needed as input
        if use_cache:
            logits = model(nxt, pos, cache)
            pos += 1
        else:
            logits = model(tokens[:, -model.args.max_seq_len:])
    return tokens, (cache.nbytes() if cache else 0)
