"""Tiny Shakespeare with a character-level tokenizer (same setup as the notebook)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import torch

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "input.txt"


class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = dict(enumerate(self.chars))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def load_text(path: Path = DATA_FILE) -> str:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    return path.read_text(encoding="utf-8")


def load_splits(train_frac: float = 0.9):
    text = load_text()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    split = int(train_frac * len(data))
    return tok, data[:split], data[split:]


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, generator: torch.Generator):
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + seq_len] for i in starts])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in starts])
    return x, y
