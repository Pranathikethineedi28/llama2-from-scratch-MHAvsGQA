"""Controlled comparison of MHA, GQA and MQA: one variable changes (n_kv_heads), everything else is fixed.

Same data, same batches (seeded), same initialization seed, same optimizer, same number of steps.
For each variant it measures:
  - parameter count
  - KV-cache size per generated token (calculated) and after generation (measured)
  - generation speed with and without the KV cache
  - training time per step
  - training and validation loss curves, including the best validation step (overfitting check)

Usage:
    python compare.py                          # default: 2,000 steps, 1 seed
    python compare.py --steps 10000 --seeds 0 1 2 --device cuda
    python compare.py --mlflow                 # also log every run to sqlite:///mlflow.db
Results are written to results/mha_vs_gqa.{json,md,png}.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from llama.data import get_batch, load_splits
from llama.model import LlamaModel, ModelArgs, count_parameters, generate

RESULTS = Path(__file__).resolve().parent / "results"
VARIANTS = {"MHA": 8, "GQA": 2, "MQA": 1}  # n_kv_heads, with 8 query heads


@torch.no_grad()
def estimate_loss(model, data, batches: int, batch_size: int, seq_len: int, seed: int, device) -> float:
    """Average loss over a FIXED set of batches, so every variant is scored on identical text."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    total = 0.0
    for _ in range(batches):
        x, y = get_batch(data, batch_size, seq_len, g)
        logits = model(x.to(device))
        total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.to(device).reshape(-1)).item()
    model.train()
    return total / batches


def generation_speed(model, prompt_len: int, new_tokens: int, use_cache: bool, device) -> tuple[float, int]:
    tokens = torch.zeros((1, prompt_len), dtype=torch.long, device=device)
    g = torch.Generator(device="cpu").manual_seed(0) if device == "cpu" else None
    start = time.perf_counter()
    _, cache_bytes = generate(model, tokens, new_tokens, use_cache=use_cache, generator=g)
    if device == "cuda":
        torch.cuda.synchronize()
    return new_tokens / (time.perf_counter() - start), cache_bytes


def run_variant(name, n_kv_heads, seed, steps, eval_every, tok, train_data, val_data, device, args, mlflow=None):
    torch.manual_seed(seed)
    margs = ModelArgs(vocab_size=tok.vocab_size, n_kv_heads=n_kv_heads, max_seq_len=args.seq_len + args.gen_tokens)
    model = LlamaModel(margs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batch_gen = torch.Generator().manual_seed(1000 + seed)  # identical training batches across variants

    curve = []
    start = time.perf_counter()
    for step in range(1, steps + 1):
        x, y = get_batch(train_data, args.batch_size, args.seq_len, batch_gen)
        logits = model(x.to(device))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.to(device).reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == steps:
            point = {
                "step": step,
                "train_loss": estimate_loss(model, train_data, args.eval_batches, args.batch_size, args.seq_len, 1, device),
                "val_loss": estimate_loss(model, val_data, args.eval_batches, args.batch_size, args.seq_len, 2, device),
            }
            curve.append(point)
            if mlflow:
                mlflow.log_metrics({"train_loss": point["train_loss"], "val_loss": point["val_loss"]}, step=step)
            print(f"  {name} seed {seed} step {step:>6}: train {point['train_loss']:.4f}  val {point['val_loss']:.4f}")
    train_seconds = time.perf_counter() - start

    best = min(curve, key=lambda p: p["val_loss"])
    tps_cache, cache_bytes = generation_speed(model, args.seq_len, args.gen_tokens, True, device)
    tps_nocache, _ = generation_speed(model, args.seq_len, args.gen_tokens, False, device)
    return {
        "variant": name,
        "seed": seed,
        "n_kv_heads": n_kv_heads,
        "params": count_parameters(model),
        "attention_params": sum(count_parameters(layer.attention) for layer in model.layers),
        "kv_cache_bytes_per_token": margs.kv_cache_bytes_per_token(4),
        "kv_cache_bytes_measured": cache_bytes,
        "tokens_in_cache": args.seq_len + args.gen_tokens,
        "ms_per_train_step": 1000 * train_seconds / steps,
        "gen_tokens_per_s_cache": tps_cache,
        "gen_tokens_per_s_no_cache": tps_nocache,
        "final_train_loss": curve[-1]["train_loss"],
        "final_val_loss": curve[-1]["val_loss"],
        "best_val_loss": best["val_loss"],
        "best_val_step": best["step"],
        "curve": curve,
    }


def llama7b_style_cache(n_kv_heads: int, context: int = 4096) -> float:
    """Calculated KV cache (GB) for one 4,096-token sequence at LLaMA-2-7B scale in fp16."""
    layers, head_dim, bytes_fp16 = 32, 128, 2
    return 2 * layers * n_kv_heads * head_dim * bytes_fp16 * context / 1024**3


def write_report(results: list[dict], args) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "mha_vs_gqa.json").write_text(json.dumps({"config": vars(args), "runs": results}, indent=2))

    def mean(variant, key):
        vals = [r[key] for r in results if r["variant"] == variant]
        return sum(vals) / len(vals)

    seeds = sorted({r["seed"] for r in results})
    lines = [
        "# MHA vs GQA vs MQA: controlled comparison", "",
        f"Tiny Shakespeare, character level. {args.steps:,} steps, batch {args.batch_size}, sequence length {args.seq_len}, "
        f"AdamW 3e-4, seeds {seeds}, device `{args.device}`. Losses are averages over the same {args.eval_batches} "
        f"fixed batches for every variant. Values are means across seeds.", "",
        "| | MHA (8 KV heads) | GQA (2 KV heads) | MQA (1 KV head) |", "|---|---:|---:|---:|",
    ]
    rows = [
        ("Total parameters", "params", "{:,.0f}"),
        ("Attention parameters", "attention_params", "{:,.0f}"),
        ("KV cache per token (bytes, fp32)", "kv_cache_bytes_per_token", "{:,.0f}"),
        ("KV cache after generation (KB, measured)", "kv_cache_bytes_measured", "{:,.1f}"),
        ("Training time per step (ms)", "ms_per_train_step", "{:,.1f}"),
        ("Generation, with KV cache (tokens/s)", "gen_tokens_per_s_cache", "{:,.0f}"),
        ("Generation, without cache (tokens/s)", "gen_tokens_per_s_no_cache", "{:,.0f}"),
        ("Final training loss", "final_train_loss", "{:.4f}"),
        ("Final validation loss", "final_val_loss", "{:.4f}"),
        ("Best validation loss", "best_val_loss", "{:.4f}"),
        ("Step of best validation loss", "best_val_step", "{:,.0f}"),
    ]
    for label, key, fmt in rows:
        vals = []
        for v in VARIANTS:
            x = mean(v, key)
            vals.append(fmt.format(x / 1024 if key == "kv_cache_bytes_measured" else x))
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    lines += [
        "", "## Why it matters at scale (calculated)", "",
        "KV cache for one 4,096-token sequence with LLaMA-2-7B dimensions (32 layers, head size 128, fp16):", "",
        "| KV heads | KV cache per sequence |", "|---|---:|",
    ]
    for kv in (32, 8, 1):
        label = {32: "32 (MHA)", 8: "8 (GQA)", 1: "1 (MQA)"}[kv]
        lines.append(f"| {label} | {llama7b_style_cache(kv):.2f} GB |")
    (RESULTS / "mha_vs_gqa.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig, ax = plt.subplots(figsize=(7, 4))
        colors = {"MHA": "#4C72B0", "GQA": "#DD8452", "MQA": "#55A868"}
        for v in VARIANTS:
            run = next(r for r in results if r["variant"] == v)
            steps = [p["step"] for p in run["curve"]]
            ax.plot(steps, [p["val_loss"] for p in run["curve"]], color=colors[v], label=f"{v} validation")
            ax.plot(steps, [p["train_loss"] for p in run["curve"]], color=colors[v], linestyle="--", alpha=0.7,
                    label=f"{v} training")
        ax.set_xlabel("step")
        ax.set_ylabel("cross-entropy loss")
        ax.set_title(f"MHA vs GQA vs MQA (seed {results[0]['seed']})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(RESULTS / "mha_vs_gqa.png", dpi=120)
    except ImportError:
        print("matplotlib not installed: skipping the loss-curve plot")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--gen-tokens", type=int, default=256)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--mlflow", action="store_true")
    args = p.parse_args()

    tok, train_data, val_data = load_splits()
    mlflow = None
    if args.mlflow:
        import mlflow  # noqa: PLC0415

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("mha-vs-gqa")

    results = []
    for seed in args.seeds:
        for name, kv in VARIANTS.items():
            if mlflow:
                with mlflow.start_run(run_name=f"{name}-seed{seed}"):
                    mlflow.log_params({"variant": name, "n_kv_heads": kv, "seed": seed, "steps": args.steps,
                                       "batch_size": args.batch_size, "seq_len": args.seq_len})
                    r = run_variant(name, kv, seed, args.steps, args.eval_every, tok, train_data, val_data,
                                    args.device, args, mlflow)
                    mlflow.log_metrics({k: float(v) for k, v in r.items() if isinstance(v, (int, float)) and k != "seed"})
            else:
                r = run_variant(name, kv, seed, args.steps, args.eval_every, tok, train_data, val_data, args.device, args)
            results.append(r)
    write_report(results, args)


if __name__ == "__main__":
    main()
