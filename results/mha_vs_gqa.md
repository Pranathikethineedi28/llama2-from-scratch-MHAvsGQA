# MHA vs GQA vs MQA: controlled comparison

Tiny Shakespeare, character level. 1,500 steps, batch 16, sequence length 128, AdamW 3e-4, seeds [0], device `cpu`. Losses are averages over the same 20 fixed batches for every variant. Values are means across seeds.

| | MHA (8 KV heads) | GQA (2 KV heads) | MQA (1 KV head) |
|---|---:|---:|---:|
| Total parameters | 4,229,888 | 3,836,672 | 3,771,136 |
| Attention parameters | 1,048,576 | 655,360 | 589,824 |
| KV cache per token (bytes, fp32) | 8,192 | 2,048 | 1,024 |
| KV cache after generation (KB, measured) | 3,064.0 | 766.0 | 383.0 |
| Training time per step (ms) | 429.4 | 339.4 | 329.6 |
| Generation, with KV cache (tokens/s) | 407 | 489 | 520 |
| Generation, without cache (tokens/s) | 47 | 55 | 51 |
| Final training loss | 1.3791 | 1.3697 | 1.3767 |
| Final validation loss | 1.5861 | 1.5772 | 1.5811 |
| Best validation loss | 1.5861 | 1.5772 | 1.5811 |
| Step of best validation loss | 1,500 | 1,500 | 1,500 |

## Why it matters at scale (calculated)

KV cache for one 4,096-token sequence with LLaMA-2-7B dimensions (32 layers, head size 128, fp16):

| KV heads | KV cache per sequence |
|---|---:|
| 32 (MHA) | 2.00 GB |
| 8 (GQA) | 0.50 GB |
| 1 (MQA) | 0.06 GB |
