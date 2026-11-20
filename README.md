# LLaMA 2 From Scratch in PyTorch

A small LLaMA 2-style language model implemented and trained from scratch in PyTorch.

I built this project to understand the architecture at the code level instead of starting with a pretrained model. I first implemented the model with Multi-Head Attention (MHA), then modified the attention layer to use Grouped-Query Attention (GQA).

The model is trained on Tiny Shakespeare for character-level next-token prediction.

## What I Implemented

The model includes:

- Token embeddings
- RMSNorm
- Rotary Positional Embeddings (RoPE)
- Causal self-attention
- Multi-Head Attention (MHA)
- Grouped-Query Attention (GQA)
- SwiGLU feed-forward layers
- Residual connections
- Transformer blocks
- Language modeling head
- Autoregressive text generation

The overall model flow is:

`tokens → embeddings → transformer blocks → RMSNorm → LM head → logits`

Each transformer block follows:

`RMSNorm → Attention → Residual → RMSNorm → SwiGLU → Residual`

## MHA Implementation

I started with standard multi-head attention.

For the current model configuration:

```text
dim = 256
n_heads = 8
head_dim = 32
```

MHA uses the same number of query, key, and value heads:

```text
Q heads = 8
K heads = 8
V heads = 8
```

The Q, K, and V projections therefore each produce 256-dimensional outputs.

## GQA Implementation

I then changed the attention layer to Grouped-Query Attention while keeping the rest of the model largely unchanged.

The GQA configuration uses:

```text
Query heads = 8
Key heads   = 2
Value heads = 2
```

Since there are 8 query heads and 2 KV heads, each KV head is shared by 4 query heads.

With a head dimension of 32:

```text
Q projection: 256 → 256
K projection: 256 → 64
V projection: 256 → 64
```

The K and V heads are repeated across their corresponding query groups during attention computation.

This allowed me to experiment with the difference between MHA and GQA without changing the rest of the Transformer architecture.

## Training

Both models are trained from random initialization using Tiny Shakespeare.

I used a simple character-level tokenizer, where every unique character in the dataset is mapped to an integer ID.

Training uses:

- Cross-entropy loss
- AdamW optimizer
- Random sequence batches
- Autoregressive next-token prediction
- GPU training with PyTorch/CUDA

## GQA Training Results

The GQA model was trained for 10,000 optimization steps.

Some checkpoints from the run:

| Step | Loss |
|---:|---:|
| 0 | 4.3485 |
| 100 | 2.3423 |
| 500 | 1.5930 |
| 1,000 | 1.4043 |
| 2,000 | 1.2734 |
| 3,000 | 1.2000 |
| 5,000 | 1.1233 |
| 7,000 | 1.0051 |
| 8,000 | 0.8425 |
| 9,000 | 0.8066 |
| 9,900 | 0.7567 |

The training loss decreased from **4.3485 to 0.7567** during the run.

On held-out text, the final **validation loss was 1.9855**. The gap between training loss (0.76) and validation loss (1.99) shows the model overfits this small dataset, which is expected for a 3.8M-parameter model trained for 10,000 steps on Tiny Shakespeare. Adding dropout, early stopping on validation loss, or more data are natural next steps.

Loss does not decrease monotonically because each step uses a randomly sampled batch.

## Sample Generation

Prompt:

```text
ROMEO:
```

Sample output from the trained GQA model:

```text
ROMEO:
So, sometimes cannot this young modest girl: our knowledge
h not a lonted and at iddeed from consul:
How canst thou, or thou beest, Romeo:
There lie shall, as yours, perforce hath appear,
Which I'll figght again, methought, death,
Still cold again, and true Romeo;
This is the duke's of their perfection oQnwaring,
That will she, brave Bolingbroke, can her son,
But stirn you get thine here's some belonging;
```

The model is intentionally small and uses character-level tokenization, so the generated text still contains malformed words. The goal here is to study the architecture and training process rather than produce production-quality language generation.

## MHA vs GQA

The main architectural difference in this experiment is the number of key/value heads.

| | MHA | GQA |
|---|---:|---:|
| Query heads | 8 | 8 |
| Key heads | 8 | 2 |
| Value heads | 8 | 2 |
| Query heads per KV head | 1 | 4 |
| Head dimension | 32 | 32 |

I kept the model dimension, Transformer structure, dataset, and training pipeline the same so I could isolate the attention change.

## Next Experiments

I plan to extend the comparison with:

- Validation loss
- Parameter count
- GPU memory usage
- Inference time
- MHA vs GQA generation comparison
- BPE/subword tokenization
- Temperature and top-k sampling

## Tech Stack

- Python
- PyTorch
- CUDA
- Kaggle

## Notes

This is a small educational implementation inspired by the LLaMA 2 architecture. It is not the official LLaMA 2 implementation or a reproduction of Meta's pretrained models.

The purpose of the project is to understand the architecture by implementing, training, modifying, and comparing its components directly.
