# LLaMA 2 From Scratch in PyTorch

A mini LLaMA 2-style decoder-only language model implemented and trained from scratch using PyTorch.

This project was built to understand the internal architecture of modern large language models by implementing the core components directly instead of relying on a pretrained model.

## Architecture

The model implements:

- Token Embeddings
- RMSNorm
- Rotary Positional Embeddings (RoPE)
- Multi-Head Attention (MHA)
- Causal Attention Masking
- SwiGLU Feed-Forward Network
- Residual Connections
- Transformer Blocks
- Final RMSNorm
- Language Modeling Head
- Autoregressive Text Generation

## Model Architecture

```text
Token IDs
    ↓
Token Embedding
    ↓
┌─────────────────────────────┐
│ RMSNorm                     │
│    ↓                        │
│ Multi-Head Attention        │
│    ├── Q, K, V projections  │
│    ├── RoPE on Q and K      │
│    ├── Causal Mask          │
│    └── Output Projection    │
│    ↓                        │
│ Residual Connection         │
│    ↓                        │
│ RMSNorm                     │
│    ↓                        │
│ SwiGLU Feed-Forward Network │
│    ↓                        │
│ Residual Connection         │
└─────────────────────────────┘
    ↓
Repeat N Transformer Blocks
    ↓
Final RMSNorm
    ↓
LM Head
    ↓
Vocabulary Logits
```

## Training

The model is trained from random initialization on the Tiny Shakespeare dataset using autoregressive next-token prediction.

A character-level tokenizer is implemented directly in Python.

Training pipeline:

```text
Raw Shakespeare Text
        ↓
Character Tokenizer
        ↓
Token IDs
        ↓
Train / Validation Split
        ↓
[B, S] Training Batches
        ↓
LLaMA
        ↓
[B, S, V] Logits
        ↓
Cross-Entropy Loss
        ↓
Backpropagation
        ↓
AdamW
```

### Initial Training Results

```text
Step 0   | Loss: 4.3586
Step 100 | Loss: 2.3370
Step 200 | Loss: 2.0411
Step 300 | Loss: 1.8070
Step 400 | Loss: 1.7028
Step 500 | Loss: 1.6103
Step 600 | Loss: 1.6752
Step 700 | Loss: 1.4934
Step 800 | Loss: 1.5101
Step 900 | Loss: 1.4482
```

The decreasing loss demonstrates that the model successfully learns character-level language patterns from the training corpus.

## Autoregressive Generation

During inference, the model generates text one token at a time:

```text
Prompt
  ↓
Encode
  ↓
LLaMA
  ↓
Last-position logits
  ↓
Softmax
  ↓
Sample next token
  ↓
Append token
  ↓
Repeat
```

Example prompt:

```text
ROMEO:
```

After extended training, the model begins learning Shakespeare-like dialogue structure, speaker names, punctuation, line breaks, and word patterns.

## Planned Experiments

The project will be extended with:

- Multi-Head Attention (MHA) vs Grouped-Query Attention (GQA)
- Parameter count comparison
- Training and validation loss comparison
- GPU memory comparison
- Training/inference speed comparison
- BPE/subword tokenization
- Temperature and Top-K sampling

## Dataset

Tiny Shakespeare is used as a lightweight corpus for training the model from scratch.

## Technologies

- Python
- PyTorch
- CUDA
- Kaggle GPU

## Project Goal

The goal is not to reproduce the full-scale pretrained LLaMA 2 model, but to understand and experimentally investigate the architecture by implementing its core mechanisms from scratch.

## Disclaimer

This is an educational, small-scale implementation inspired by the LLaMA 2 architecture. It is not the official Meta LLaMA 2 implementation or pretrained model.
