# GPT From Scratch

This repository contains a **from-scratch implementation of a GPT-style language model** built using Python and PyTorch. The goal of this project is to deeply understand how GPT models work internally by implementing every core component manually — from tokenization to training and text generation — without relying on high-level transformer libraries.


---

## Project Overview

GPT (Generative Pretrained Transformer) is an **autoregressive, decoder-only transformer model** trained to predict the next token given a sequence of previous tokens. This project implements the complete GPT pipeline:

* Raw text → tokens
* Tokens → embeddings
* Embeddings → transformer blocks
* Transformer outputs → logits
* Logits → next-token prediction

---

## High-Level Architecture

```
Input Text
    ↓
Tokenization (Text → Token IDs)
    ↓
Token Embedding + Positional Embedding
    ↓
N × Transformer Decoder Blocks
    │   ├─ LayerNorm
    │   ├─ Masked Multi-Head Self Attention
    │   ├─ Residual Connection
    │   ├─ Feed Forward Network
    │   └─ Residual Connection
    ↓
Final LayerNorm
    ↓
Linear Projection (Vocabulary Size)
    ↓
Softmax → Next Token Probabilities
```

---

## Step-by-Step Components

## 1. Tokenization

Tokenization is the process of converting raw text into a sequence of **integer token IDs** that the model can process.

### Step 1: Vocabulary Construction

* The entire training corpus is scanned
* Unique characters or subwords are collected
* Each unique unit is assigned a unique integer ID

Example:

```
Vocabulary = {
  "a": 0,
  "b": 1,
  "c": 2,
  " ": 3,
  "<EOS>": 4
}
```

### Step 2: Encoding (Text → Tokens)

Input text:

```
"hello"
```

Encoding process:

```
h → 7
e → 4
l → 11
l → 11
o → 14
```

Result:

```
[7, 4, 11, 11, 14]
```

### Step 3: Decoding (Tokens → Text)

The inverse mapping converts model outputs back to readable text:

```
[7, 4, 11, 11, 14] → "hello"
```

### Step 4: Preparing Training Sequences

* A sliding window is used over token sequences
* Input tokens predict the **next token**

Example:

```
Input : [h, e, l, l]
Target: [e, l, l, o]
```

This creates the autoregressive learning objective.

---

## 2. Embeddings

Neural networks operate on vectors, not integers. Embeddings convert token IDs into dense vectors.

### Token Embeddings

* A learnable lookup table of shape:

```
[vocab_size, embedding_dim]
```

* Each token ID maps to a dense vector

Example:

```
Token ID: 14 → Vector: [0.12, -0.34, 0.88, ...]
```

### Positional Embeddings

Transformers do not inherently understand order. Positional embeddings inject sequence information.

* Learnable position vectors
* Added element-wise to token embeddings

```
Final Embedding = Token Embedding + Positional Embedding
```

---

## 3. Attention Mechanism

Attention allows the model to dynamically focus on relevant parts of the input sequence.

### Self Attention

Each token computes three vectors:

* **Query (Q)**
* **Key (K)**
* **Value (V)**

They are obtained via linear projections:

```
Q = X · Wq
K = X · Wk
V = X · Wv
```

### Attention Score Calculation

```
Attention(Q, K, V) = softmax((Q · Kᵀ) / √d_k) · V
```

This computes how much each token should attend to every other token.

### Causal Masking

GPT uses **masked self-attention**:

* Tokens can only attend to **previous tokens**
* Future tokens are masked using an upper triangular mask

This enforces autoregressive behavior.

---

### Multi-Head Attention

Instead of a single attention operation, GPT uses multiple heads.

* Input is split into multiple subspaces
* Each head performs self-attention independently
* Outputs are concatenated and projected

```
MultiHead(Q, K, V) = Concat(head₁, head₂, ..., headₙ) · Wo
```

This allows the model to capture different types of relationships simultaneously.

---

## 4. Transformer Block Components

Each transformer decoder block contains:

### 1. Layer Normalization

* Normalizes activations
* Improves training stability

### 2. Masked Multi-Head Self Attention

* Computes contextual representations
* Uses causal masking

### 3. Residual Connections

```
X = X + Attention(X)
X = X + FeedForward(X)
```

Residuals help with gradient flow in deep networks.

### 4. Feed Forward Network (FFN)

A two-layer MLP applied independently to each token:

```
FFN(x) = GELU(x · W1) · W2
```

---

## 5. GPT-Only Architecture

GPT uses a **decoder-only transformer architecture**:

* No encoder
* No cross-attention
* Only masked self-attention

Key characteristics:

* Autoregressive generation
* Left-to-right token prediction
* Same transformer block repeated N times

This design is optimized for language modeling and text generation.

---

## 6. Training Loop

### Step 1: Forward Pass

* Input token batch → embeddings
* Pass through transformer blocks
* Output logits of shape:

```
[batch_size, sequence_length, vocab_size]
```

### Step 2: Loss Computation

* Cross-entropy loss between predicted logits and target tokens
* Targets are shifted by one token

### Step 3: Backpropagation

* Compute gradients using autograd
* Update parameters using Adam optimizer

### Step 4: Iterative Training

* Loop over dataset for multiple epochs
* Periodically log loss

---

## 7. Text Generation

During inference:

1. Provide an initial prompt
2. Predict next token probabilities
3. Sample or take argmax
4. Append token to input
5. Repeat until max length or end token

This produces coherent, autoregressive text output.

---

## Summary

This repository demonstrates a complete GPT-style language model built from first principles, covering:

* Tokenization and data preparation
* Embedding and positional encoding
* Masked multi-head self-attention
* Transformer decoder blocks
* GPT-only architecture
* Training and text generation
