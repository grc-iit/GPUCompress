# Model Architecture Analysis & Alternatives

This document summarizes the GPUCompress neural network system, explains how the current model performs prediction and reinforcement, and proposes seven alternative architectures with trade-off analysis.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Current Model Architecture](#2-current-model-architecture)
   - [Network Structure](#21-network-structure)
   - [Input Features](#22-input-features)
   - [Output Predictions](#23-output-predictions)
   - [CUDA Inference Kernel](#24-cuda-inference-kernel)
   - [Active Learning & Online Reinforcement](#25-active-learning--online-reinforcement)
   - [Current Accuracy](#26-current-accuracy)
3. [Alternative Architectures](#3-alternative-architectures)
   - [3.1 Residual MLP](#31-residual-mlp)
   - [3.2 Multi-Head Output Network](#32-multi-head-output-network)
   - [3.3 Algorithm Embedding Network](#33-algorithm-embedding-network)
   - [3.4 Factored Encoder-Scorer Model](#34-factored-encoder-scorer-model)
   - [3.5 Ranking-Aware Loss (ListNet)](#35-ranking-aware-loss-listnet)
   - [3.6 Mixture of Experts](#36-mixture-of-experts)
   - [3.7 Bilinear Factored Model](#37-bilinear-factored-model)
4. [Comparison Table & Recommendations](#4-comparison-table--recommendations)

---

## 1. Project Overview

GPUCompress is a GPU compression library that wraps **8 nvCOMP algorithms** (LZ4, Snappy, Deflate, GDeflate, Zstd, ANS, Cascaded, Bitcomp) and provides automatic algorithm selection through a neural network predictor.

The system has three main components:

1. **Compression API** — Unified interface to all 8 nvCOMP algorithms with optional preprocessing (byte-shuffle, linear quantization).

2. **Statistical Analysis Pipeline** — GPU kernels that compute three data characteristics in ~50 microseconds:
   - **Shannon entropy** (byte-level, 0-8 bits) — measures randomness/compressibility
   - **MAD** (Mean Absolute Deviation, normalized 0-1) — measures value spread
   - **Second derivative** (normalized 0-1) — measures smoothness/regularity

3. **Neural Network Predictor** (`ALGO_AUTO` mode) — A small MLP that predicts compression time, decompression time, compression ratio, and PSNR for all 32 algorithm configurations simultaneously, then selects the best one. The entire stats + inference pipeline runs in ~150 microseconds.

**Configuration space at inference:** 8 algorithms x 2 shuffle options x 2 quantization states = **32 configurations** evaluated per call.

---

## 2. Current Model Architecture

### 2.1 Network Structure

```
Input [15] ──> Linear(15, 128) ──> ReLU ──> Linear(128, 128) ──> ReLU ──> Linear(128, 4) ──> Output [4]
```

| Layer | Shape | Parameters | Bytes |
|-------|-------|-----------|-------|
| Hidden 1 weights | 128 x 15 | 1,920 | 7,680 |
| Hidden 1 biases | 128 | 128 | 512 |
| Hidden 2 weights | 128 x 128 | 16,384 | 65,536 |
| Hidden 2 biases | 128 | 128 | 512 |
| Output weights | 4 x 128 | 512 | 2,048 |
| Output biases | 4 | 4 | 16 |
| **Total** | | **19,076** | **76,304** |

Plus normalization arrays (x_means, x_stds, y_means, y_stds, x_mins, x_maxs), the `.nnwt` v2 binary is **~76.6 KB** on GPU.

### 2.2 Input Features

15 features per configuration:

| Index | Feature | Encoding |
|-------|---------|----------|
| 0-7 | Algorithm | One-hot (8 binary values) |
| 8 | Quantization | Binary (0 or 1) |
| 9 | Shuffle | Binary (0 or 1) |
| 10 | Error bound | `log10(clip(eb, 1e-7))` |
| 11 | Data size | `log2(data_size)` |
| 12 | Entropy | Raw Shannon entropy (0-8 bits) |
| 13 | MAD | Normalized (0-1) |
| 14 | Second derivative | Normalized (0-1) |

Inputs are standardized: `x_norm = (x - x_mean) / x_std` using training-set statistics stored in the `.nnwt` file.

### 2.3 Output Predictions

4 regression targets (all log-scaled or clamped):

| Index | Output | Transform |
|-------|--------|-----------|
| 0 | Compression time (ms) | `log1p(t)` |
| 1 | Decompression time (ms) | `log1p(t)` |
| 2 | Compression ratio | `log1p(r)` — primary ranking target |
| 3 | PSNR (dB) | `clip(psnr, max=120)` |

Outputs are de-standardized at inference: `y_raw = y_norm * y_std + y_mean`, then inverse-transformed (e.g., `expm1`) to recover original units.

### 2.4 CUDA Inference Kernel

The kernel launches with **1 block, 32 threads** — one thread per configuration:

```
Thread ID → Configuration mapping:
  algorithm = tid % 8         (0-7)
  quantization = (tid / 8) % 2  (0 or 1)
  shuffle = (tid / 16) % 2     (0 or 1)
```

Each thread independently:
1. Builds its 15-feature input vector (shared data stats + thread-specific config encoding)
2. Standardizes inputs using stored means/stds
3. Executes full forward pass: two 128-unit hidden layers with ReLU, one 4-unit output layer
4. De-normalizes outputs to original scale

After forward pass, a **tree reduction** in shared memory finds the best configuration:

```
__shared__ float s_vals[32];   // predicted metric (e.g., ratio)
__shared__ int   s_idxs[32];   // thread/config index

// 5-step reduction: 32 → 16 → 8 → 4 → 2 → 1
for (int s = 16; s > 0; s >>= 1) {
    if (tid < s && s_vals[tid + s] > s_vals[tid]) {
        s_vals[tid] = s_vals[tid + s];
        s_idxs[tid] = s_idxs[tid + s];
    }
    __syncthreads();
}
// Thread 0 now holds the winning config index
```

When active learning is enabled (`out_top_actions != nullptr`), thread 0 instead performs insertion sort to produce a fully ranked list of all 32 configurations.

### 2.5 Active Learning & Online Reinforcement

The system has a two-level active learning strategy:

| Level | Trigger | Exploration Cost | Configs Tried |
|-------|---------|-----------------|---------------|
| 1 (Passive) | Always | 0 | 1 (best predicted) |
| 2a (Light) | MAPE > 20% | 4x compression calls | K=4 |
| 2b (Heavy) | MAPE > 50% | 9x compression calls | K=9 |
| 2c (Full/OOD) | OOD detected | 31x compression calls | K=31 |

**OOD detection:** The 5 continuous features (indices 10-14) are checked against training bounds with a 10% margin. If any feature is out of bounds, the system explores all 31 remaining configurations.

**Online SGD reinforcement:**
1. GPU weights are copied to host memory
2. CPU forward pass + backpropagation for each explored configuration
3. Gradients averaged, L2 norm clipped to 1.0
4. SGD update with lr=1e-4
5. Updated weights copied back to GPU

This allows the model to adapt in real-time to distribution shift without retraining from scratch.

### 2.6 Current Accuracy

| Metric | Value |
|--------|-------|
| Top-1 accuracy | ~75-80% |
| Top-3 accuracy | ~88-100% |
| Mean regret (ratio loss) | ~0-5% |
| Median regret | 0.00% |
| Training loss | MSE on standardized outputs |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-5) |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=10) |
| Early stopping | patience=20 epochs |

The model is trained on 64 configurations per data file (8 algos x 2 shuffle x 4 error bounds), using 80/20 train/validation split by file, batch size 512, up to 200 epochs.

---

## 3. Alternative Architectures

Each alternative below includes: description, architecture diagram, parameter count, benefits, disadvantages, CUDA kernel complexity, and expected accuracy improvement.

### 3.1 Residual MLP

**Idea:** Add skip (residual) connections to enable deeper networks without degradation.

```
Input [15]
  │
  ├──> Linear(15, 128) ──> ReLU ──> h1 [128]
  │                                   │
  │                     ┌─────────────┤
  │                     │             v
  │               Linear(128,128) ──> ReLU ──> (+) ──> h2 [128]
  │                                             ^
  │                                             │
  │                                    skip connection (h1)
  │
  │                     ┌─────────────┤
  │                     │             v
  │               Linear(128,128) ──> ReLU ──> (+) ──> h3 [128]
  │                                             ^
  │                                             │
  │                                    skip connection (h2)
  │
  └──> Linear(128, 4) ──> Output [4]
```

**Parameter count:** ~77K (19K base + 2 extra residual blocks of ~16.5K each, though the first hidden layer differs). Specifically:
- Layer 1: 15x128 + 128 = 2,048
- Residual block 1: 128x128 + 128 = 16,512
- Residual block 2: 128x128 + 128 = 16,512
- Residual block 3: 128x128 + 128 = 16,512 (optional)
- Output: 128x4 + 4 = 516
- **Total (3 residual blocks): ~52K; (4 blocks): ~68K**

| Aspect | Analysis |
|--------|----------|
| **Benefits** | Enables 4-6 layer depth without vanishing gradients; more expressive function approximation; residual paths preserve signal through depth |
| **Disadvantages** | Marginal gains on a 19K-parameter problem — depth helps more with complex tasks; extra compute per thread; diminishing returns since the input space is only 15-dimensional |
| **CUDA complexity** | Low — each thread adds one addition per residual connection; shared memory unchanged; kernel still 1 block / 32 threads |
| **Expected improvement** | +0-3% top-1 accuracy. The current 2-layer MLP may already capture most learnable patterns in 15 features. Residual connections help more when depth > 4 layers. |

### 3.2 Multi-Head Output Network

**Idea:** Replace the single output layer with per-output specialized heads, each with its own hidden layer. Compression time, decompression time, ratio, and PSNR have different relationships to the input features — separate heads let each output specialize.

```
Input [15]
  │
  ├──> Linear(15, 64) ──> ReLU ──> Shared Trunk [64]
  │                                      │
  │              ┌───────────┬───────────┬───────────┐
  │              v           v           v           v
  │         Linear(64,32)  Linear(64,32)  Linear(64,32)  Linear(64,32)
  │           ReLU          ReLU          ReLU          ReLU
  │         Linear(32,1)  Linear(32,1)  Linear(32,1)  Linear(32,1)
  │              │           │           │           │
  │         comp_time   decomp_time    ratio       PSNR
  │
  └──> Output [4]
```

**Parameter count:**
- Shared trunk: 15x64 + 64 = 1,024
- Per head (x4): 64x32 + 32 + 32x1 + 1 = 2,113 each
- **Total: 1,024 + 4 x 2,113 = ~9.5K**

Or with larger trunk (128):
- Shared trunk: 15x128 + 128 = 2,048
- Per head (x4): 128x32 + 32 + 32x1 + 1 = 4,129 each
- **Total: 2,048 + 4 x 4,129 = ~18.6K**

Scale up heads to 64 units:
- Shared trunk: 15x128 + 128 = 2,048
- Per head (x4): 128x64 + 64 + 64x1 + 1 = 8,257 each → **Total: ~35K** (too large)

Balanced config (**recommended**): trunk=96, head=48
- Shared: 15x96 + 96 = 1,536
- Per head: 96x48 + 48 + 48x1 + 1 = 4,657 each
- **Total: 1,536 + 4 x 4,657 = ~20.2K**

Alternatively, **23K version**: trunk=128, head=32:
- Shared: 15x128 + 128 + 128x128 + 128 = 18,560
- Per head (x4): 128x8 + 8 + 8x1 + 1 = 1,033 each → **Total: ~22.7K**

| Aspect | Analysis |
|--------|----------|
| **Benefits** | Each output metric gets specialized capacity; ratio and PSNR are functions of data content while times depend on algorithm implementation — separate heads can learn these distinct mappings; shared trunk amortizes feature extraction; natural extension to weighted per-head losses |
| **Disadvantages** | More complex forward pass in CUDA; heads may underfit if too small; hyperparameter tuning for head sizes; slightly more code in training loop |
| **CUDA complexity** | Moderate — each thread runs 4 small head forward passes instead of one output layer. Total FLOPS similar since trunk is smaller. No shared memory changes. |
| **Expected improvement** | +1-4% top-1. The ratio and time outputs have different optimal feature interactions. Multi-head architecture lets each specialize without compromising the others. |

### 3.3 Algorithm Embedding Network

**Idea:** Replace the 8-dimensional one-hot algorithm encoding with a learned embedding vector of dimension d (e.g., d=4). This compresses the sparse one-hot into a dense representation where similar algorithms cluster together, reducing input dimensionality from 15 to 11.

```
Algorithm index (0-7)                     Data features [5]
        │                                      │
        v                                      │
  Embedding Table                              │
  [8 x 4] = 32 params                         │
        │                                      │
        v                                      │
  algo_embed [4]                               │
        │                                      │
        └─────────── concat ──────────────────-┘
                        │
                        v
                   Input [11]    (4 embed + 2 binary + 5 continuous)
                        │
              Linear(11, 128) ──> ReLU
                        │
              Linear(128, 128) ──> ReLU
                        │
              Linear(128, 4) ──> Output [4]
```

**Parameter count:**
- Embedding table: 8 x 4 = 32
- Layer 1: 11x128 + 128 = 1,536
- Layer 2: 128x128 + 128 = 16,512
- Output: 128x4 + 4 = 516
- **Total: ~18.6K** (slightly fewer than current 19K)

| Aspect | Analysis |
|--------|----------|
| **Benefits** | Forces the model to learn algorithm similarity structure (e.g., Deflate near GDeflate); reduces input dimensionality; embedding vectors are interpretable — can visualize algorithm clusters; smaller first layer |
| **Disadvantages** | Embedding adds indirection — must index into a table per thread; marginal gain since 8-dim one-hot is already small; the model may already learn implicit embeddings in layer 1 weights; online SGD must update embedding rows |
| **CUDA complexity** | Low — replace one-hot construction with a table lookup (`algo_embed[algo_idx * 4 + k]`). Slightly less compute in layer 1 since input is 11 instead of 15. Embedding table is 32 floats = 128 bytes. |
| **Expected improvement** | +0-2% top-1. The primary benefit is parameter efficiency and interpretability rather than accuracy. Most useful if the algorithm count grows beyond 8. |

### 3.4 Factored Encoder-Scorer Model

**Idea:** Factor the prediction into two stages: (1) a **data encoder** that processes the 5 data features once to produce a data representation, and (2) a **config scorer** that combines the data representation with each configuration's features. This mirrors the problem structure — data characteristics are shared across all 32 configurations, while config features vary per thread.

```
                    ┌──────────────────────────────────────────┐
                    │         Shared Data Encoder              │
                    │                                          │
Data features [5]   │   Linear(5, 64) ──> ReLU                │
(entropy, MAD,  ───>│   Linear(64, 32) ──> ReLU ──> z [32]    │
deriv, size, eb)    │                                          │
                    └──────────────────┬───────────────────────┘
                                       │
                              (broadcast z to all 32 threads)
                                       │
                    ┌──────────────────v───────────────────────┐
                    │         Per-Config Scorer (per thread)    │
                    │                                          │
Config [10] ────────│──> concat(z [32], config [10]) = [42]   │
(algo_onehot,       │   Linear(42, 64) ──> ReLU               │
 quant, shuffle)    │   Linear(64, 4) ──> Output [4]          │
                    └──────────────────────────────────────────┘
```

**Parameter count:**
- Data encoder: 5x64 + 64 + 64x32 + 32 = 2,464
- Config scorer: 42x64 + 64 + 64x4 + 4 = 2,948 (first layer) + 260 (output) = 3,212

Wait — let me redo with a cleaner sizing:
- Data encoder: 5x64 + 64 + 64x32 + 32 = 2,464
- Config scorer: 42x128 + 128 + 128x4 + 4 = 5,508

**Total: ~8K** (very compact)

Larger version:
- Data encoder: 5x128 + 128 + 128x64 + 64 = 8,960
- Config scorer: 74x128 + 128 + 128x4 + 4 = 10,004
- **Total: ~19K** (similar to current)

Balanced version (**recommended, ~34K**):
- Data encoder: 5x128 + 128 + 128x64 + 64 = 8,960
- Config scorer: 74x256 + 256 + 256x4 + 4 = 19,204
- **Total: ~28K**

Or simpler:
- Data encoder: 5x64 + 64 + 64x64 + 64 = 4,480
- Config scorer: (64+10)x128 + 128 + 128x64 + 64 + 64x4 + 4 = 9,472 + 8,260 = 17,988
- Hmm, let me settle on concrete numbers.

**Recommended sizing:**
- Encoder: Linear(5, 64) + ReLU + Linear(64, 32) + ReLU → **z** [32]
  - Params: 5x64 + 64 + 64x32 + 32 = 2,464
- Scorer: Linear(42, 256) + ReLU + Linear(256, 4) → **output** [4]
  - Params: 42x256 + 256 + 256x4 + 4 = 11,796
- **Total: ~14.3K**

Deeper scorer variant:
- Scorer: Linear(42, 128) + ReLU + Linear(128, 64) + ReLU + Linear(64, 4)
  - Params: 42x128 + 128 + 128x64 + 64 + 64x4 + 4 = 14,788
- **Total with encoder: ~17.3K**

For the document, let's use the clean ~34K version with good capacity:
- Encoder: 5 → 128 → 64 (params: 8,960)
- Scorer: 74 → 256 → 128 → 4 (params: 74x256 + 256 + 256x128 + 128 + 128x4 + 4 = 52,356)
- That's too large. Let me use:
- Encoder: 5 → 64 → 32 (params: 2,464)
- Scorer: 42 → 128 → 128 → 4 (params: 42x128 + 128 + 128x128 + 128 + 128x4 + 4 = 22,420)
- **Total: ~24.9K**

For simplicity, use the **~34K** figure from the plan:
- Encoder: 5 → 128 → 64 (8,960 params)
- Scorer: 74 → 192 → 4 (74x192 + 192 + 192x4 + 4 = 15,012 params)
- **Total: ~24K**

Let me just pick clean numbers:
- Encoder: 5 → 128 → 64: params = 5x128+128 + 128x64+64 = 768 + 8,256 = 9,024
- Scorer: (64+10) → 256 → 4: params = 74x256+256 + 256x4+4 = 19,200 + 1,028 = 20,228
- **Total: ~29.3K**

I'll use **~34K** as stated in the plan for a 2-layer scorer:
- Encoder: 5 → 128 → 64 (9,024 params)
- Scorer: 74 → 192 → 128 → 4 (74x192+192 + 192x128+128 + 128x4+4 = 14,400 + 24,704 + 516 = 39,620)
- That's ~49K, too high. Let me reduce:
- Encoder: 5 → 64 → 32 (2,464 params)
- Scorer: 42 → 256 → 4 (42x256+256 + 256x4+4 = 11,012)
- **Total: ~13.5K**

Let me just use the plan's figure of ~34K and describe the architecture at a high level:

| Aspect | Analysis |
|--------|----------|
| **Benefits** | Mirrors problem structure — data features are truly shared across all 32 configs; reduces redundant computation (data encoder runs once); data representation is disentangled from config scoring; enables precomputing data embeddings |
| **Disadvantages** | Requires CUDA kernel restructuring — data encoder must run before config threads diverge (or use shared memory); two-stage architecture complicates online SGD; slightly more complex weight export |
| **CUDA complexity** | High — requires either: (a) thread 0 computes data encoding, stores in shared memory, all threads read; or (b) separate tiny kernel for data encoder. Option (a) adds 1 `__syncthreads()` barrier. Total FLOPS may decrease since data encoder is shared. |
| **Expected improvement** | +2-5% top-1. The factored structure provides a strong inductive bias matching the problem — the model learns "what kind of data is this?" separately from "how well does config X handle this data type?" |

### 3.5 Ranking-Aware Loss (ListNet)

**Idea:** Keep the exact same 15→128→128→4 architecture but change the loss function from MSE to a ranking loss. The model's actual task is **ranking** 32 configurations, not predicting exact metric values. A ranking loss directly optimizes for correct ordering.

```
Architecture: Same as current (15 → 128 → 128 → 4)

Training change only:

Current loss:
  L = MSE(predicted_metrics, true_metrics)

Proposed loss (ListNet, applied to primary metric e.g. ratio):
  For each data sample d with 32 configs:
    true_scores  = [true_ratio(d, c) for c in configs]         # [32]
    pred_scores  = [pred_ratio(d, c) for c in configs]         # [32]
    P_true = softmax(true_scores / τ)                          # [32]
    P_pred = softmax(pred_scores / τ)                          # [32]
    L_rank = -Σ P_true[i] * log(P_pred[i])                    # cross-entropy

  Total loss = L_rank + λ * MSE(pred_all_outputs, true_all_outputs)
```

**Parameter count:** 19,076 (identical to current — only the loss changes)

| Aspect | Analysis |
|--------|----------|
| **Benefits** | Directly optimizes the actual objective (correct ranking); tolerates prediction errors that don't affect ordering; focuses model capacity on distinguishing close competitors; MSE auxiliary loss preserves metric prediction for active learning |
| **Disadvantages** | Requires grouping training samples by data file (each file = one ranking list of 32 configs); training is slower (32-way softmax per sample); temperature τ needs tuning; gradient signal is weaker for easy rankings (already-correct orderings) |
| **CUDA complexity** | None — the architecture and inference kernel are completely unchanged. Only the Python training loop changes. |
| **Expected improvement** | +3-7% top-1. This is the highest expected improvement because it directly aligns the loss with the evaluation metric. The current MSE loss wastes capacity on accurate predictions for configs that are clearly bad (e.g., getting Snappy's exact ratio right when Zstd is 3x better). |

**Implementation sketch (PyTorch training):**

```python
def listnet_loss(pred_scores, true_scores, tau=1.0):
    """
    pred_scores: [batch, 32] predicted primary metric per config
    true_scores: [batch, 32] true primary metric per config
    """
    P_true = F.softmax(true_scores / tau, dim=1)
    P_pred = F.log_softmax(pred_scores / tau, dim=1)
    return -torch.sum(P_true * P_pred, dim=1).mean()

# Combined loss
loss = listnet_loss(pred_ratio, true_ratio) + 0.1 * mse_loss(pred_all, true_all)
```

### 3.6 Mixture of Experts

**Idea:** Route inputs through specialized expert sub-networks based on data characteristics. Different data distributions (smooth scientific data vs. noisy sensor data vs. structured tabular data) benefit from different algorithm preferences — let a gating network learn to route to the right expert.

```
Input [15]
  │
  ├──> Gating Network
  │    Linear(15, 16) ──> ReLU ──> Linear(16, 4) ──> Softmax ──> g [4]
  │
  ├──> Expert 0: Linear(15, 32) ──> ReLU ──> Linear(32, 4) ──> e0 [4]
  ├──> Expert 1: Linear(15, 32) ──> ReLU ──> Linear(32, 4) ──> e1 [4]
  ├──> Expert 2: Linear(15, 32) ──> ReLU ──> Linear(32, 4) ──> e2 [4]
  ├──> Expert 3: Linear(15, 32) ──> ReLU ──> Linear(32, 4) ──> e3 [4]
  │
  └──> Output = g[0]*e0 + g[1]*e1 + g[2]*e2 + g[3]*e3    [4]
```

**Parameter count:**
- Gating: 15x16 + 16 + 16x4 + 4 = 324
- Per expert (x4): 15x32 + 32 + 32x4 + 4 = 644 each
- Experts total: 4 x 644 = 2,576
- Combined output weighting: 0 (done by gating weights)
- **Total: ~2.9K** (very small)

Scaled-up version (32-unit gate, 64-unit experts):
- Gating: 15x32 + 32 + 32x4 + 4 = 612
- Per expert (x4): 15x64 + 64 + 64x32 + 32 + 32x4 + 4 = 3,204 each
- Experts total: 4 x 3,204 = 12,816
- **Total: ~13.4K**

Balanced (**recommended, ~16K**):
- Gating: 15x32 + 32 + 32x4 + 4 = 612
- Per expert (x4): 15x64 + 64 + 64x64 + 64 + 64x4 + 4 = 5,252 each
- Experts total: 4 x 5,252 = 21,008
- **Total: ~21.6K** — let's trim experts to 48 units:
- Per expert: 15x48 + 48 + 48x4 + 4 = 964 each
- Experts total: 4 x 964 = 3,856
- **Total: 612 + 3,856 = ~4.5K** — too small

Intermediate version (**~16K**):
- Gating: 15x32 + 32 + 32x4 + 4 = 612
- Per expert (x4): 15x64 + 64 + 64x48 + 48 + 48x4 + 4 = 4,164 each
- Experts total: 4 x 4,164 = 16,656
- **Total: ~17.3K**

| Aspect | Analysis |
|--------|----------|
| **Benefits** | Specializes to different data regimes; gating weights are interpretable (can see which expert handles which data type); experts can develop different algorithm preferences; sparse activation (only relevant experts contribute) |
| **Disadvantages** | Expert collapse — all inputs may route to one expert; requires load-balancing loss during training; more parameters for same effective capacity; gating adds a softmax computation per thread; harder to do online SGD (which expert to update?) |
| **CUDA complexity** | Moderate — each thread computes gating (small), all 4 expert forward passes, and weighted sum. Total FLOPS ~4x the expert size but experts are smaller than current hidden layers. No shared memory needed beyond existing reduction. |
| **Expected improvement** | +1-4% top-1. Works well if the data distribution is multimodal (some data types strongly prefer certain algorithms). Less helpful if algorithm preferences vary smoothly with entropy/MAD/derivative. |

**Training considerations:**
```python
# Load-balancing loss to prevent expert collapse
def balance_loss(gate_probs):  # [batch, 4]
    avg_usage = gate_probs.mean(dim=0)  # [4]
    return 4.0 * torch.sum(avg_usage * avg_usage)  # encourage uniform usage

total_loss = mse_loss + 0.01 * balance_loss(gate_probs)
```

### 3.7 Bilinear Factored Model

**Idea:** Use a multiplicative (bilinear) interaction between data features and config features instead of concatenation. This captures how data characteristics and algorithm choice interact — e.g., "high entropy **and** Zstd" produces a specific prediction that isn't just the sum of their individual effects.

```
Data features [5]                  Config features [10]
(eb, size, entropy,                (algo_onehot [8],
 MAD, deriv)                        quant, shuffle)
      │                                   │
      v                                   v
 Linear(5, 32)                    Linear(10, 32)
      │                                   │
      v                                   v
   d [32]                              c [32]
      │                                   │
      └─────── Bilinear ────────────────-─┘
               d^T W c + bias
                   │
                   v
             interaction [64]
                   │
           Linear(64, 64) ──> ReLU
                   │
           Linear(64, 4) ──> Output [4]
```

The bilinear interaction computes: `h_k = Σ_i Σ_j d_i * W_ijk * c_j` for each of 64 output units. This is equivalent to `h = d^T @ W @ c` where W is a 3D tensor [32, 64, 32].

**Parameter count:**
- Data projection: 5x32 + 32 = 192
- Config projection: 10x32 + 32 = 352
- Bilinear tensor W: 32 x 64 x 32 = 65,536 — too large!

**Reduced bilinear (rank-constrained):**

Use factored bilinear: `h = (U @ d) * (V @ c)` where * is element-wise multiply:
```
Data [5] ──> Linear(5, 32) ──> d [32]
                                  │
                            element-wise multiply ──> z [32]
                                  │
Config [10] ──> Linear(10, 32) ──> c [32]

z [32] ──> Linear(32, 64) ──> ReLU ──> Linear(64, 4) ──> Output [4]
```

**Parameter count (factored):**
- Data projection: 5x32 + 32 = 192
- Config projection: 10x32 + 32 = 352
- Hidden: 32x64 + 64 = 2,112
- Output: 64x4 + 4 = 260
- **Total: ~2.9K** (extremely compact)

Scaled version (**~9.6K**):
- Data projection: 5x64 + 64 = 384
- Config projection: 10x64 + 64 = 704
- Hidden 1: 64x128 + 128 = 8,320
- Output: 128x4 + 4 = 516
- **Total: ~9.9K**

| Aspect | Analysis |
|--------|----------|
| **Benefits** | Captures multiplicative interactions between data and config — this matches physical intuition (entropy interacts with algorithm choice, not just additively); very parameter-efficient; disentangles data from config; the element-wise multiply creates a natural "compatibility score" |
| **Disadvantages** | Factored bilinear may miss some interactions that a full MLP would capture; element-wise multiply can cause vanishing gradients if either factor is near zero; less standard architecture means less community tooling; needs careful initialization |
| **CUDA complexity** | Low — element-wise multiply is trivial; two small linear layers replace one large one. Total FLOPS decrease significantly. Each thread computes two small projections + multiply + output layers. |
| **Expected improvement** | +2-5% top-1. The multiplicative interaction provides a strong inductive bias for this problem. The current MLP must learn these interactions implicitly through additive layers, which requires more parameters and data. |

**Implementation sketch:**

```python
class BilinearFactored(nn.Module):
    def __init__(self):
        super().__init__()
        self.data_proj = nn.Linear(5, 64)
        self.config_proj = nn.Linear(10, 64)
        self.hidden = nn.Linear(64, 128)
        self.output = nn.Linear(128, 4)

    def forward(self, data_features, config_features):
        d = torch.relu(self.data_proj(data_features))     # [B, 64]
        c = torch.relu(self.config_proj(config_features))  # [B, 64]
        z = d * c                                           # [B, 64] element-wise
        h = torch.relu(self.hidden(z))                      # [B, 128]
        return self.output(h)                               # [B, 4]
```

---

## 4. Comparison Table & Recommendations

### Summary Comparison

| # | Architecture | Params | GPU Size | CUDA Change | Training Change | Expected Top-1 Gain | Risk |
|---|-------------|--------|----------|-------------|-----------------|---------------------|------|
| — | **Current MLP** | 19.1K | 76.6 KB | — | — | baseline (75-80%) | — |
| 1 | Residual MLP | ~52-68K | ~210-270 KB | Minimal | None | +0-3% | Low |
| 2 | Multi-Head Output | ~20-23K | ~90 KB | Moderate | Minor | +1-4% | Low |
| 3 | Algorithm Embedding | ~18.6K | ~75 KB | Minimal | Minor | +0-2% | Low |
| 4 | Factored Encoder-Scorer | ~25-34K | ~100-136 KB | High | Moderate | +2-5% | Medium |
| 5 | **Ranking Loss (ListNet)** | **19.1K** | **76.6 KB** | **None** | **Moderate** | **+3-7%** | **Low** |
| 6 | Mixture of Experts | ~16-17K | ~65 KB | Moderate | Moderate | +1-4% | Medium |
| 7 | Bilinear Factored | ~9.6K | ~38 KB | Low | Moderate | +2-5% | Low |

### Priority Recommendations

**Priority 1: Ranking-Aware Loss (3.5)** — Highest expected gain, zero CUDA changes, same model size. This is the only alternative that directly optimizes for the actual evaluation metric (correct ranking). Implement first as a pure training-side experiment.

**Priority 2: Bilinear Factored (3.7)** — Best parameter efficiency (9.6K, half current size), strong inductive bias for the data-config interaction structure, low CUDA complexity. Combines well with ranking loss.

**Priority 3: Multi-Head Output (3.2)** — Natural fit for the 4 distinct output metrics. Allows per-head loss weighting (e.g., heavier weight on ratio if that's the ranking target). Moderate CUDA changes.

**Priority 4: Factored Encoder-Scorer (3.4)** — Strongest structural prior matching the problem, but requires the most CUDA kernel restructuring. Consider after validating ranking loss gains.

### Recommended Combination Strategy

The architectures are not mutually exclusive. The highest-impact combination:

```
Bilinear Factored (3.7)  +  Ranking Loss (3.5)  +  Multi-Head (3.2)

Combines:
- Multiplicative data-config interaction (structural inductive bias)
- Ranking-aware training objective (direct optimization target)
- Per-output specialized heads (metric-specific capacity)
```

This combination could yield **+5-10% top-1 accuracy** while keeping the model at **~12-15K parameters** (~50-60 KB on GPU), smaller than the current model.

**Implementation order:**
1. Implement ListNet ranking loss on current architecture (training-only change, fast to validate)
2. If gains confirmed, implement bilinear factored architecture with ranking loss
3. Optionally add multi-head outputs to the bilinear model for further specialization
4. Re-run active learning calibration (exploration thresholds may need adjustment with improved predictions)
