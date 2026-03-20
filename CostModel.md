# GPUCompress NN Cost Model — End-to-End Design Summary

## 1. Problem Overview

GPUCompress uses a neural network (15→128→128→4) to predict:

* Compression time (ct)
* Decompression time (dt)
* Compression ratio (ratio)
* PSNR

The goal is to select the best compression configuration (out of 32) using a **cost model**.

---

## 2. Original Cost Model

```
cost = ct + dt + data_size / (ratio * bandwidth)
```

### Issue

* IO term is extremely small at modern bandwidths
* Compute dominates
* Ratio becomes irrelevant

### Result

* Fast algorithms (Snappy/LZ4) always win
* High-ratio algorithms (Zstd, Bitcomp) never selected

---

## 3. First Fix: Add Ratio Utility

```
cost = ct + dt + ds/(ratio*bw) - α * log2(ratio)
```

### Improvement

* Ratio now contributes

### Problem

* Unit mismatch:

  * Time terms: ~10–30 ms
  * log term: ~0–10
* Requires fragile tuning of α

---

## 4. Key Insight

The issue is **mixing physical time and abstract utility**.

→ Need scale-invariant formulation

---

## 5. Mode-Based Optimization

Different users want different behaviors:

| Mode                 | Objective                      |
| -------------------- | ------------------------------ |
| Speed                | Minimize compute time          |
| Balanced             | Trade compute vs compression   |
| Ratio-first          | Maximize compression           |
| Throughput-per-ratio | Efficiency per compressed byte |

---

## 6. Better Formulation (Log-Space)

```
cost = α * log(ct + γ * dt)
     + β * log(IO_cost)
     - δ * log(ratio)
```

Where:

* IO_cost = data_size / (ratio * bw_eff)

---

## 7. Multi-Backend Extension

Multiple I/O paths:

* NVMe
* HDD
* DRAM
* Remote memory

Define:

```
1 / bw_eff = Σ (p_i / bw_i)
```

Final IO term:

```
IO_cost = data_size / (ratio * bw_eff)
```

### Benefit

* Automatically adapts to system usage
* Compression becomes more valuable when slow paths are used

---

## 8. Final Unified Cost Model

```
cost = α * log(ct + γ * dt)
     + β * log(data_size / (ratio * bw_eff))
     - δ * log(ratio)
```

---

## 9. Mode Presets

### Speed Mode

```
α=1, β=0, δ=0
```

### Balanced Mode

```
α=1, β=1, δ=0.5
```

### Ratio-First Mode

```
α=0.3, β=1, δ=1
```

### Throughput-per-Ratio Mode

```
α=1, β=0, δ=1
```

---

## 10. Interpretation

* α → importance of compute time
* β → importance of I/O system
* δ → importance of compression ratio
* γ → read vs write emphasis

---

## 11. CUDA Implementation

### Precompute (CPU)

```
float inv_bw_eff = 0.0f;
for (int i = 0; i < num_paths; i++) {
    inv_bw_eff += p[i] / bw[i];
}
```

### Kernel

```
float t = ct + gamma * dt;
float io = data_size * inv_bw_eff / ratio;

float cost =
    alpha * logf(t + 1e-6f) +
    beta  * logf(io + 1e-6f) -
    delta * logf(ratio + 1e-6f);
```

---

## 12. Key Advantages

* Scale invariant
* Works across chunk sizes and bandwidths
* Supports multiple storage backends
* Stable for SGD training
* Supports user-defined optimization policies

---

## 13. Final Takeaway

The cost model is no longer just a latency estimator.

It becomes a:

> **Policy-controlled, system-aware optimization function**

Combining:

* NN predictions
* System characteristics
* User intent

into one unified decision rule.
