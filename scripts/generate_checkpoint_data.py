#!/usr/bin/env python3
"""
Generate synthetic AI training checkpoint data for GPUCompress benchmarks.

Creates raw float32 binary files that mimic the statistical properties of
real neural network training artifacts:

  - Model weights:     Normal distribution, small magnitude (Kaiming/Xavier init)
  - Gradients:         Near-zero mean, noisy, heavy tails
  - Adam optimizer m:  Exponential moving average of gradients (smoother than grads)
  - Adam optimizer v:  EMA of squared gradients (positive, near-zero early training)
  - Activations (ReLU): 50% zeros (ReLU sparsity), half-normal positive values
  - Activations (GELU): Smooth, fewer exact zeros than ReLU
  - Embeddings:        High-dimensional, clustered, many repeated patterns
  - LayerNorm output:  Normalized (mean~0, std~1), very compressible
  - Attention logits:  Softmax input, sparse structure (causal mask → -inf regions)
  - Mixed precision:   FP16 weights stored as FP32 (quantization grid → high compressibility)

Each file is a flat float32 array of size N elements (default: 512x512x512 = ~512 MB).
These can be fed directly to the SDRBench generic_benchmark driver.

Usage:
    python3 scripts/generate_checkpoint_data.py [--size SIZE_MB] [--outdir DIR]

    # Default: 512 MB per tensor, output to data/sdrbench/ai_checkpoint/
    python3 scripts/generate_checkpoint_data.py

    # Smaller for quick testing
    python3 scripts/generate_checkpoint_data.py --size 64

    # Custom output
    python3 scripts/generate_checkpoint_data.py --size 256 --outdir /tmp/ckpt_data
"""

import argparse
import os
import numpy as np
import sys


def generate_weights(n, rng):
    """Model weights: Kaiming normal init (std = sqrt(2/fan_in)), fan_in ~ 768."""
    std = np.sqrt(2.0 / 768)
    return rng.normal(0, std, size=n).astype(np.float32)


def generate_gradients(n, rng):
    """Gradients: near-zero mean, heavy-tailed (Student's t, df=3)."""
    # Scale to realistic gradient magnitude (~1e-4)
    return (rng.standard_t(df=3, size=n) * 1e-4).astype(np.float32)


def generate_adam_m(n, rng):
    """Adam first moment (m): smoothed gradients, EMA with beta1=0.9."""
    # Simulate 100 steps of EMA on random gradients
    m = np.zeros(n, dtype=np.float64)
    beta1 = 0.9
    for _ in range(100):
        g = rng.standard_t(df=3, size=n) * 1e-4
        m = beta1 * m + (1 - beta1) * g
    return m.astype(np.float32)


def generate_adam_v(n, rng):
    """Adam second moment (v): EMA of squared gradients, always positive."""
    v = np.zeros(n, dtype=np.float64)
    beta2 = 0.999
    for _ in range(100):
        g = rng.standard_t(df=3, size=n) * 1e-4
        v = beta2 * v + (1 - beta2) * (g ** 2)
    return v.astype(np.float32)


def generate_relu_activations(n, rng):
    """ReLU output: ~50% exact zeros, rest half-normal."""
    pre = rng.normal(0, 1, size=n)
    return np.maximum(pre, 0).astype(np.float32)


def generate_gelu_activations(n, rng):
    """GELU output: smooth approximation, fewer exact zeros than ReLU."""
    x = rng.normal(0, 1, size=n).astype(np.float64)
    # GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    gelu = x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    return gelu.astype(np.float32)


def generate_embeddings(n, rng):
    """Token embeddings: clustered, d_model=768, vocab repeated patterns."""
    # Simulate embedding table: vocab_size rows of d_model dims
    d_model = 768
    vocab_size = min(50000, n // d_model)
    if vocab_size < 1:
        vocab_size = 1
    table = rng.normal(0, 0.02, size=(vocab_size, d_model)).astype(np.float32)
    # Repeat to fill n elements (simulates batch of looked-up embeddings)
    repeats = (n + vocab_size * d_model - 1) // (vocab_size * d_model)
    data = np.tile(table.ravel(), repeats)[:n]
    return data


def generate_layernorm_output(n, rng):
    """LayerNorm output: mean~0, std~1, very smooth and compressible."""
    data = rng.normal(0, 1, size=n).astype(np.float64)
    # Apply layernorm-like normalization in chunks of 768
    chunk = 768
    for i in range(0, n - chunk + 1, chunk):
        segment = data[i:i+chunk]
        data[i:i+chunk] = (segment - segment.mean()) / (segment.std() + 1e-5)
    return data.astype(np.float32)


def generate_attention_logits(n, rng):
    """Attention logits (pre-softmax): sparse, causal mask regions are -1e9."""
    # Simulate seq_len=512 attention matrix, with causal mask
    seq_len = 512
    mat_size = seq_len * seq_len
    n_mats = max(1, n // mat_size)
    parts = []
    remaining = n
    for _ in range(n_mats):
        sz = min(mat_size, remaining)
        # Random logits
        logits = rng.normal(0, 1, size=(seq_len, seq_len)).astype(np.float32)
        # Causal mask: upper triangle = -1e9 (very compressible constant)
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        logits[mask] = -1e9
        parts.append(logits.ravel()[:sz])
        remaining -= sz
        if remaining <= 0:
            break
    if remaining > 0:
        parts.append(rng.normal(0, 1, size=remaining).astype(np.float32))
    return np.concatenate(parts)[:n]


def generate_fp16_weights(n, rng):
    """FP16 weights stored as FP32: quantization grid creates high compressibility."""
    w = rng.normal(0, np.sqrt(2.0 / 768), size=n).astype(np.float16)
    return w.astype(np.float32)  # FP16 → FP32 preserves the quantized grid


GENERATORS = {
    "weights":           ("Model weights (Kaiming init)",           generate_weights),
    "gradients":         ("Gradients (heavy-tailed, near-zero)",    generate_gradients),
    "adam_m":            ("Adam 1st moment (smoothed gradients)",    generate_adam_m),
    "adam_v":            ("Adam 2nd moment (squared grad EMA)",      generate_adam_v),
    "relu_activations":  ("ReLU activations (50% sparse)",          generate_relu_activations),
    "gelu_activations":  ("GELU activations (smooth)",              generate_gelu_activations),
    "embeddings":        ("Token embeddings (clustered, repeated)",  generate_embeddings),
    "layernorm_output":  ("LayerNorm output (normalized)",          generate_layernorm_output),
    "attention_logits":  ("Attention logits (causal mask, sparse)",  generate_attention_logits),
    "fp16_weights":      ("FP16-as-FP32 weights (quantized grid)",  generate_fp16_weights),
}


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic AI checkpoint data")
    parser.add_argument("--size", type=int, default=512,
                        help="Size per tensor in MB (default: 512)")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory (default: data/sdrbench/ai_checkpoint/)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    # Resolve output directory
    if args.outdir:
        outdir = args.outdir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        outdir = os.path.join(os.path.dirname(script_dir),
                              "data", "sdrbench", "ai_checkpoint")

    n_elements = args.size * 1024 * 1024 // 4  # float32 = 4 bytes
    total_bytes = n_elements * 4
    os.makedirs(outdir, exist_ok=True)

    # Compute dims: find cube root or use flat 1D
    side = int(round(n_elements ** (1.0/3.0)))
    # Adjust to make it exact
    while side * side * side < n_elements:
        side += 1
    while side * side * side > n_elements and side > 1:
        side -= 1
    if side * side * side == n_elements:
        dims_str = f"{side},{side},{side}"
    else:
        # Fall back to 2D: find closest rectangle
        side2d = int(round(n_elements ** 0.5))
        while n_elements % side2d != 0 and side2d > 1:
            side2d -= 1
        other = n_elements // side2d
        dims_str = f"{side2d},{other}"

    print(f"Generating synthetic AI checkpoint data")
    print(f"  Size per tensor : {args.size} MB ({n_elements:,} float32 elements)")
    print(f"  Dims            : {dims_str}")
    print(f"  Output          : {outdir}")
    print(f"  Seed            : {args.seed}")
    print(f"  Tensors         : {len(GENERATORS)}")
    print()

    rng = np.random.default_rng(args.seed)

    for name, (desc, gen_fn) in GENERATORS.items():
        fname = f"{name}.f32"
        fpath = os.path.join(outdir, fname)

        sys.stdout.write(f"  {name:25s} {desc:45s} ... ")
        sys.stdout.flush()

        data = gen_fn(n_elements, rng)
        assert data.dtype == np.float32
        assert data.shape == (n_elements,), f"Expected {n_elements}, got {data.shape}"

        data.tofile(fpath)
        file_mb = os.path.getsize(fpath) / (1024 * 1024)
        sys.stdout.write(f"{file_mb:.0f} MB\n")

    # Write metadata
    meta_path = os.path.join(outdir, "README.txt")
    with open(meta_path, "w") as f:
        f.write(f"Synthetic AI Checkpoint Data for GPUCompress Benchmarks\n")
        f.write(f"Generated by: scripts/generate_checkpoint_data.py\n")
        f.write(f"Size per tensor: {args.size} MB ({n_elements:,} float32)\n")
        f.write(f"Dims: {dims_str}\n")
        f.write(f"Seed: {args.seed}\n\n")
        f.write(f"Tensors:\n")
        for name, (desc, _) in GENERATORS.items():
            f.write(f"  {name}.f32 — {desc}\n")
        f.write(f"\nUsage with SDRBench:\n")
        f.write(f"  ./build/generic_benchmark weights.nnwt \\\n")
        f.write(f"    --data-dir {outdir} --dims {dims_str} --ext .f32 \\\n")
        f.write(f"    --chunk-mb 4 --name ai_checkpoint\n")

    print(f"\nDone. {len(GENERATORS)} tensors written to {outdir}")
    print(f"Dims for SDRBench: --dims {dims_str}")


if __name__ == "__main__":
    main()
