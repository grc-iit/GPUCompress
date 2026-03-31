#!/usr/bin/env python3
"""
Export real PyTorch model checkpoints as raw float32 binary files
for GPUCompress SDRBench benchmarks.

Exports model weights, optimizer state (Adam moments), and optionally
gradients from pretrained models. Each tensor is saved as a flat .f32 file
compatible with the generic_benchmark driver.

Supported models:
  - resnet50:  ResNet-50 (25.6M params, ~98 MB weights)
  - resnet152: ResNet-152 (60.2M params, ~230 MB weights)
  - vgg16:     VGG-16 (138M params, ~528 MB weights)

For each model, exports:
  - Individual layer tensors (conv1.weight.f32, fc.weight.f32, etc.)
  - Concatenated full checkpoint (all_weights.f32, adam_m.f32, adam_v.f32)
  - Fake gradients (one backward pass on random data)

Usage:
    python3 scripts/export_pytorch_checkpoints.py [--model resnet50] [--outdir DIR]

    # Default: ResNet-50
    python3 scripts/export_pytorch_checkpoints.py

    # VGG-16 (large, ~528 MB)
    python3 scripts/export_pytorch_checkpoints.py --model vgg16

    # Custom output
    python3 scripts/export_pytorch_checkpoints.py --model resnet152 --outdir /tmp/ckpt
"""

import argparse
import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models


def get_model(name):
    """Load a pretrained model."""
    factory = {
        "resnet50": lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
        "resnet152": lambda: models.resnet152(weights=models.ResNet152_Weights.DEFAULT),
        "vgg16": lambda: models.vgg16(weights=models.VGG16_Weights.DEFAULT),
    }
    if name not in factory:
        print(f"Unknown model: {name}. Available: {list(factory.keys())}")
        sys.exit(1)
    print(f"Loading pretrained {name}...")
    model = factory[name]()
    model.eval()
    return model


def export_tensor(tensor, path):
    """Save a PyTorch tensor as raw float32 binary."""
    data = tensor.detach().float().cpu().contiguous().numpy()
    data.tofile(path)
    return data.size, os.path.getsize(path)


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch checkpoints as raw .f32")
    parser.add_argument("--model", type=str, default="resnet50",
                        help="Model name: resnet50, resnet152, vgg16")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--adam-steps", type=int, default=50,
                        help="Simulated Adam optimizer steps (default: 50)")
    args = parser.parse_args()

    # Resolve output
    if args.outdir:
        outdir = args.outdir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        outdir = os.path.join(os.path.dirname(script_dir),
                              "data", "sdrbench", f"pytorch_{args.model}")
    os.makedirs(outdir, exist_ok=True)

    model = get_model(args.model)
    state_dict = model.state_dict()

    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    total_mb = total_params * 4 / (1024 * 1024)
    print(f"  Parameters: {total_params:,} ({total_mb:.1f} MB as float32)")
    print(f"  Layers: {len(state_dict)}")
    print(f"  Output: {outdir}")
    print()

    # ── Export individual layer tensors ──
    print("Exporting individual layers...")
    layer_sizes = {}
    for name, param in state_dict.items():
        safe_name = name.replace(".", "_")
        path = os.path.join(outdir, f"layer_{safe_name}.f32")
        n_elem, n_bytes = export_tensor(param, path)
        layer_sizes[name] = n_elem
        if n_bytes > 1024 * 1024:
            print(f"  {name:50s} {list(param.shape)} → {n_bytes/1024/1024:.1f} MB")

    # ── Export concatenated full checkpoint ──
    print("\nExporting concatenated tensors...")

    # All weights as one flat array
    all_weights = torch.cat([p.detach().float().flatten() for p in model.parameters()])
    path = os.path.join(outdir, "all_weights.f32")
    n_elem, n_bytes = export_tensor(all_weights, path)
    print(f"  all_weights.f32          {n_elem:>12,} elements  {n_bytes/1024/1024:>8.1f} MB")

    # ── Simulate Adam optimizer state ──
    print(f"\nSimulating {args.adam_steps} Adam optimizer steps...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Run a few forward/backward passes on random data to populate optimizer state
    model.train()
    for step in range(args.adam_steps):
        # Random input batch (batch=4, 3x224x224 for image models)
        x = torch.randn(4, 3, 224, 224)
        try:
            out = model(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        except Exception as e:
            if step == 0:
                print(f"  Warning: forward pass failed ({e}), using random optimizer state")
                # Manually populate optimizer state with realistic values
                for group in optimizer.param_groups:
                    for p in group['params']:
                        state = optimizer.state[p]
                        state['step'] = torch.tensor(float(args.adam_steps))
                        state['exp_avg'] = torch.randn_like(p) * 1e-4
                        state['exp_avg_sq'] = torch.rand_like(p) * 1e-6
            break

        if (step + 1) % 10 == 0:
            print(f"  Step {step+1}/{args.adam_steps}")

    # Export Adam first moment (m) and second moment (v)
    m_tensors = []
    v_tensors = []
    for group in optimizer.param_groups:
        for p in group['params']:
            if p in optimizer.state:
                state = optimizer.state[p]
                m_tensors.append(state['exp_avg'].detach().float().flatten())
                v_tensors.append(state['exp_avg_sq'].detach().float().flatten())

    if m_tensors:
        adam_m = torch.cat(m_tensors)
        path = os.path.join(outdir, "adam_m.f32")
        n_elem, n_bytes = export_tensor(adam_m, path)
        print(f"  adam_m.f32               {n_elem:>12,} elements  {n_bytes/1024/1024:>8.1f} MB")

        adam_v = torch.cat(v_tensors)
        path = os.path.join(outdir, "adam_v.f32")
        n_elem, n_bytes = export_tensor(adam_v, path)
        print(f"  adam_v.f32               {n_elem:>12,} elements  {n_bytes/1024/1024:>8.1f} MB")

    # ── Export gradients from last backward pass ──
    print("\nExporting gradients...")
    model.train()
    optimizer.zero_grad()
    x = torch.randn(4, 3, 224, 224)
    try:
        out = model(x)
        loss = out.sum()
        loss.backward()
    except:
        pass

    grad_tensors = []
    for p in model.parameters():
        if p.grad is not None:
            grad_tensors.append(p.grad.detach().float().flatten())

    if grad_tensors:
        all_grads = torch.cat(grad_tensors)
        path = os.path.join(outdir, "gradients.f32")
        n_elem, n_bytes = export_tensor(all_grads, path)
        print(f"  gradients.f32            {n_elem:>12,} elements  {n_bytes/1024/1024:>8.1f} MB")

    # ── Compute dims for SDRBench ──
    # Find the closest cube root for the concatenated tensors
    n = all_weights.numel()
    side = int(round(n ** (1.0/3.0)))
    # Pad to exact cube if close
    while side ** 3 < n:
        side += 1

    # For non-cube sizes, use 2D
    if abs(side ** 3 - n) / n < 0.01:
        # Close enough to cube, pad
        dims_str = f"{side},{side},{side}"
        padded_n = side ** 3
    else:
        # Use flat 1D with a factored 2D shape
        s = int(np.sqrt(n))
        while n % s != 0 and s > 1:
            s -= 1
        dims_str = f"{s},{n // s}"
        padded_n = n

    # Write metadata
    meta_path = os.path.join(outdir, "README.txt")
    with open(meta_path, "w") as f:
        f.write(f"PyTorch {args.model} Checkpoint Data for GPUCompress Benchmarks\n")
        f.write(f"Generated by: scripts/export_pytorch_checkpoints.py\n")
        f.write(f"Model: {args.model} ({total_params:,} parameters)\n")
        f.write(f"Adam steps: {args.adam_steps}\n\n")
        f.write(f"Files:\n")
        for fname in sorted(os.listdir(outdir)):
            if fname.endswith('.f32'):
                sz = os.path.getsize(os.path.join(outdir, fname))
                f.write(f"  {fname:50s} {sz/1024/1024:>8.1f} MB\n")
        f.write(f"\nFor SDRBench use the concatenated files:\n")
        f.write(f"  all_weights.f32, adam_m.f32, adam_v.f32, gradients.f32\n")

    # Summary
    print(f"\n{'='*60}")
    print(f"  {args.model} checkpoint exported to {outdir}")
    print(f"{'='*60}")
    total_files = len([f for f in os.listdir(outdir) if f.endswith('.f32')])
    total_disk = sum(os.path.getsize(os.path.join(outdir, f))
                     for f in os.listdir(outdir) if f.endswith('.f32'))
    print(f"  Files:      {total_files}")
    print(f"  Total size: {total_disk/1024/1024:.1f} MB")
    print(f"  Concat dims for SDRBench: --dims {dims_str}")
    print()
    print(f"  Run with SDRBench:")
    print(f"    BENCHMARKS=sdrbench SDR_DATASETS=pytorch_{args.model} \\")
    print(f"      bash benchmarks/benchmark.sh")


if __name__ == "__main__":
    main()
