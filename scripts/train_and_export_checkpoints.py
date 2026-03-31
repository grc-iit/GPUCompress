#!/usr/bin/env python3
"""
Fine-tune a pretrained ViT model and export training checkpoints as raw .f32 files.

Produces evolving checkpoint data at multiple training stages, with 4 tensor
types per checkpoint: weights, adam_m (1st moment), adam_v (2nd moment), gradients.
Each .f32 file is a flat float32 array, compatible with the SDRBench
generic_benchmark.cu driver.

Files are named epoch-major (epoch01_adam_m.f32, epoch01_adam_v.f32, etc.)
so alphabetical sort = epoch order for NN-RL learning.

Usage:
    # Full ViT-Large run (~45-60 min on A100, generates ~37 GB)
    python3 scripts/train_and_export_checkpoints.py

    # Quick test with ViT-Base (~10 min, ~2.6 GB)
    python3 scripts/train_and_export_checkpoints.py --model vit_b_16 --epochs 5 --checkpoint-epochs 1,3,5

    # Custom output
    python3 scripts/train_and_export_checkpoints.py --outdir /tmp/ckpt --epochs 10
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T


def compute_dims_2d(n_elements):
    """Find a 2D factorization for SDRBench --dims. Returns (d0, d1) with d0*d1 >= n_elements."""
    # Try exact factorization first
    s = int(math.isqrt(n_elements))
    while s > 1 and n_elements % s != 0:
        s -= 1
    if n_elements % s == 0:
        return (s, n_elements // s)
    # No clean factor — pad to next value divisible by a reasonable factor
    target = n_elements
    while True:
        s = int(math.isqrt(target))
        while s > 1 and target % s != 0:
            s -= 1
        if s > 1:
            return (s, target // s)
        target += 1


def export_tensor_padded(tensor_list, path, target_elements):
    """Concatenate parameter tensors, pad to target_elements, save as .f32."""
    flat = torch.cat([p.detach().float().cpu().flatten() for p in tensor_list])
    n = flat.numel()
    if n < target_elements:
        flat = torch.cat([flat, torch.zeros(target_elements - n)])
    elif n > target_elements:
        flat = flat[:target_elements]
    flat.numpy().tofile(path)
    return os.path.getsize(path)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune ViT and export training checkpoints as .f32")
    parser.add_argument("--model", default="vit_l_16",
                        choices=["vit_l_16", "vit_b_16"],
                        help="Model: vit_l_16 (304M, ~1.16GB) or vit_b_16 (86M, ~0.33GB)")
    parser.add_argument("--dataset", default="cifar10",
                        help="Dataset (default: cifar10)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Total training epochs (default: 20)")
    parser.add_argument("--checkpoint-epochs", type=str, default="1,2,3,5,8,10,15,20",
                        help="Comma-separated epochs to export (default: 1,2,3,5,8,10,15,20)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="AdamW weight decay (default: 0.01)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory (default: data/sdrbench/vit_{model}_{dataset})")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Dataset cache root (default: data/)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true",
                        help="Use automatic mixed precision (faster training)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_epochs = sorted(set(int(e) for e in args.checkpoint_epochs.split(",")))
    if max(checkpoint_epochs) > args.epochs:
        print(f"Warning: max checkpoint epoch {max(checkpoint_epochs)} > total epochs {args.epochs}")
        checkpoint_epochs = [e for e in checkpoint_epochs if e <= args.epochs]

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_root = args.data_root or os.path.join(project_dir, "data")
    model_short = args.model.replace("vit_", "vit")  # vit_l_16 → vitl16
    if args.outdir:
        outdir = args.outdir
    else:
        outdir = os.path.join(project_dir, "data", "sdrbench",
                              f"vit_{args.model.split('_')[1]}_{args.dataset}")
    os.makedirs(outdir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Load model ──
    print(f"Loading pretrained {args.model}...")
    if args.model == "vit_l_16":
        weights = torchvision.models.ViT_L_16_Weights.DEFAULT
        model = torchvision.models.vit_l_16(weights=weights)
        hidden_dim = 1024
    elif args.model == "vit_b_16":
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        model = torchvision.models.vit_b_16(weights=weights)
        hidden_dim = 768
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Replace classification head for CIFAR-10 (10 classes)
    num_classes = 10
    model.heads.head = nn.Linear(hidden_dim, num_classes)

    # Unfreeze ALL layers (full fine-tuning)
    for p in model.parameters():
        p.requires_grad = True

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    param_mb = n_params * 4 / (1024 * 1024)

    # Compute padded dims
    d0, d1 = compute_dims_2d(n_params)
    target_elements = d0 * d1
    pad_elements = target_elements - n_params

    print(f"  Model       : {args.model}")
    print(f"  Parameters  : {n_params:,} ({param_mb:.1f} MB as float32)")
    print(f"  Padded dims : {d0} x {d1} = {target_elements:,} ({pad_elements} padding zeros)")
    print(f"  Device      : {device}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Checkpoints : {checkpoint_epochs}")
    print(f"  Output      : {outdir}")
    print()

    # ── Dataset ──
    transform_train = T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_val = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"Loading CIFAR-10 (auto-downloads if needed)...")
    cifar_root = os.path.join(data_root, "cifar10")
    train_ds = torchvision.datasets.CIFAR10(root=cifar_root, train=True,
                                             download=True, transform=transform_train)
    val_ds = torchvision.datasets.CIFAR10(root=cifar_root, train=False,
                                           download=True, transform=transform_val)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    steps_per_epoch = len(train_loader)
    print(f"  Train: {len(train_ds)} images, {steps_per_epoch} steps/epoch")
    print(f"  Val  : {len(val_ds)} images")
    print()

    # ── Optimizer + scheduler ──
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=args.amp)

    # ── Training loop ──
    print("=" * 60)
    print("  Training")
    print("=" * 60)

    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_start = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            with torch.amp.autocast(device_type="cuda", enabled=args.amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            epoch_correct += predicted.eq(labels).sum().item()
            epoch_total += images.size(0)

            if (batch_idx + 1) % 50 == 0:
                pct = 100.0 * (batch_idx + 1) / steps_per_epoch
                sys.stdout.write(f"\r  Epoch {epoch:2d}/{args.epochs} "
                                 f"[{pct:5.1f}%] loss={loss.item():.4f}")
                sys.stdout.flush()

        scheduler.step()

        epoch_time = time.time() - epoch_start
        train_acc = 100.0 * epoch_correct / epoch_total
        avg_loss = epoch_loss / epoch_total

        # Quick validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast(device_type="cuda", enabled=args.amp):
                    outputs = model(images)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += images.size(0)
        val_acc = 100.0 * val_correct / val_total

        print(f"\r  Epoch {epoch:2d}/{args.epochs}  "
              f"loss={avg_loss:.4f}  train_acc={train_acc:.1f}%  "
              f"val_acc={val_acc:.1f}%  time={epoch_time:.1f}s  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        # ── Export checkpoint if this is a checkpoint epoch ──
        if epoch in checkpoint_epochs:
            print(f"\n  >>> Exporting checkpoint at epoch {epoch}...")
            export_start = time.time()

            # 1. Weights
            path = os.path.join(outdir, f"epoch{epoch:02d}_weights.f32")
            sz = export_tensor_padded(list(model.parameters()), path, target_elements)
            print(f"      epoch{epoch:02d}_weights.f32     {sz/1024/1024:.1f} MB")

            # 2. Adam first moment (exp_avg)
            m_tensors = []
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p in optimizer.state and "exp_avg" in optimizer.state[p]:
                        m_tensors.append(optimizer.state[p]["exp_avg"])
                    else:
                        m_tensors.append(torch.zeros_like(p))
            path = os.path.join(outdir, f"epoch{epoch:02d}_adam_m.f32")
            sz = export_tensor_padded(m_tensors, path, target_elements)
            print(f"      epoch{epoch:02d}_adam_m.f32      {sz/1024/1024:.1f} MB")

            # 3. Adam second moment (exp_avg_sq)
            v_tensors = []
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p in optimizer.state and "exp_avg_sq" in optimizer.state[p]:
                        v_tensors.append(optimizer.state[p]["exp_avg_sq"])
                    else:
                        v_tensors.append(torch.zeros_like(p))
            path = os.path.join(outdir, f"epoch{epoch:02d}_adam_v.f32")
            sz = export_tensor_padded(v_tensors, path, target_elements)
            print(f"      epoch{epoch:02d}_adam_v.f32      {sz/1024/1024:.1f} MB")

            # 4. Gradients (run one extra forward+backward on a training batch)
            model.train()
            optimizer.zero_grad()
            images, labels = next(iter(train_loader))
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast(device_type="cuda", enabled=args.amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()

            grad_tensors = []
            for p in model.parameters():
                if p.grad is not None:
                    grad_tensors.append(p.grad)
                else:
                    grad_tensors.append(torch.zeros_like(p))
            path = os.path.join(outdir, f"epoch{epoch:02d}_gradients.f32")
            sz = export_tensor_padded(grad_tensors, path, target_elements)
            print(f"      epoch{epoch:02d}_gradients.f32   {sz/1024/1024:.1f} MB")

            optimizer.zero_grad()  # Clean up gradients
            export_time = time.time() - export_start
            print(f"      Export time: {export_time:.1f}s\n")

    total_time = time.time() - total_start

    # ── Write metadata ──
    dims_str = f"{d0},{d1}"
    meta_path = os.path.join(outdir, "README.txt")
    with open(meta_path, "w") as f:
        f.write(f"ViT Training Checkpoint Data for GPUCompress Benchmarks\n")
        f.write(f"Generated by: scripts/train_and_export_checkpoints.py\n\n")
        f.write(f"Model: {args.model} ({n_params:,} parameters, {param_mb:.1f} MB)\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Epochs: {args.epochs}, checkpoints at: {checkpoint_epochs}\n")
        f.write(f"Optimizer: AdamW (lr={args.lr}, wd={args.weight_decay})\n")
        f.write(f"Dims for SDRBench: --dims {dims_str}\n")
        f.write(f"Padded elements: {target_elements:,} ({pad_elements} zeros appended)\n")
        f.write(f"Total training time: {total_time:.0f}s\n\n")
        f.write(f"Files:\n")
        for fname in sorted(os.listdir(outdir)):
            if fname.endswith(".f32"):
                fsize = os.path.getsize(os.path.join(outdir, fname))
                f.write(f"  {fname:40s} {fsize/1024/1024:>8.1f} MB\n")
        f.write(f"\nUsage with SDRBench:\n")
        f.write(f"  BENCHMARKS=sdrbench SDR_DATASETS=vit_{args.model.split('_')[1]}_{args.dataset} \\\n")
        f.write(f"    CHUNK_MB=4 POLICIES=balanced VERIFY=0 bash benchmarks/benchmark.sh\n")

    # ── Summary ──
    n_files = len([f for f in os.listdir(outdir) if f.endswith(".f32")])
    total_disk = sum(os.path.getsize(os.path.join(outdir, f))
                     for f in os.listdir(outdir) if f.endswith(".f32"))

    print()
    print("=" * 60)
    print(f"  Checkpoint Export Complete")
    print("=" * 60)
    print(f"  Model         : {args.model} ({n_params:,} params)")
    print(f"  Training time : {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Final val_acc : {val_acc:.1f}%")
    print(f"  Files         : {n_files} .f32 files")
    print(f"  Total disk    : {total_disk/1024/1024/1024:.1f} GB")
    print(f"  Dims          : --dims {dims_str}")
    print(f"  Output        : {outdir}")
    print()
    print(f"  Add to run_sdr_pm_eval.sh:")
    ds_name = f"vit_{args.model.split('_')[1]}_{args.dataset}"
    print(f'    DS_SUBDIR[{ds_name}]="{os.path.basename(outdir)}"')
    print(f'    DS_DIMS[{ds_name}]="{dims_str}"')
    print(f'    DS_EXT[{ds_name}]=".f32"')
    print()
    print(f"  Run benchmark:")
    print(f"    BENCHMARKS=sdrbench SDR_DATASETS={ds_name} \\")
    print(f"      CHUNK_MB=4 POLICIES=balanced VERIFY=0 bash benchmarks/benchmark.sh")


if __name__ == "__main__":
    main()
