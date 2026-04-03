#!/usr/bin/env python3
"""
Fine-tune GPT-2 Small and export training checkpoints as raw .f32 files.

GPT-2 Small (124M params, 473 MB/file) produces fundamentally different
checkpoint data than vision transformers:
  - Token embedding matrix (50257×768 = 38.6M params) is near-incompressible
  - Causal attention weights have different gradient patterns
  - LLM checkpoints are among the hardest to compress losslessly

Usage:
    # Default: GPT-2 Small, 5 epochs, checkpoints at 1,2,3,5
    python3 scripts/train_gpt2_checkpoints.py

    # Quick test: 2 epochs
    python3 scripts/train_gpt2_checkpoints.py --epochs 2 --checkpoint-epochs 1,2
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
from torch.utils.data import DataLoader, Dataset

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset


class TextDataset(Dataset):
    """Tokenized text dataset for causal LM training."""
    def __init__(self, encodings, block_size):
        self.input_ids = []
        # Concatenate all tokens and split into blocks
        all_ids = []
        for ids in encodings:
            all_ids.extend(ids)
        for i in range(0, len(all_ids) - block_size, block_size):
            self.input_ids.append(torch.tensor(all_ids[i:i + block_size], dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        x = self.input_ids[idx]
        return x, x.clone()  # input = target for causal LM


def compute_dims_2d(n_elements):
    """Find 2D factorization for SDRBench --dims."""
    s = int(math.isqrt(n_elements))
    while s > 1 and n_elements % s != 0:
        s -= 1
    if n_elements % s == 0:
        return (s, n_elements // s)
    target = n_elements
    while True:
        s = int(math.isqrt(target))
        while s > 1 and target % s != 0:
            s -= 1
        if s > 1:
            return (s, target // s)
        target += 1


def export_tensor_padded(tensor_list, path, target_elements):
    """Concatenate tensors, pad to target_elements, save as .f32."""
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
        description="Fine-tune GPT-2 Small and export checkpoints as .f32")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--checkpoint-epochs", type=str, default="1,2,3,5")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=512,
                        help="Sequence length for training (default: 512)")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-accumulation", type=int, default=8,
                        help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--hdf5-direct", action="store_true",
                        help="Write checkpoints directly from GPU via HDF5 VOL (no CPU roundtrip)")
    parser.add_argument("--chunk-mb", type=int, default=4,
                        help="HDF5 chunk size in MB for --hdf5-direct (default: 4)")
    parser.add_argument("--error-bound", type=float, default=0.0,
                        help="Lossy error bound for --hdf5-direct (default: 0.0 = lossless)")
    parser.add_argument("--policy", type=str, default="balanced",
                        choices=["balanced", "ratio", "speed"],
                        help="NN cost model policy for --hdf5-direct (default: balanced)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark all compression algorithms at each checkpoint (requires --hdf5-direct)")
    parser.add_argument("--benchmark-configs", type=str, default=None,
                        help="Multi-config benchmark: 'chunk_mb:error_bound:outdir,...'")
    parser.add_argument("--sgd-lr", type=float, default=0.2,
                        help="SGD learning rate for online NN updates (default: 0.2)")
    parser.add_argument("--sgd-mape", type=float, default=0.10,
                        help="MAPE threshold to trigger SGD update (default: 0.10)")
    parser.add_argument("--explore-k", type=int, default=4,
                        help="Number of exploration alternatives (default: 4)")
    parser.add_argument("--explore-thresh", type=float, default=0.20,
                        help="Cost error threshold for exploration (default: 0.20)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_epochs = sorted(set(int(e) for e in args.checkpoint_epochs.split(",")))
    checkpoint_epochs = [e for e in checkpoint_epochs if e <= args.epochs]

    # Resolve output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    if args.outdir:
        outdir = args.outdir
    else:
        outdir = os.path.join(project_dir, "data", "ai_training", "gpt2_wikitext2")
    os.makedirs(outdir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── HDF5 direct writer (optional) ──
    hdf5_writer = None
    if args.hdf5_direct:
        from gpucompress_hdf5 import GPUCompressHDF5Writer, concat_and_pad_gpu
        weights_path = os.path.join(project_dir, "neural_net", "weights", "model.nnwt")
        if not os.path.exists(weights_path):
            weights_path = None
        hdf5_writer = GPUCompressHDF5Writer(
            lib_dir=os.path.join(project_dir, "build"),
            weights_path=weights_path,
        )
        hdf5_writer.init()
        hdf5_writer.set_policy(args.policy)
        print(f"  HDF5 direct write enabled (chunk={args.chunk_mb}MB, eb={args.error_bound}, policy={args.policy})")

    # ── Load model + tokenizer ──
    print("Loading GPT-2 Small...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Unfreeze all layers
    for p in model.parameters():
        p.requires_grad = True

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    param_mb = n_params * 4 / (1024 * 1024)

    d0, d1 = compute_dims_2d(n_params)
    target_elements = d0 * d1
    pad_elements = target_elements - n_params

    print(f"  Parameters  : {n_params:,} ({param_mb:.1f} MB as float32)")
    print(f"  Padded dims : {d0} x {d1} = {target_elements:,} ({pad_elements} padding)")
    print(f"  Device      : {device}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Checkpoints : {checkpoint_epochs}")
    print(f"  Block size  : {args.block_size}")
    print(f"  Output      : {outdir}")
    print()

    # ── Inline full benchmark (optional) ──
    inline_bench = None
    bench_configs = []
    if hdf5_writer is not None and (args.benchmark or args.benchmark_configs):
        from gpucompress_hdf5 import InlineFullBenchmark
        nn_weights = os.path.join(project_dir, "neural_net", "weights", "model.nnwt")

        if args.benchmark_configs:
            for cfg_str in args.benchmark_configs.split(","):
                parts = cfg_str.strip().split(":")
                if len(parts) != 3:
                    continue
                c_mb, c_eb, c_outdir = int(parts[0]), float(parts[1]), parts[2]
                os.makedirs(c_outdir, exist_ok=True)
                bench = InlineFullBenchmark(hdf5_writer, nn_weights, target_elements,
                                           sgd_lr=args.sgd_lr, sgd_mape=args.sgd_mape,
                                           explore_k=args.explore_k, explore_thresh=args.explore_thresh)
                bench_configs.append({
                    "chunk_mb": c_mb, "error_bound": c_eb, "outdir": c_outdir,
                    "bench": bench, "bench_csv": None, "chunk_csv": None,
                })
            print(f"  Multi-config benchmark: {len(bench_configs)} configs × 15 algorithms")
            print(f"  SGD: lr={args.sgd_lr}, mape={args.sgd_mape} | Explore: k={args.explore_k}, thresh={args.explore_thresh}")
        else:
            inline_bench = InlineFullBenchmark(hdf5_writer, nn_weights, target_elements,
                                               sgd_lr=args.sgd_lr, sgd_mape=args.sgd_mape,
                                               explore_k=args.explore_k, explore_thresh=args.explore_thresh)
            print(f"  Inline benchmark: 15 configs (9 fixed + 6 NN)")
            print(f"  SGD: lr={args.sgd_lr}, mape={args.sgd_mape} | Explore: k={args.explore_k}, thresh={args.explore_thresh}")

    # ── Load WikiText-2 ──
    print("Loading WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Tokenize
    print("Tokenizing...")
    train_encodings = []
    for text in dataset["train"]["text"]:
        if text.strip():
            ids = tokenizer.encode(text)
            if len(ids) > 0:
                train_encodings.append(ids)

    val_encodings = []
    for text in dataset["validation"]["text"]:
        if text.strip():
            ids = tokenizer.encode(text)
            if len(ids) > 0:
                val_encodings.append(ids)

    train_ds = TextDataset(train_encodings, args.block_size)
    val_ds = TextDataset(val_encodings, args.block_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    steps_per_epoch = len(train_loader)
    print(f"  Train: {len(train_ds)} blocks of {args.block_size} tokens, {steps_per_epoch} steps/epoch")
    print(f"  Val  : {len(val_ds)} blocks")
    print()

    # ── Optimizer ──
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ──
    print("=" * 60)
    print("  Training GPT-2 Small on WikiText-2")
    print("=" * 60)

    total_start = time.time()
    bench_csv = None
    chunk_csv = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_start = time.time()
        optimizer.zero_grad()

        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss / args.gradient_accumulation
            loss.backward()

            if (batch_idx + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += outputs.loss.item() * input_ids.size(0)
            epoch_tokens += input_ids.numel()

            if (batch_idx + 1) % 50 == 0:
                pct = 100.0 * (batch_idx + 1) / steps_per_epoch
                sys.stdout.write(f"\r  Epoch {epoch:2d}/{args.epochs} "
                                 f"[{pct:5.1f}%] loss={outputs.loss.item():.4f}")
                sys.stdout.flush()

        scheduler.step()
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(train_ds)
        ppl = math.exp(min(avg_loss, 20))  # cap to avoid overflow

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                outputs = model(input_ids, labels=labels)
                val_loss += outputs.loss.item() * input_ids.size(0)
        val_avg = val_loss / len(val_ds)
        val_ppl = math.exp(min(val_avg, 20))

        print(f"\r  Epoch {epoch:2d}/{args.epochs}  "
              f"loss={avg_loss:.4f}  ppl={ppl:.1f}  "
              f"val_loss={val_avg:.4f}  val_ppl={val_ppl:.1f}  "
              f"time={epoch_time:.1f}s  lr={scheduler.get_last_lr()[0]:.2e}")

        # ── Export checkpoint ──
        if epoch in checkpoint_epochs:
            print(f"\n  >>> Exporting checkpoint at epoch {epoch}...")
            export_start = time.time()

            # Helper: collect Adam state tensors
            def _adam_tensors(key):
                tensors = []
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        if p in optimizer.state and key in optimizer.state[p]:
                            tensors.append(optimizer.state[p][key])
                        else:
                            tensors.append(torch.zeros_like(p))
                return tensors

            # Compute gradients for this checkpoint
            model.train()
            optimizer.zero_grad()
            input_ids, labels = next(iter(train_loader))
            input_ids, labels = input_ids.to(device), labels.to(device)
            outputs = model(input_ids, labels=labels)
            outputs.loss.backward()
            grad_tensors = [p.grad if p.grad is not None else torch.zeros_like(p)
                            for p in model.parameters()]

            # Export 4 tensor types
            tensor_sets = [
                ("weights",   list(model.parameters())),
                ("adam_m",    _adam_tensors("exp_avg")),
                ("adam_v",    _adam_tensors("exp_avg_sq")),
                ("gradients", grad_tensors),
            ]

            CSV_HEADER = ("epoch,tensor,algorithm,policy,mode,ratio,"
                         "write_ms,read_ms,write_mbps,read_mbps,"
                         "file_bytes,orig_bytes,mismatches,"
                         "n_chunks,sgd_fires,explorations,"
                         "mape_ratio_pct,mape_comp_pct,mape_decomp_pct,mape_psnr_pct,"
                         "mae_ratio,mae_comp_ms,mae_decomp_ms,mae_psnr_db,"
                         "r2_ratio,r2_comp,r2_decomp,r2_psnr,"
                         "nn_ms,stats_ms,preproc_ms,comp_ms,decomp_ms,"
                         "explore_ms,sgd_ms,"
                         "stage1_ms,drain_ms,io_drain_ms,pipeline_ms,"
                         "s2_busy_ms,s3_busy_ms,"
                         "psnr_db,rmse,max_abs_err,bit_rate\n")
            CHUNK_CSV_HEADER = ("epoch,tensor,algorithm,policy,mode,"
                                "chunk_idx,action,actual_ratio,predicted_ratio,"
                                "comp_ms,predicted_comp_time,"
                                "decomp_ms,predicted_decomp_time,"
                                "sgd_fired,exploration_triggered\n")

            if inline_bench is not None and bench_csv is None:
                bench_csv = open(os.path.join(outdir, "inline_benchmark.csv"), "w")
                bench_csv.write(CSV_HEADER)
                chunk_csv = open(os.path.join(outdir, "inline_benchmark_chunks.csv"), "w")
                chunk_csv.write(CHUNK_CSV_HEADER)

            if bench_configs:
                for bc in bench_configs:
                    if bc["bench_csv"] is None:
                        bc["bench_csv"] = open(os.path.join(bc["outdir"], "inline_benchmark.csv"), "w")
                        bc["bench_csv"].write(CSV_HEADER)
                        bc["chunk_csv"] = open(os.path.join(bc["outdir"], "inline_benchmark_chunks.csv"), "w")
                        bc["chunk_csv"].write(CHUNK_CSV_HEADER)

            for name, tensors in tensor_sets:
                flat = concat_and_pad_gpu(tensors, target_elements)

                if bench_configs:
                    for bc in bench_configs:
                        mode_label = "lossless" if bc["error_bound"] == 0.0 else f"lossy(eb={bc['error_bound']})"
                        print(f"      [{bc['chunk_mb']}MB {mode_label}]")
                        bc["bench"].run_checkpoint(
                            hdf5_writer, flat, target_elements,
                            epoch, name, bc["outdir"],
                            chunk_elements=bc["chunk_mb"] * 1024 * 1024 // 4,
                            error_bound=bc["error_bound"],
                            csv_file=bc["bench_csv"],
                            chunk_csv_file=bc["chunk_csv"],
                        )
                    h5_path = os.path.join(outdir, f"epoch{epoch:02d}_{name}.h5")
                    hdf5_writer.write_gpu_tensor(
                        flat.data_ptr(), target_elements, h5_path, "data",
                        chunk_elements=args.chunk_mb * 1024 * 1024 // 4,
                        error_bound=args.error_bound,
                    )
                elif inline_bench is not None:
                    inline_bench.run_checkpoint(
                        hdf5_writer, flat, target_elements,
                        epoch, name, outdir,
                        chunk_elements=args.chunk_mb * 1024 * 1024 // 4,
                        error_bound=args.error_bound,
                        csv_file=bench_csv,
                        chunk_csv_file=chunk_csv,
                    )
                    h5_path = os.path.join(outdir, f"epoch{epoch:02d}_{name}.h5")
                    hdf5_writer.write_gpu_tensor(
                        flat.data_ptr(), target_elements, h5_path, "data",
                        chunk_elements=args.chunk_mb * 1024 * 1024 // 4,
                        error_bound=args.error_bound,
                    )
                elif hdf5_writer is not None:
                    h5_path = os.path.join(outdir, f"epoch{epoch:02d}_{name}.h5")
                    hdf5_writer.write_gpu_tensor(
                        flat.data_ptr(), target_elements, h5_path, "data",
                        chunk_elements=args.chunk_mb * 1024 * 1024 // 4,
                        error_bound=args.error_bound,
                    )
                    sz = os.path.getsize(h5_path)
                    print(f"      epoch{epoch:02d}_{name}.h5  {sz/1024/1024:>7.1f} MB")
                else:
                    f32_path = os.path.join(outdir, f"epoch{epoch:02d}_{name}.f32")
                    export_tensor_padded(tensors, f32_path, target_elements)
                    sz = os.path.getsize(f32_path)
                    print(f"      epoch{epoch:02d}_{name}.f32  {sz/1024/1024:>7.1f} MB")

                del flat
                torch.cuda.empty_cache()

            optimizer.zero_grad()
            export_time = time.time() - export_start
            print(f"      Export time: {export_time:.1f}s\n")

    total_time = time.time() - total_start

    # ── Cleanup ──
    if bench_csv is not None:
        bench_csv.close()
    if chunk_csv is not None:
        chunk_csv.close()
    for bc in bench_configs:
        if bc.get("bench_csv"):
            bc["bench_csv"].close()
        if bc.get("chunk_csv"):
            bc["chunk_csv"].close()
    if hdf5_writer is not None:
        hdf5_writer.cleanup()

    # ── Auto-generate plots ──
    plot_dirs = []
    if bench_csv is not None:
        plot_dirs.append(outdir)
    for bc in bench_configs:
        plot_dirs.append(bc["outdir"])
    for pdir in plot_dirs:
        csv_path = os.path.join(pdir, "inline_benchmark.csv")
        if os.path.exists(csv_path):
            print(f"\n  Generating plots for {os.path.basename(pdir)}...")
            try:
                from plot_inline_benchmark import main as plot_main
                sys.argv = ["plot_inline_benchmark.py", csv_path]
                plot_main()
            except Exception as e:
                print(f"    Plot generation failed: {e}")

    # ── Write metadata ──
    _ext = ".h5" if args.hdf5_direct else ".f32"
    dims_str = f"{d0},{d1}"
    meta_path = os.path.join(outdir, "README.txt")
    with open(meta_path, "w") as f:
        f.write(f"GPT-2 Small Training Checkpoint Data for GPUCompress Benchmarks\n")
        f.write(f"Generated by: scripts/train_gpt2_checkpoints.py\n\n")
        f.write(f"Model: GPT-2 Small ({n_params:,} parameters, {param_mb:.1f} MB)\n")
        f.write(f"Dataset: WikiText-2\n")
        f.write(f"Epochs: {args.epochs}, checkpoints at: {checkpoint_epochs}\n")
        f.write(f"Optimizer: AdamW (lr={args.lr}, wd={args.weight_decay})\n")
        f.write(f"Block size: {args.block_size} tokens\n")
        f.write(f"Dims for benchmark: --dims {dims_str}\n\n")
        f.write(f"Files:\n")
        for fname in sorted(os.listdir(outdir)):
            if fname.endswith(_ext):
                fsize = os.path.getsize(os.path.join(outdir, fname))
                f.write(f"  {fname:40s} {fsize/1024/1024:>8.1f} MB\n")

    # ── Summary ──
    n_files = len([f for f in os.listdir(outdir) if f.endswith(_ext)])
    total_disk = sum(os.path.getsize(os.path.join(outdir, f))
                     for f in os.listdir(outdir) if f.endswith(_ext))

    print()
    print("=" * 60)
    print(f"  GPT-2 Checkpoint Export Complete")
    print("=" * 60)
    print(f"  Model         : GPT-2 Small ({n_params:,} params)")
    print(f"  Training time : {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Final val_ppl : {val_ppl:.1f}")
    print(f"  Files         : {n_files} {_ext} files")
    print(f"  Total disk    : {total_disk/1024/1024/1024:.1f} GB")
    print(f"  Dims          : --dims {dims_str}")
    print(f"  Output        : {outdir}")
    print()
    print(f"  Run benchmark:")
    print(f"    BENCHMARKS=ai_training AI_MODEL=gpt2 AI_DATASET=wikitext2 \\")
    print(f"      CHUNK_MB=4 POLICIES=balanced VERIFY=0 bash benchmarks/benchmark.sh")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
