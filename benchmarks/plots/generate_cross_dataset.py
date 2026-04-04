#!/usr/bin/env python3
"""
Generate cross-dataset convergence comparison figure.

Overlays MAPE convergence curves from multiple datasets (VPIC, SDRBench, AI)
on one 2×2 grid (ratio, comp_time, decomp_time, PSNR MAPE).

Usage:
    # Auto-discover from recent results:
    python3 benchmarks/plots/generate_cross_dataset.py

    # Explicit CSV paths:
    python3 benchmarks/plots/generate_cross_dataset.py \
        --csv benchmarks/vpic-kokkos/results/eval_NX320/balanced/benchmark_vpic_deck_timesteps.csv \
        --name "VPIC (NX=320)" \
        --csv benchmarks/sdrbench/results/eval_nyx/balanced/merged_csv/benchmark_nyx.csv \
        --name "NYX" \
        --output benchmarks/results/cross_dataset_convergence.png
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import visualize as viz


def auto_discover():
    """Find timestep CSVs from recent benchmark results."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_paths = []
    names = []

    # VPIC results
    for d in sorted(glob.glob(os.path.join(base, "vpic-kokkos/results/eval_*/balanced_*/benchmark_vpic_deck_timesteps.csv"))):
        label = os.path.basename(os.path.dirname(os.path.dirname(d)))
        csv_paths.append(d)
        names.append(f"VPIC {label.replace('eval_', '')}")

    # SDRBench results
    for d in sorted(glob.glob(os.path.join(base, "sdrbench/results/eval_*/balanced_*/merged_csv/benchmark_*.csv"))):
        ds = os.path.basename(d).replace("benchmark_", "").replace(".csv", "")
        eval_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(d))))
        csv_paths.append(d)
        names.append(f"{ds} ({eval_dir.replace('eval_', '')})")

    # AI inline benchmark results
    for d in sorted(glob.glob(os.path.join(base, "../data/ai_training/*/inline_benchmark.csv"))):
        label = os.path.basename(os.path.dirname(d))
        csv_paths.append(d)
        names.append(f"AI: {label}")

    return csv_paths, names


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset convergence comparison")
    parser.add_argument("--csv", action="append", default=[], help="Timestep CSV path (repeat for each dataset)")
    parser.add_argument("--name", action="append", default=[], help="Dataset display name (same order as --csv)")
    parser.add_argument("--output", default="benchmarks/results/cross_dataset_convergence.png",
                        help="Output PNG path")
    parser.add_argument("--auto", action="store_true", help="Auto-discover from results directories")
    args = parser.parse_args()

    if args.auto or (not args.csv):
        csv_paths, names = auto_discover()
        if not csv_paths:
            print("No datasets found. Run benchmarks first or use --csv/--name explicitly.")
            return
    else:
        csv_paths = args.csv
        names = args.name
        if len(csv_paths) != len(names):
            print(f"ERROR: {len(csv_paths)} CSVs but {len(names)} names")
            return

    print(f"Cross-dataset convergence: {len(csv_paths)} datasets")
    for p, n in zip(csv_paths, names):
        exists = "✓" if os.path.exists(p) else "✗"
        print(f"  {exists} {n}: {p}")

    viz.make_cross_dataset_convergence_figure(csv_paths, names, args.output)


if __name__ == "__main__":
    main()
