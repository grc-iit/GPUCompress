#!/usr/bin/env python3
"""
Visualize per-chunk MAPE for ratio, comp time, and decomp time.

Reads the CSV produced by test_vol_nn_predictions and plots three subplots
showing how prediction error evolves over chunks (i.e., as SGD learns).

Usage:
    .venv/bin/python3 benchmarks/visualize_mape.py [csv_path] [--out fig.png]

Default CSV: /tmp/test_vol_nn_predictions.csv
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_CSV = "/tmp/test_vol_nn_predictions.csv"


def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "chunk":       int(row["chunk"]),
                "pattern":     row["pattern"],
                "nn_pick":     row["nn_pick"],
                "ratio_mape":  float(row["ratio_mape"]),
                "comp_mape":   float(row["comp_mape"]),
                "decomp_mape": float(row["decomp_mape"]),
                "sgd_fired":   int(row["sgd_fired"]),
            })
    return rows


def plot_mape(rows, out_path):
    chunks      = np.array([r["chunk"] for r in rows])
    ratio_mape  = np.array([r["ratio_mape"] for r in rows])
    comp_mape   = np.array([r["comp_mape"] for r in rows])
    decomp_mape = np.array([r["decomp_mape"] for r in rows])
    sgd_fired   = np.array([r["sgd_fired"] for r in rows], dtype=bool)
    patterns    = [r["pattern"] for r in rows]

    # Unique patterns for color coding
    unique_pats = list(dict.fromkeys(patterns))
    pat_colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_pats), 1)))
    pat_cmap = {p: pat_colors[i] for i, p in enumerate(unique_pats)}
    colors = [pat_cmap[p] for p in patterns]

    metrics = [
        ("Ratio MAPE (%)",       ratio_mape),
        ("Comp Time MAPE (%)",   comp_mape),
        ("Decomp Time MAPE (%)", decomp_mape),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for ax, (label, mape) in zip(axes, metrics):
        # Bar chart colored by pattern
        ax.bar(chunks, mape, color=colors, edgecolor="white", linewidth=0.3, width=1.0)

        # Mark SGD fires
        sgd_chunks = chunks[sgd_fired]
        sgd_vals   = mape[sgd_fired]
        if len(sgd_chunks) > 0:
            ax.scatter(sgd_chunks, sgd_vals + max(mape) * 0.02,
                       marker="v", color="black", s=18, zorder=5, label="SGD fired")

        # Rolling average (window=6, i.e. one full pattern cycle)
        if len(mape) >= 6:
            window = min(6, len(mape))
            kernel = np.ones(window) / window
            rolling = np.convolve(mape, kernel, mode="valid")
            x_roll = chunks[window - 1:]
            ax.plot(x_roll, rolling, color="red", linewidth=2, alpha=0.8,
                    label=f"rolling avg (w={window})")

        ax.set_ylabel(label, fontsize=11)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Chunk index", fontsize=11)
    axes[0].set_title("NN Prediction MAPE over Chunks (SGD Online Learning)", fontsize=13, fontweight="bold")

    # Pattern legend
    from matplotlib.patches import Patch
    legend_pats = [Patch(facecolor=pat_cmap[p], edgecolor="gray", label=p) for p in unique_pats]
    fig.legend(handles=legend_pats, loc="lower center",
               ncol=min(len(unique_pats), 8), fontsize=9,
               bbox_to_anchor=(0.5, -0.01), frameon=True)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize per-chunk MAPE from NN predictions CSV")
    parser.add_argument("csv", nargs="?", default=DEFAULT_CSV,
                        help=f"CSV path (default: {DEFAULT_CSV})")
    parser.add_argument("--out", default=None,
                        help="Output image path (default: <csv_dir>/mape_over_chunks.png)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: CSV not found: {args.csv}", file=sys.stderr)
        print("Run test_vol_nn_predictions first to generate the CSV.", file=sys.stderr)
        sys.exit(1)

    if args.out is None:
        csv_dir = os.path.dirname(os.path.abspath(args.csv))
        args.out = os.path.join(csv_dir, "mape_over_chunks.png")

    rows = load_csv(args.csv)
    if not rows:
        print("No data in CSV", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(rows)} chunks from {args.csv}")
    plot_mape(rows, args.out)


if __name__ == "__main__":
    main()
