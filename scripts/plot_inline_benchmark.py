#!/usr/bin/env python3
"""
Plot inline benchmark results from training with --benchmark.

Reads inline_benchmark.csv and generates figures matching the VPIC/Gray-Scott
benchmark visualization suite:

  1_summary.png                  - 4-panel: ratio, write, read, Pareto
  3_algorithm_evolution.png      - Per-epoch algorithm selection heatmap (NN configs)
  4_nn_predicted_vs_actual.png   - MAPE ratio/comp/decomp per NN config
  5a_sgd_convergence.png         - MAPE over epochs for NN-RL configs
  5b_sgd_exploration_firing.png  - SGD fires + explorations per epoch
  5c_mae_over_time.png           - MAE ratio/comp/decomp over epochs
  6b_pipeline_waterfall.png      - Pipeline stage timing breakdown
  6c_gpu_breakdown_over_time.png - Component timing (NN, stats, comp, decomp, SGD, explore)
  7_epoch_evolution.png          - Ratio over epochs per algorithm

Usage:
    python3 scripts/plot_inline_benchmark.py data/ai_training/vit_b_cifar10_15cfg/inline_benchmark.csv
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# All possible algorithms in display order (matches VPIC)
FIXED_ALGOS = ["no-comp", "lz4", "snappy", "deflate", "gdeflate", "zstd", "ans", "cascaded", "bitcomp"]
NN_ALGOS = ["bal_nn", "bal_rl", "bal_exp",
            "rat_nn", "rat_rl", "rat_exp"]
ALL_ALGOS = FIXED_ALGOS + NN_ALGOS
TENSOR_ORDER = ["weights", "adam_m", "adam_v", "gradients"]

# Colors matching VPIC visualize.py
ALGO_COLORS = {
    "no-comp":  "#7f8c8d",   # muted grey
    "lz4":      "#3498db",   # blue
    "snappy":   "#5dade2",   # light blue
    "deflate":  "#2e86c1",   # medium blue
    "gdeflate": "#2980b9",   # darker blue
    "zstd":     "#1a5276",   # dark blue
    "ans":      "#148f77",   # teal
    "cascaded": "#1abc9c",   # turquoise
    "bitcomp":  "#0e6655",   # dark teal
    # NN configs: balanced = warm, ratio_lean = mid, ratio = cool
    "bal_nn":       "#e67e22",   # orange (like VPIC nn)
    "bal_rl":       "#8e44ad",   # purple (like VPIC nn-rl)
    "bal_exp":      "#c0392b",   # red (like VPIC nn-rl+exp)
    "ratlean_nn":   "#e74c3c",   # bright red
    "ratlean_rl":   "#c0392b",   # dark red
    "ratlean_exp":  "#a93226",   # deep red
    "rat_nn":       "#f39c12",   # gold
    "rat_rl":       "#6c3483",   # dark purple
    "rat_exp":      "#922b21",   # dark red
    "nn-auto":  "#e67e22",
}

ALGO_HATCHES = {
    "no-comp": "", "lz4": "", "snappy": "", "deflate": "", "gdeflate": "",
    "zstd": "", "ans": "", "cascaded": "", "bitcomp": "",
    "bal_nn": "", "bal_rl": "\\\\", "bal_exp": "xx",
    "ratlean_nn": "", "ratlean_rl": "\\\\", "ratlean_exp": "xx",
    "rat_nn": "", "rat_rl": "\\\\", "rat_exp": "xx",
}

ALGO_LABELS = {
    "no-comp": "No Comp", "lz4": "LZ4", "snappy": "Snappy",
    "deflate": "Deflate", "gdeflate": "GDeflate", "zstd": "Zstd",
    "ans": "ANS", "cascaded": "Cascaded", "bitcomp": "Bitcomp",
    "bal_nn": "Bal NN", "bal_rl": "Bal NN+SGD", "bal_exp": "Bal NN+SGD\n+Explore",
    "ratlean_nn": "RatLean NN", "ratlean_rl": "RatLean\nNN+SGD", "ratlean_exp": "RatLean\n+Explore",
    "rat_nn": "Rat NN", "rat_rl": "Rat NN+SGD", "rat_exp": "Rat NN+SGD\n+Explore",
}

TENSOR_COLORS = {
    "weights": "#1f77b4", "adam_m": "#ff7f0e",
    "adam_v": "#2ca02c", "gradients": "#d62728",
}

TENSOR_MARKERS = {"weights": "o", "adam_m": "s", "adam_v": "^", "gradients": "D"}


def _sc_bar(ax, algos, values, title, ylabel, fmt="%.2f", yerr=None):
    """SC-style bar chart matching VPIC visualize.plot_bars."""
    x = np.arange(len(algos))
    colors = [ALGO_COLORS.get(a, "#bdc3c7") for a in algos]
    hatches = [ALGO_HATCHES.get(a, "") for a in algos]
    err_kw = dict(ecolor="#333", capsize=3, capthick=0.8, elinewidth=0.8) if yerr else {}
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.4,
                  width=0.6, zorder=3, yerr=yerr, error_kw=err_kw)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    for i, (bar, v) in enumerate(zip(bars, values)):
        if v > 0:
            offset = (yerr[i] if yerr and yerr[i] > 0 else 0)
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                    fmt % v, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABELS.get(a, a) for a in algos], rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.2, linestyle="--", zorder=0)
    ax.set_axisbelow(True)


def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["epoch"] = int(r["epoch"])
            for k in ["ratio", "write_ms", "read_ms", "write_mbps", "read_mbps"]:
                r[k] = float(r.get(k, 0))
            for k in ["file_bytes", "orig_bytes", "mismatches", "n_chunks",
                       "sgd_fires", "explorations"]:
                r[k] = int(float(r.get(k, 0)))
            for k in ["mape_ratio_pct", "mape_comp_pct", "mape_decomp_pct",
                       "mae_ratio", "mae_comp_ms", "mae_decomp_ms",
                       "nn_ms", "stats_ms", "preproc_ms", "comp_ms", "decomp_ms",
                       "explore_ms", "sgd_ms",
                       "stage1_ms", "drain_ms", "io_drain_ms", "pipeline_ms",
                       "s2_busy_ms", "s3_busy_ms"]:
                r[k] = float(r.get(k, 0))
            rows.append(r)
    return rows


def _get_algos(rows):
    """Return algorithms present in data, in display order."""
    present = set(r["algorithm"] for r in rows)
    return [a for a in ALL_ALGOS if a in present]


def fig1_summary(rows, outdir, epoch=None, subtitle=""):
    """4-panel summary matching VPIC style: ratio, write, read, Pareto."""
    if epoch is None:
        epoch = max(r["epoch"] for r in rows)
    erows = [r for r in rows if r["epoch"] == epoch]
    algos = _get_algos(erows)

    # Average across tensors for each algorithm
    avg = {}
    for a in algos:
        arows = [r for r in erows if r["algorithm"] == a]
        avg[a] = {
            "ratio": np.mean([r["ratio"] for r in arows]),
            "write_mbps": np.mean([r["write_mbps"] for r in arows]),
            "read_mbps": np.mean([r["read_mbps"] for r in arows]),
        }

    fig = plt.figure(figsize=(14, 10), facecolor="white")
    # Detect if this is a single-tensor view
    tensors_in_data = sorted(set(r["tensor"] for r in erows))
    if not subtitle:
        if len(tensors_in_data) == 1:
            subtitle = f"Epoch {epoch} | {len(algos)} configs | {tensors_in_data[0]}"
        else:
            subtitle = f"Epoch {epoch} | {len(algos)} configs | avg across {len(tensors_in_data)} tensors"
    else:
        subtitle = f"Epoch {epoch} | {len(algos)} configs | {subtitle}"

    fig.text(0.5, 0.99, "GPUCompress Benchmark: AI Training Checkpoint",
             ha="center", fontsize=14, fontweight="bold", va="top")
    fig.text(0.5, 0.965, subtitle,
             ha="center", fontsize=9, color="#444", va="top", fontfamily="monospace")

    from matplotlib import gridspec
    gs = gridspec.GridSpec(2, 2, hspace=0.50, wspace=0.30,
                           top=0.92, bottom=0.08, left=0.07, right=0.97)

    # Panel 1: Ratio
    ax = fig.add_subplot(gs[0, 0])
    _sc_bar(ax, algos, [avg[a]["ratio"] for a in algos],
            "Compression Ratio (higher = better)", "Ratio", fmt="%.2fx")

    # Panel 2: Write Throughput
    ax = fig.add_subplot(gs[0, 1])
    _sc_bar(ax, algos, [avg[a]["write_mbps"] for a in algos],
            "End-to-End Write Throughput (full pipeline)", "MiB/s", fmt="%.0f")

    # Panel 3: Read Throughput
    ax = fig.add_subplot(gs[1, 0])
    _sc_bar(ax, algos, [avg[a]["read_mbps"] for a in algos],
            "End-to-End Read Throughput (full pipeline)", "MiB/s", fmt="%.0f")

    # Panel 4: Pareto
    ax = fig.add_subplot(gs[1, 1])
    for a in algos:
        r = avg[a]
        color = ALGO_COLORS.get(a, "#bdc3c7")
        label = ALGO_LABELS.get(a, a).replace("\n", " ")
        marker = {"bal_rl": "s", "bal_exp": "^", "rat_rl": "s", "rat_exp": "^",
                  "bal_nn": "D", "rat_nn": "D"}.get(a, "o")
        ax.scatter(r["ratio"], r["write_mbps"], s=120, color=color,
                   edgecolors="black", linewidth=0.8, zorder=5, marker=marker)
        ax.annotate(label, (r["ratio"], r["write_mbps"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=7, color="#333")
    ax.set_xlabel("Compression Ratio")
    ax.set_ylabel("Write Throughput (MiB/s)")
    ax.set_title("Ratio vs Throughput (Pareto)", fontweight="bold")
    ax.grid(alpha=0.2, linestyle="--", zorder=0)
    ax.set_axisbelow(True)

    fig.savefig(os.path.join(outdir, "1_summary.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.join(outdir, '1_summary.png')}")


def load_chunk_csv(path):
    """Load per-chunk CSV if it exists."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["epoch"] = int(r["epoch"])
            r["chunk_idx"] = int(r["chunk_idx"])
            r["action"] = int(r["action"])
            for k in ["actual_ratio", "predicted_ratio", "comp_ms",
                       "predicted_comp_time", "decomp_ms", "predicted_decomp_time"]:
                r[k] = float(r.get(k, 0))
            r["sgd_fired"] = int(r.get("sgd_fired", 0))
            r["exploration_triggered"] = int(r.get("exploration_triggered", 0))
            rows.append(r)
    return rows


# Action ID to algorithm name mapping
# Action encoding from gpucompress.h:567:
#   algo = action % 8, quant = (action/8) % 2, shuffle = (action/16) % 2
# 0-7: plain, 8-15: +quant, 16-23: +shuffle, 24-31: +quant+shuffle
ACTION_NAMES = {
    0: "lz4", 1: "snappy", 2: "deflate", 3: "gdeflate",
    4: "zstd", 5: "ans", 6: "cascaded", 7: "bitcomp",
    8: "lz4+q", 9: "snappy+q", 10: "deflate+q", 11: "gdeflate+q",
    12: "zstd+q", 13: "ans+q", 14: "cascaded+q", 15: "bitcomp+q",
    16: "lz4+shuf", 17: "snappy+shuf", 18: "deflate+shuf", 19: "gdeflate+shuf",
    20: "zstd+shuf", 21: "ans+shuf", 22: "cascaded+shuf", 23: "bitcomp+shuf",
    24: "lz4+q+s", 25: "snappy+q+s", 26: "deflate+q+s", 27: "gdeflate+q+s",
    28: "zstd+q+s", 29: "ans+q+s", 30: "cascaded+q+s", 31: "bitcomp+q+s",
}


def _build_action_cmap():
    """Build colormap and norm for the 32-action palette.

    Action encoding: algo = action % 8, quant = (action/8) % 2, shuffle = (action/16) % 2
    0-7: plain, 8-15: +quant, 16-23: +shuffle, 24-31: +quant+shuffle
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm
    palette = [
        # 0-7: plain (cool blues/teals)
        "#3498db",  # 0  lz4
        "#85c1e9",  # 1  snappy
        "#2e86c1",  # 2  deflate
        "#2980b9",  # 3  gdeflate
        "#1a5276",  # 4  zstd
        "#148f77",  # 5  ans
        "#1abc9c",  # 6  cascaded
        "#117a65",  # 7  bitcomp
        # 8-15: +quant (warm reds/oranges)
        "#e74c3c",  # 8  lz4+q
        "#e67e22",  # 9  snappy+q
        "#c0392b",  # 10 deflate+q
        "#d35400",  # 11 gdeflate+q
        "#922b21",  # 12 zstd+q
        "#f39c12",  # 13 ans+q
        "#8e44ad",  # 14 cascaded+q
        "#6c3483",  # 15 bitcomp+q
        # 16-23: +shuffle (medium blues, slightly lighter)
        "#5dade2",  # 16 lz4+shuf
        "#aed6f1",  # 17 snappy+shuf
        "#2471a3",  # 18 deflate+shuf
        "#1f618d",  # 19 gdeflate+shuf
        "#154360",  # 20 zstd+shuf
        "#0e6655",  # 21 ans+shuf
        "#16a085",  # 22 cascaded+shuf
        "#0b5345",  # 23 bitcomp+shuf
        # 24-31: +quant+shuffle (dark reds/purples)
        "#f1948a",  # 24 lz4+q+s
        "#f0b27a",  # 25 snappy+q+s
        "#a93226",  # 26 deflate+q+s
        "#ba4a00",  # 27 gdeflate+q+s
        "#7b241c",  # 28 zstd+q+s
        "#d4ac0d",  # 29 ans+q+s
        "#7d3c98",  # 30 cascaded+q+s
        "#5b2c6f",  # 31 bitcomp+q+s
    ]
    cmap = ListedColormap(palette)
    norm = BoundaryNorm(np.arange(-0.5, 32.5, 1), cmap.N)
    return cmap, norm


def fig3_algorithm_evolution(chunk_rows, outdir, subtitle=""):
    """Per-chunk algorithm selection heatmap for NN configs across epochs.

    VPIC-style layout: columns = NN configs, rows = epochs.
    Each cell is an imshow strip (1 row × N chunks) colored by action.
    """
    from matplotlib.patches import Patch

    nn_chunks = [r for r in chunk_rows if r.get("policy", "-") != "-"]
    if not nn_chunks:
        print("  Skipping 3_algorithm_evolution (no NN chunk data)")
        return

    epochs = sorted(set(r["epoch"] for r in nn_chunks))
    nn_algos = sorted(set(r["algorithm"] for r in nn_chunks))
    max_chunk = max(r["chunk_idx"] for r in nn_chunks) + 1

    cmap, norm = _build_action_cmap()
    n_epochs = len(epochs)
    n_configs = len(nn_algos)

    # Layout: rows = epochs, columns = NN configs (matches VPIC orientation)
    fig_w = max(8, 3 * n_configs + 1)
    fig_h = max(6, 1.2 * n_epochs + 3)
    fig, axes = plt.subplots(n_epochs, n_configs,
                              figsize=(fig_w, fig_h),
                              squeeze=False)
    _title = "NN Algorithm Selection per Chunk Over Epochs"
    if subtitle:
        _title += f"\n{subtitle}"
    fig.suptitle(_title, fontsize=14, fontweight="bold", y=0.97)

    used_actions = set()

    for col, algo in enumerate(nn_algos):
        for row, epoch in enumerate(epochs):
            ax = axes[row][col]

            # Build 1-row strip: chunk actions
            crows = sorted(
                [r for r in nn_chunks
                 if r["algorithm"] == algo and r["epoch"] == epoch],
                key=lambda r: r["chunk_idx"])
            strip = []
            for r in crows:
                a = r["action"]
                used_actions.add(a)
                strip.append(a)
            # Pad to max_chunk
            while len(strip) < max_chunk:
                strip.append(-1)

            arr = np.array([strip], dtype=float)
            arr[arr < 0] = np.nan

            ax.imshow(arr, aspect="auto", cmap=cmap, norm=norm,
                      interpolation="nearest")
            ax.set_yticks([])

            # Column headers (top row only)
            if row == 0:
                label = ALGO_LABELS.get(algo, algo).replace("\n", " ")
                ax.set_title(label, fontsize=11, fontweight="bold", pad=8)
            # Row labels (left column only)
            if col == 0:
                ax.set_ylabel(f"E{epoch}", fontsize=10, fontweight="bold",
                              rotation=0, labelpad=25, va="center")
            # X-axis ticks (bottom row only)
            if row == n_epochs - 1:
                step = max(1, max_chunk // 8)
                ax.set_xticks(range(0, max_chunk, step))
                ax.set_xlabel("Chunk Index", fontsize=9)
                ax.tick_params(axis="x", labelsize=7)
            else:
                ax.set_xticks([])

            # Clean borders
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#888888")

    # Legend — only actions actually used
    legend_patches = []
    for a in sorted(used_actions):
        if 0 <= a < 32:
            legend_patches.append(
                Patch(facecolor=cmap(norm(a)),
                      edgecolor="black", linewidth=0.5,
                      label=ACTION_NAMES.get(a, f"act{a}")))

    if legend_patches:
        fig.legend(handles=legend_patches, loc="lower center", fontsize=8,
                   ncol=min(len(legend_patches), 6), framealpha=0.95,
                   edgecolor="#cccccc", fancybox=True,
                   bbox_to_anchor=(0.5, 0.005))

    fig.tight_layout(rect=[0.05, 0.07, 1, 0.93])
    path = os.path.join(outdir, "3_algorithm_evolution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


def fig4_predicted_vs_actual(chunk_rows, outdir):
    """Per-NN-config, per-tensor: predicted vs actual per chunk at each epoch.

    Matches VPIC style: rows = epochs (milestone timesteps), columns = metrics,
    X-axis = chunk index, lines = predicted vs actual.
    One figure per (NN config × tensor).
    """
    nn_chunks = [r for r in chunk_rows if r.get("policy", "-") != "-"
                 and r.get("predicted_ratio", 0) > 0]
    if not nn_chunks:
        print("  Skipping 4_predicted_vs_actual (no NN chunk data)")
        return

    nn_algos = sorted(set(r["algorithm"] for r in nn_chunks))
    epochs = sorted(set(r["epoch"] for r in nn_chunks))
    tensors = [t for t in TENSOR_ORDER if any(r["tensor"] == t for r in nn_chunks)]

    # Pick milestone epochs (up to 5 evenly spaced)
    if len(epochs) <= 5:
        milestones = epochs
    else:
        step = max(1, len(epochs) // 5)
        milestones = epochs[::step]
        if epochs[-1] not in milestones:
            milestones.append(epochs[-1])

    for algo in nn_algos:
        for tensor in tensors:
            at_rows = [r for r in nn_chunks
                       if r["algorithm"] == algo and r["tensor"] == tensor]
            if not at_rows:
                continue

            n_rows = len(milestones)
            fig, axes = plt.subplots(n_rows, 3, figsize=(14, 3 * n_rows + 1),
                                      squeeze=False)
            label = ALGO_LABELS.get(algo, algo).replace("\n", " ")
            fig.suptitle(f"NN Predicted vs Actual Per Chunk — {label} — {tensor}\n"
                         f"(ratio, compression time, decompression time)",
                         fontsize=12, fontweight="bold")

            for ri, ep in enumerate(milestones):
                ep_rows = sorted([r for r in at_rows if r["epoch"] == ep],
                                 key=lambda r: r["chunk_idx"])
                if not ep_rows:
                    for ci in range(3):
                        axes[ri, ci].set_visible(False)
                    continue

                chunks = [r["chunk_idx"] for r in ep_rows]

                # Column 1: Ratio
                ax = axes[ri, 0]
                actual_r = [r["actual_ratio"] for r in ep_rows]
                pred_r = [r["predicted_ratio"] for r in ep_rows]
                ax.fill_between(chunks, actual_r, alpha=0.15, color="#3498db")
                ax.plot(chunks, actual_r, "o-", color="#3498db", markersize=3,
                        linewidth=1, label="Actual")
                ax.plot(chunks, pred_r, "s--", color="#e74c3c", markersize=3,
                        linewidth=1, label="Predicted")
                if actual_r and pred_r:
                    mape = np.mean([abs(a - p) / max(abs(a), 1e-9)
                                    for a, p in zip(actual_r, pred_r)]) * 100
                    ax.text(0.98, 0.95, f"MAPE: {mape:.0f}%",
                            transform=ax.transAxes, ha="right", va="top", fontsize=7)
                ax.set_ylabel(f"E={ep}", fontsize=9, fontweight="bold")
                if ri == 0:
                    ax.set_title("Compression Ratio (x)", fontweight="bold")
                    ax.legend(fontsize=7)
                if ri == n_rows - 1:
                    ax.set_xlabel("Chunk Index")

                # Column 2: Compression Time
                ax = axes[ri, 1]
                actual_c = [r["comp_ms"] for r in ep_rows]
                pred_c = [r["predicted_comp_time"] for r in ep_rows]
                ax.fill_between(chunks, actual_c, alpha=0.15, color="#3498db")
                ax.plot(chunks, actual_c, "o-", color="#3498db", markersize=3, linewidth=1)
                ax.plot(chunks, pred_c, "s--", color="#e74c3c", markersize=3, linewidth=1)
                if actual_c and pred_c:
                    mape = np.mean([abs(a - p) / max(abs(a), 1e-9)
                                    for a, p in zip(actual_c, pred_c)]) * 100
                    ax.text(0.98, 0.95, f"MAPE: {mape:.0f}%",
                            transform=ax.transAxes, ha="right", va="top", fontsize=7)
                if ri == 0:
                    ax.set_title("Comp Time (ms)", fontweight="bold")
                if ri == n_rows - 1:
                    ax.set_xlabel("Chunk Index")

                # Column 3: Decompression Time
                ax = axes[ri, 2]
                actual_d = [r["decomp_ms"] for r in ep_rows]
                pred_d = [r["predicted_decomp_time"] for r in ep_rows]
                ax.fill_between(chunks, actual_d, alpha=0.15, color="#3498db")
                ax.plot(chunks, actual_d, "o-", color="#3498db", markersize=3, linewidth=1)
                ax.plot(chunks, pred_d, "s--", color="#e74c3c", markersize=3, linewidth=1)
                if actual_d and pred_d:
                    mape = np.mean([abs(a - p) / max(abs(a), 1e-9)
                                    for a, p in zip(actual_d, pred_d)]) * 100
                    ax.text(0.98, 0.95, f"MAPE: {mape:.0f}%",
                            transform=ax.transAxes, ha="right", va="top", fontsize=7)
                if ri == 0:
                    ax.set_title("Decomp Time (ms)", fontweight="bold")
                if ri == n_rows - 1:
                    ax.set_xlabel("Chunk Index")

            fig.tight_layout(rect=[0, 0, 1, 0.94])
            path = os.path.join(outdir, f"4_{algo}_{tensor}_predicted_vs_actual.png")
            fig.savefig(path, dpi=200)
            plt.close(fig)
            print(f"  Saved {path}")


def fig5a_sgd_convergence(rows, outdir, subtitle=""):
    """MAPE over epochs for NN-RL configs."""
    epochs = sorted(set(r["epoch"] for r in rows))
    if len(epochs) < 2:
        print("  Skipping 5a_sgd_convergence (need >= 2 epochs)")
        return

    nn_rows = [r for r in rows if r.get("mode", "-") in ("nn-rl", "nn-rl+exp50")]
    if not nn_rows:
        print("  Skipping 5a_sgd_convergence (no RL configs)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _t = "NN Cost Model Convergence (MAPE % over Epochs)"
    if subtitle:
        _t += f"\n{subtitle}"
    fig.suptitle(_t, fontsize=13)

    metrics = [("mape_ratio_pct", "Ratio MAPE %"),
               ("mape_comp_pct", "Compression Time MAPE %"),
               ("mape_decomp_pct", "Decompression Time MAPE %")]

    for ax, (metric, title) in zip(axes, metrics):
        for algo in NN_ALGOS:
            arows = [r for r in nn_rows if r["algorithm"] == algo]
            if not arows:
                continue
            # Average across tensors per epoch
            ep_vals = defaultdict(list)
            for r in arows:
                ep_vals[r["epoch"]].append(r[metric])
            xs = sorted(ep_vals.keys())
            ys = [np.mean(ep_vals[e]) for e in xs]
            ax.plot(xs, ys, "o-", label=algo, color=ALGO_COLORS.get(algo, "gray"), markersize=5)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAPE %")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(epochs)

    fig.tight_layout()
    path = os.path.join(outdir, "5a_sgd_convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def fig5b_sgd_exploration_firing(rows, outdir, subtitle=""):
    """SGD fires + explorations per epoch for NN configs."""
    epochs = sorted(set(r["epoch"] for r in rows))
    nn_rows = [r for r in rows if r.get("policy", "-") != "-"]
    if not nn_rows:
        print("  Skipping 5b (no NN configs)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _t = "SGD Fires & Explorations Over Epochs"
    if subtitle:
        _t += f"\n{subtitle}"
    fig.suptitle(_t, fontsize=13)

    for ax, (metric, title) in zip(axes, [("sgd_fires", "SGD Fires"), ("explorations", "Explorations")]):
        for algo in NN_ALGOS:
            arows = [r for r in nn_rows if r["algorithm"] == algo]
            if not arows:
                continue
            ep_vals = defaultdict(list)
            for r in arows:
                ep_vals[r["epoch"]].append(r[metric])
            xs = sorted(ep_vals.keys())
            ys = [sum(ep_vals[e]) for e in xs]
            ax.plot(xs, ys, "o-", label=algo, color=ALGO_COLORS.get(algo, "gray"), markersize=5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Count (total across tensors)")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(epochs)

    fig.tight_layout()
    path = os.path.join(outdir, "5b_sgd_exploration_firing.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def fig5c_mae_over_time(rows, outdir, subtitle=""):
    """MAE over epochs for NN-RL configs."""
    epochs = sorted(set(r["epoch"] for r in rows))
    if len(epochs) < 2:
        print("  Skipping 5c (need >= 2 epochs)")
        return

    nn_rows = [r for r in rows if r.get("mode", "-") in ("nn-rl", "nn-rl+exp50")]
    if not nn_rows:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _t = "Mean Absolute Error Over Epochs"
    if subtitle:
        _t += f"\n{subtitle}"
    fig.suptitle(_t, fontsize=13)

    metrics = [("mae_ratio", "Ratio MAE"),
               ("mae_comp_ms", "Compression MAE (ms)"),
               ("mae_decomp_ms", "Decompression MAE (ms)")]

    for ax, (metric, title) in zip(axes, metrics):
        for algo in NN_ALGOS:
            arows = [r for r in nn_rows if r["algorithm"] == algo]
            if not arows:
                continue
            ep_vals = defaultdict(list)
            for r in arows:
                ep_vals[r["epoch"]].append(r[metric])
            xs = sorted(ep_vals.keys())
            ys = [np.mean(ep_vals[e]) for e in xs]
            ax.plot(xs, ys, "o-", label=algo, color=ALGO_COLORS.get(algo, "gray"), markersize=5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title.split("(")[0].strip())
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(epochs)

    fig.tight_layout()
    path = os.path.join(outdir, "5c_mae_over_time.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def fig6b_pipeline_waterfall(rows, outdir, epoch=None, subtitle=""):
    """Pipeline stage timing breakdown: stage1, drain, io_drain."""
    if epoch is None:
        epoch = max(r["epoch"] for r in rows)
    erows = [r for r in rows if r["epoch"] == epoch and r.get("pipeline_ms", 0) > 0]
    if not erows:
        print("  Skipping 6b (no pipeline timing data)")
        return

    algos = _get_algos(erows)
    algos = [a for a in algos if a != "no-comp"]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Average across tensors
    stage1 = [np.mean([r["stage1_ms"] for r in erows if r["algorithm"] == a]) for a in algos]
    drain = [np.mean([r["drain_ms"] for r in erows if r["algorithm"] == a]) for a in algos]
    io_drain = [np.mean([r["io_drain_ms"] for r in erows if r["algorithm"] == a]) for a in algos]

    x = np.arange(len(algos))
    ax.bar(x, stage1, label="Stage 1 (NN inference)", color="#1f77b4")
    ax.bar(x, drain, bottom=stage1, label="Stage 2 drain (compress workers)", color="#ff7f0e")
    ax.bar(x, io_drain, bottom=np.array(stage1) + np.array(drain),
           label="Stage 3 drain (I/O)", color="#2ca02c")

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Time (ms)")
    _t = f"VOL Pipeline Stage Breakdown (Epoch {epoch})"
    if subtitle:
        _t += f"\n{subtitle}"
    ax.set_title(_t)
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=40, ha="right", fontsize=8)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, "6b_pipeline_waterfall.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def fig6c_gpu_breakdown(rows, outdir, epoch=None, subtitle=""):
    """Component timing: NN, stats, preproc, comp, decomp, SGD, explore."""
    if epoch is None:
        epoch = max(r["epoch"] for r in rows)
    erows = [r for r in rows if r["epoch"] == epoch]
    nn_erows = [r for r in erows if r.get("policy", "-") != "-"]
    if not nn_erows:
        print("  Skipping 6c (no NN configs)")
        return

    algos = [a for a in NN_ALGOS if any(r["algorithm"] == a for r in nn_erows)]

    components = [
        ("nn_ms", "NN Inference", "#1f77b4"),
        ("stats_ms", "Statistics", "#ff7f0e"),
        ("preproc_ms", "Preprocessing", "#2ca02c"),
        ("comp_ms", "Compression", "#d62728"),
        ("decomp_ms", "Decompression", "#9467bd"),
        ("sgd_ms", "SGD Update", "#8c564b"),
        ("explore_ms", "Exploration", "#e377c2"),
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(algos))
    bottom = np.zeros(len(algos))

    for key, label, color in components:
        vals = [np.mean([r[key] for r in nn_erows if r["algorithm"] == a]) for a in algos]
        ax.bar(x, vals, bottom=bottom, label=label, color=color)
        bottom += np.array(vals)

    ax.set_xlabel("NN Config")
    ax.set_ylabel("Time (ms)")
    _t = f"GPU Component Timing Breakdown (Epoch {epoch})"
    if subtitle:
        _t += f"\n{subtitle}"
    ax.set_title(_t)
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=30, ha="right")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, "6c_gpu_breakdown_over_time.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def fig6d_pipeline_overhead(rows, outdir, epoch=None, subtitle=""):
    """Cross-phase pipeline overhead: s2_busy vs s3_busy."""
    if epoch is None:
        epoch = max(r["epoch"] for r in rows)
    erows = [r for r in rows if r["epoch"] == epoch and r.get("pipeline_ms", 0) > 0]
    if not erows:
        print("  Skipping 6d (no pipeline data)")
        return

    algos = _get_algos(erows)
    algos = [a for a in algos if a != "no-comp"]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(algos))
    width = 0.35

    s2 = [np.mean([r["s2_busy_ms"] for r in erows if r["algorithm"] == a]) for a in algos]
    s3 = [np.mean([r["s3_busy_ms"] for r in erows if r["algorithm"] == a]) for a in algos]

    ax.bar(x - width/2, s2, width, label="S2 busy (compress workers)", color="#ff7f0e")
    ax.bar(x + width/2, s3, width, label="S3 busy (I/O writes)", color="#2ca02c")

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Time (ms)")
    _t = f"Pipeline Bottleneck: Compress vs I/O (Epoch {epoch})"
    if subtitle:
        _t += f"\n{subtitle}"
    ax.set_title(_t)
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=40, ha="right", fontsize=8)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, "6d_cross_phase_pipeline_overhead.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def fig7_epoch_evolution(rows, outdir):
    """Ratio and throughput over epochs per algorithm."""
    epochs = sorted(set(r["epoch"] for r in rows))
    if len(epochs) < 2:
        print("  Skipping 7_epoch_evolution (need >= 2 epochs)")
        return

    algos = _get_algos(rows)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Compression Metrics Over Training Epochs", fontsize=14)

    for idx, tensor in enumerate(["weights", "adam_v"]):
        trows = [r for r in rows if r["tensor"] == tensor]

        # Ratio
        ax = axes[0, idx]
        for a in algos:
            arows = [r for r in trows if r["algorithm"] == a]
            if not arows:
                continue
            ep = defaultdict(list)
            for r in arows:
                ep[r["epoch"]].append(r["ratio"])
            xs = sorted(ep.keys())
            ys = [np.mean(ep[e]) for e in xs]
            ax.plot(xs, ys, "o-", label=a, color=ALGO_COLORS.get(a, "gray"), markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Compression Ratio")
        ax.set_title(f"{tensor} — Ratio Over Epochs")
        ax.legend(fontsize=6, loc="best", ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(epochs)

        # Write throughput
        ax = axes[1, idx]
        for a in algos:
            arows = [r for r in trows if r["algorithm"] == a]
            if not arows:
                continue
            ep = defaultdict(list)
            for r in arows:
                ep[r["epoch"]].append(r["write_mbps"])
            xs = sorted(ep.keys())
            ys = [np.mean(ep[e]) for e in xs]
            ax.plot(xs, ys, "o-", label=a, color=ALGO_COLORS.get(a, "gray"), markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Write Throughput (MB/s)")
        ax.set_title(f"{tensor} — Write Throughput Over Epochs")
        ax.legend(fontsize=6, loc="best", ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(epochs)

    fig.tight_layout()
    path = os.path.join(outdir, "7_epoch_evolution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot inline benchmark results")
    parser.add_argument("csv_path", help="Path to inline_benchmark.csv")
    parser.add_argument("--outdir", default=None,
                        help="Output directory for plots (default: same as CSV)")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Epoch to plot for single-epoch figures (default: latest)")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"ERROR: {args.csv_path} not found")
        sys.exit(1)

    outdir = args.outdir or os.path.dirname(args.csv_path)
    os.makedirs(outdir, exist_ok=True)

    all_rows = load_csv(args.csv_path)
    epochs = sorted(set(r["epoch"] for r in all_rows))
    algos = sorted(set(r["algorithm"] for r in all_rows))
    tensors = sorted(set(r["tensor"] for r in all_rows))
    print(f"Loaded {len(all_rows)} rows: epochs={epochs}, algos={len(algos)}, tensors={tensors}")

    # Load per-chunk CSV if available
    chunk_csv_path = args.csv_path.replace(".csv", "_chunks.csv")
    all_chunk_rows = load_chunk_csv(chunk_csv_path)
    if all_chunk_rows:
        print(f"Loaded {len(all_chunk_rows)} chunk rows from {chunk_csv_path}")

    # Derive chunk size from data (n_chunks > 0 rows)
    _chunk_rows_with = [r for r in all_rows if r["n_chunks"] > 0]
    if _chunk_rows_with:
        _orig = _chunk_rows_with[0]["orig_bytes"]
        _nchunks = _chunk_rows_with[0]["n_chunks"]
        chunk_mb = int(round(_orig / _nchunks / (1024 * 1024)))
    else:
        chunk_mb = 0
    # Detect lossless vs lossy from directory name or data
    _has_lossy = any("lossy" in str(r.get("mode", "")) for r in all_rows)
    _mode_tag = "lossy" if _has_lossy else "lossless"

    def generate_set(rows, chunk_rows, out, label):
        """Generate all figures for a filtered row set."""
        ctx = f"{label} | {chunk_mb}MB chunks | {_mode_tag}" if chunk_mb else label
        print(f"\n--- {label} ({len(rows)} rows) ---")
        os.makedirs(out, exist_ok=True)
        fig1_summary(rows, out, args.epoch, subtitle=ctx)
        if chunk_rows:
            fig3_algorithm_evolution(chunk_rows, out, subtitle=ctx)
            fig4_predicted_vs_actual(chunk_rows, out)
        fig5a_sgd_convergence(rows, out, subtitle=ctx)
        fig5b_sgd_exploration_firing(rows, out, subtitle=ctx)
        fig5c_mae_over_time(rows, out, subtitle=ctx)
        fig6b_pipeline_waterfall(rows, out, args.epoch, subtitle=ctx)
        fig6c_gpu_breakdown(rows, out, args.epoch, subtitle=ctx)
        fig6d_pipeline_overhead(rows, out, args.epoch, subtitle=ctx)
        # fig7_epoch_evolution removed

    # ── Per-tensor × per-policy split ──
    tensors_present = [t for t in TENSOR_ORDER if any(r["tensor"] == t for r in all_rows)]

    policy_splits = [
        ("balanced",   set(FIXED_ALGOS + ["bal_nn", "bal_rl", "bal_exp"]),           "bal_"),
        ("ratio",      set(FIXED_ALGOS + ["rat_nn", "rat_rl", "rat_exp"]),           "rat_"),
    ]

    for tensor in tensors_present:
        t_rows = [r for r in all_rows if r["tensor"] == tensor]
        t_chunks = [r for r in all_chunk_rows if r["tensor"] == tensor] if all_chunk_rows else []

        for pol_name, pol_algos, pol_prefix in policy_splits:
            if not any(r["algorithm"].startswith(pol_prefix) for r in all_rows):
                continue
            pt_rows = [r for r in t_rows if r["algorithm"] in pol_algos]
            pt_chunks = [r for r in t_chunks if r["algorithm"] in pol_algos] if t_chunks else []
            pt_dir = os.path.join(outdir, tensor, pol_name)
            generate_set(pt_rows, pt_chunks, pt_dir, f"{tensor} / {pol_name}")

    print(f"\nDone. {len(os.listdir(outdir))} files in {outdir}/")


if __name__ == "__main__":
    main()
