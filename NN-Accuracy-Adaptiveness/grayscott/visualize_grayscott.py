#!/usr/bin/env python3
"""
Gray-Scott Adaptiveness Benchmark Visualizer.

Reads CSV results from benchmark_grayscott_vol and produces:
  1. Summary bar chart: Exhaustive vs Baseline vs Best SGD aggregate ratios
  2. SGD heatmap: LR vs MAPE threshold, color = compression ratio
  3. SGD convergence: Per-chunk MAPE curves, one PNG per LR
  4. Per-chunk ratio comparison: Exhaustive vs Baseline vs Best SGD line plots
  5. Per-chunk config & ratio: 3-subplot breakdown (oracle/baseline/SGD)
  6. Upper-bound configs: Side-by-side bars per chunk
  7. Config cross-check: Heatmap showing algorithm choice alignment

Usage:
  python3 benchmarks/grayscott/visualize_grayscott.py [options]

  --sgd-csv PATH      SGD study CSV (default: benchmarks/grayscott/sgd_study.csv)
  --chunks-csv PATH   Chunks detail CSV (default: benchmarks/grayscott/benchmark_grayscott_vol_chunks.csv)
  --ub-csv PATH       Upper bound CSV (default: benchmarks/grayscott/upper_bound.csv)
  --agg-csv PATH      Aggregate CSV (default: benchmarks/grayscott/benchmark_grayscott_vol.csv)
  --output-dir DIR    Output directory (default: benchmarks/grayscott/)
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np


# -- CSV parsing (no pandas dependency) ----------------------------------------

def parse_csv(path):
    """Parse a CSV file into a list of dicts with auto float conversion."""
    rows = []
    with open(path) as f:
        header = [h.strip() for h in f.readline().split(",")]
        for line in f:
            vals = [v.strip() for v in line.split(",")]
            if len(vals) != len(header):
                continue
            row = {}
            for h, v in zip(header, vals):
                try:
                    row[h] = float(v)
                except ValueError:
                    row[h] = v
            rows.append(row)
    return rows


def g(row, *keys, default=0.0):
    """Get first matching key from row, with default."""
    for k in keys:
        if k in row:
            return row[k]
    return default


# -- Constants -----------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(_SCRIPT_DIR, "results")

ALGO_COLORS = {
    "lz4":       "#e74c3c",
    "snappy":    "#3498db",
    "deflate":   "#2ecc71",
    "gdeflate":  "#1abc9c",
    "zstd":      "#9b59b6",
    "ans":       "#e67e22",
    "cascaded":  "#f39c12",
    "bitcomp":   "#34495e",
}

ALGO_NAMES_BY_ID = {
    0: "lz4", 1: "snappy", 2: "deflate", 3: "gdeflate",
    4: "zstd", 5: "ans", 6: "cascaded", 7: "bitcomp",
}

MAPE_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#e67e22", "#9b59b6",
               "#1abc9c", "#34495e", "#f39c12"]


# -- Helpers -------------------------------------------------------------------

def _parse_nn_action(action_str):
    """Parse nn_action string -> (algo_name, tag_string).

    The CSV now stores human-readable strings like 'lz4', 'snappy+shuf',
    'zstd+shuf+quant'.  Extract the base algo and return the full tag.
    """
    s = str(action_str).strip()
    algo = s.split("+")[0] if "+" in s else s
    return algo, s


def _get_ub_best_per_chunk(ub_rows):
    """Return dict: chunk -> (best_ratio, tag, algo) from upper_bound.csv."""
    best = {}
    for r in ub_rows:
        if r.get("status") != "ok":
            continue
        ci = int(g(r, "chunk"))
        ratio = g(r, "ratio")
        if ratio > best.get(ci, (0, "", ""))[0]:
            algo = r.get("algorithm", "?")
            shuf = int(g(r, "shuffle"))
            tag = algo
            if shuf:
                tag += f"+s{shuf}"
            best[ci] = (ratio, tag, algo)
    return best


def _get_chunks_by_phase(chunks_rows, phase_prefix):
    """Return dict: chunk -> (ratio, tag, algo) for rows matching phase prefix."""
    result = {}
    for r in chunks_rows:
        phase = r.get("phase", "")
        if isinstance(phase, str) and phase.startswith(phase_prefix):
            ci = int(g(r, "chunk"))
            algo, tag = _parse_nn_action(r.get("nn_action", ""))
            result[ci] = (g(r, "actual_ratio"), tag, algo)
    return result


def _get_sgd_phase_chunks(chunks_rows, lr, mt):
    """Return dict: chunk -> (ratio, tag, algo) for a specific SGD config."""
    phase = f"sgd_{lr:.2f}_{mt:.2f}"
    result = {}
    for r in chunks_rows:
        if r.get("phase") == phase:
            ci = int(g(r, "chunk"))
            algo, tag = _parse_nn_action(r.get("nn_action", ""))
            result[ci] = (g(r, "actual_ratio"), tag, algo)
    return result


def _log_percent_formatter(val, _):
    if val >= 1e6:
        return f"{val / 1e6:.0f}M%"
    if val >= 1e3:
        return f"{val / 1e3:.0f}K%"
    return f"{val:.0f}%"


def _ratio_formatter(val, _):
    if val < 10:
        return f"{val:.1f}x"
    return f"{val:.0f}x"


# -- Plot 1: Summary Bar Chart ------------------------------------------------

def plot_summary(agg_rows, sgd_rows, output_dir):
    """Bar chart: Exhaustive vs Baseline vs Best SGD aggregate ratios."""
    oracle_ratio = 0
    baseline_ratio = 0
    for r in agg_rows:
        if r.get("phase") in ("oracle", "exhaustive"):
            oracle_ratio = g(r, "ratio")
        elif r.get("phase") == "baseline":
            baseline_ratio = g(r, "ratio")

    best_sgd_ratio = 0
    best_sgd_label = ""
    if sgd_rows:
        best = max(sgd_rows, key=lambda r: g(r, "ratio"))
        best_sgd_ratio = g(best, "ratio")
        best_sgd_label = f"lr={g(best,'lr'):.2f}, mt={g(best,'mape_threshold'):.2f}"

    labels = ["Exhaustive\n(best-static)"]
    values = [oracle_ratio]
    colors = ["#e74c3c"]

    labels.append("Baseline\n(NN only)")
    values.append(baseline_ratio)
    colors.append("#2980b9")

    if best_sgd_ratio > 0:
        labels.append(f"Best SGD\n({best_sgd_label})")
        values.append(best_sgd_ratio)
        colors.append("#27ae60")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white",
                  linewidth=1.5, alpha=0.85, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}x", ha="center", va="bottom", fontsize=12,
                fontweight="bold")

    ax.set_ylabel("Compression Ratio", fontsize=12)
    fig.suptitle("Gray-Scott NN Adaptiveness: Aggregate Compression Ratio", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.text(0.5, 0.99,
             "Compares exhaustive (per-chunk search), NN baseline (inference-only),\n"
             "and best SGD config (online learning) on Gray-Scott reaction-diffusion data.",
             ha="center", fontsize=8.5, color="#555", va="top", style="italic")
    ax.set_title("")
    ax.grid(True, alpha=0.3, axis="y")

    if oracle_ratio > 0 and baseline_ratio > 0:
        pct = baseline_ratio / oracle_ratio * 100
        ax.text(0.98, 0.95, f"Baseline = {pct:.1f}% of Exhaustive",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                color="#555", style="italic")

    fig.tight_layout()
    path = os.path.join(output_dir, "summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Plot 2: SGD Heatmap ------------------------------------------------------

def plot_sgd_heatmap(sgd_rows, output_dir):
    """Heatmap: LR vs MAPE threshold, color = compression ratio."""
    lrs = sorted(set(g(r, "lr") for r in sgd_rows))
    mapes = sorted(set(g(r, "mape_threshold") for r in sgd_rows))
    if not lrs or not mapes:
        return

    # Ratio heatmap
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax_idx, (col, label, cmap) in enumerate([
        ("ratio", "Compression Ratio", "YlGn"),
        ("final_mape", "Final MAPE (%)", "RdYlGn_r"),
    ]):
        ax = axes[ax_idx]
        grid = np.full((len(lrs), len(mapes)), np.nan)
        for r in sgd_rows:
            li = lrs.index(g(r, "lr"))
            mi = mapes.index(g(r, "mape_threshold"))
            grid[li, mi] = g(r, col)

        im = ax.imshow(grid, aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(len(mapes)))
        ax.set_xticklabels([f"{m:.2f}" for m in mapes], fontsize=9)
        ax.set_yticks(range(len(lrs)))
        ax.set_yticklabels([f"{lr:.3f}" for lr in lrs], fontsize=9)
        ax.set_xlabel("MAPE Threshold")
        ax.set_ylabel("Learning Rate")
        ax.set_title(label, fontsize=12)
        fig.colorbar(im, ax=ax, shrink=0.8)

        for i in range(len(lrs)):
            for j in range(len(mapes)):
                if not np.isnan(grid[i, j]):
                    val = grid[i, j]
                    fmt = f"{val:.2f}x" if ax_idx == 0 else f"{val:.1f}%"
                    ax.text(j, i, fmt, ha="center", va="center",
                            fontsize=8, color="white" if val > np.nanmean(grid) else "black")

    fig.suptitle("SGD Hyperparameter Study: Gray-Scott Data", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.95,
             "Grid search over SGD learning rate and MAPE threshold.\n"
             "Left: compression ratio achieved. Right: final prediction error (MAPE %).",
             ha="center", fontsize=8.5, color="#555", va="top", style="italic")
    fig.tight_layout()
    path = os.path.join(output_dir, "sgd_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Plot 3: SGD Convergence --------------------------------------------------

def plot_sgd_convergence(chunks_rows, sgd_rows, output_dir):
    """One PNG per LR, each with one subplot per MAPE threshold.
    Shows per-chunk MAPE with convergence markers."""
    # Collect all SGD chunk rows
    sgd_phases = {}  # (lr, mt) -> [rows sorted by chunk]
    for r in chunks_rows:
        phase = r.get("phase", "")
        if not isinstance(phase, str) or not phase.startswith("sgd_"):
            continue
        # Parse lr and mt from phase name like "sgd_0.25_0.15"
        parts = phase.split("_")
        if len(parts) < 3:
            continue
        try:
            lr = float(parts[1])
            mt = float(parts[2])
        except ValueError:
            continue
        key = (lr, mt)
        if key not in sgd_phases:
            sgd_phases[key] = []
        sgd_phases[key].append(r)

    if not sgd_phases:
        return

    lrs = sorted(set(k[0] for k in sgd_phases))
    mapes = sorted(set(k[1] for k in sgd_phases))

    for lr in lrs:
        n_mt = len(mapes)
        fig, axes = plt.subplots(n_mt, 1, figsize=(14, 4 * n_mt), sharex=True)
        if n_mt == 1:
            axes = [axes]
        fig.suptitle(f"SGD Convergence on Gray-Scott Data — LR = {lr:.3f}",
                     fontsize=14, fontweight="bold", y=1.03)
        fig.text(0.5, 1.01,
                 "Per-chunk MAPE as SGD processes each chunk sequentially.\n"
                 "Green vertical line marks convergence point; dashed horizontal line is the MAPE threshold.",
                 ha="center", fontsize=8.5, color="#555", va="top", style="italic")

        for mi, mt in enumerate(mapes):
            ax = axes[mi]
            key = (lr, mt)
            rows = sgd_phases.get(key, [])
            rows.sort(key=lambda r: int(g(r, "chunk")))

            if not rows:
                ax.set_title(f"MT = {mt:.2f}  (no data)", fontsize=11)
                continue

            # Compute per-chunk MAPE
            indices, mape_vals = [], []
            for r in rows:
                pred = g(r, "predicted_ratio")
                actual = g(r, "actual_ratio")
                if pred > 0 and actual > 0:
                    ape = abs(pred - actual) / actual * 100.0
                    indices.append(int(g(r, "chunk")))
                    mape_vals.append(ape)

            if indices:
                ax.plot(indices, mape_vals,
                        color=MAPE_COLORS[mi % len(MAPE_COLORS)],
                        linewidth=1.5, alpha=0.9)

                # Threshold line at mt*100%
                thresh_pct = mt * 100
                ax.axhline(y=thresh_pct, color="gray", linestyle="--",
                           alpha=0.4, linewidth=0.8)
                ax.text(max(indices) * 0.98, thresh_pct,
                        f"{thresh_pct:.0f}%", fontsize=7,
                        ha="right", va="bottom", color="gray", alpha=0.6)

            # Find convergence from sgd_rows
            matching = [r for r in sgd_rows
                        if abs(g(r, "lr") - lr) < 1e-6
                        and abs(g(r, "mape_threshold") - mt) < 1e-6]
            conv_chunk = int(g(matching[0], "convergence_chunks")) if matching else 0
            final_mape = g(matching[0], "final_mape") if matching else 0
            ratio = g(matching[0], "ratio") if matching else 0
            sgd_fires = int(g(matching[0], "sgd_fire_rate") * g(matching[0], "n_chunks")) if matching else 0

            title = f"MT = {mt:.2f}  |  ratio={ratio:.2f}x  mape={final_mape:.1f}%  sgd_fires={sgd_fires}"
            if conv_chunk > 0:
                title += f"  conv@chunk={conv_chunk}"
                ax.axvline(x=conv_chunk, color="green", linestyle=":",
                           alpha=0.6, linewidth=1)
            ax.set_title(title, fontsize=11)
            ax.set_ylabel("Per-Chunk MAPE (%)")
            ax.set_yscale("log")
            ax.set_ylim(bottom=1)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(_log_percent_formatter))
            ax.grid(True, alpha=0.3, which="both")

        axes[-1].set_xlabel("Chunk Index")
        fig.tight_layout()
        path = os.path.join(output_dir, f"sgd_convergence_lr{lr:.3f}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


# -- Plot 4: Per-Chunk Ratio Comparison ----------------------------------------

def plot_per_chunk_comparison(chunks_rows, sgd_rows, ub_rows, output_dir):
    """Line plot: Exhaustive vs Baseline vs Best SGD per-chunk ratios."""
    ub_best = _get_ub_best_per_chunk(ub_rows)
    bl = _get_chunks_by_phase(chunks_rows, "baseline")

    if not ub_best and not bl:
        return

    # Find best SGD config
    best_sgd = {}
    sgd_label = ""
    if sgd_rows:
        best = max(sgd_rows, key=lambda r: g(r, "ratio"))
        best_lr = g(best, "lr")
        best_mt = g(best, "mape_threshold")
        sgd_label = f"lr={best_lr:.2f}, mt={best_mt:.2f}"
        best_sgd = _get_sgd_phase_chunks(chunks_rows, best_lr, best_mt)

    all_indices = sorted(set(
        list(ub_best.keys()) + list(bl.keys()) + list(best_sgd.keys())
    ))
    if not all_indices:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.array(all_indices, dtype=float)

    if ub_best:
        ax.plot(x, [ub_best.get(i, (0, "", ""))[0] for i in all_indices],
                color="#e74c3c", linewidth=2.0, alpha=0.9,
                label="Exhaustive (best-static)", marker="o", markersize=3, zorder=4)
    if bl:
        ax.plot(x, [bl.get(i, (0, "", ""))[0] for i in all_indices],
                color="#2980b9", linewidth=2.0, alpha=0.9,
                label="Baseline (NN inference-only)", marker="s", markersize=3, zorder=3)
    if best_sgd:
        ax.plot(x, [best_sgd.get(i, (0, "", ""))[0] for i in all_indices],
                color="#27ae60", linewidth=2.0, alpha=0.9,
                label=f"Best SGD ({sgd_label})", marker="^", markersize=3, zorder=2)

    ax.set_xlabel("Chunk Index")
    ax.set_ylabel("Compression Ratio")
    fig.suptitle("Per-Chunk Compression Ratio: Gray-Scott Data", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.text(0.5, 0.99,
             "Compression ratio for each data chunk — exhaustive upper bound vs NN baseline vs best SGD.\n"
             "Shows how well the NN tracks the optimal per-chunk compression across the dataset.",
             ha="center", fontsize=8.5, color="#555", va="top", style="italic")
    ax.set_title("")

    # Use log scale if range is large
    all_vals = []
    for i in all_indices:
        for d in [ub_best, bl, best_sgd]:
            v = d.get(i, (0, "", ""))[0]
            if v > 0:
                all_vals.append(v)
    if all_vals and max(all_vals) / max(min(all_vals), 0.01) > 5:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_ratio_formatter))

    if len(all_indices) > 40:
        step = max(1, len(all_indices) // 25)
        ax.set_xticks(all_indices[::step])
    else:
        ax.set_xticks(all_indices)

    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    path = os.path.join(output_dir, "per_chunk_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Plot 5: Per-Chunk Config & Ratio (3-subplot breakdown) --------------------

def plot_per_chunk_config(chunks_rows, sgd_rows, ub_rows, output_dir):
    """3-subplot plot: Oracle / Baseline / Best SGD, markers colored by config."""
    ub_best = _get_ub_best_per_chunk(ub_rows)
    bl = _get_chunks_by_phase(chunks_rows, "baseline")

    best_sgd = {}
    sgd_label = ""
    if sgd_rows:
        best = max(sgd_rows, key=lambda r: g(r, "ratio"))
        best_lr = g(best, "lr")
        best_mt = g(best, "mape_threshold")
        sgd_label = f"lr={best_lr:.2f}, mt={best_mt:.2f}"
        best_sgd = _get_sgd_phase_chunks(chunks_rows, best_lr, best_mt)

    all_indices = sorted(set(
        list(ub_best.keys()) + list(bl.keys()) + list(best_sgd.keys())
    ))
    if not all_indices:
        return

    # Collect all unique config tags for color mapping
    all_tags = set()
    for ci in all_indices:
        for d in [ub_best, bl, best_sgd]:
            if ci in d:
                all_tags.add(d[ci][1])
    tag_list = sorted(all_tags)
    config_cmap = plt.cm.tab10
    tag_colors = {t: config_cmap(i % 10) for i, t in enumerate(tag_list)}

    series = [
        ("Exhaustive (best-static)", ub_best, "#e74c3c", "o"),
        ("Baseline (NN inference-only)", bl, "#2980b9", "s"),
        (f"Best SGD ({sgd_label})" if sgd_label else "Best SGD", best_sgd, "#27ae60", "^"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    for ax, (label, configs, line_color, marker) in zip(axes, series):
        ratios = [configs.get(ci, (0, "", ""))[0] for ci in all_indices]
        tags = [configs.get(ci, (0, "unknown", ""))[1] for ci in all_indices]

        if not any(r > 0 for r in ratios):
            ax.text(0.5, 0.5, f"{label}: no data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="gray")
            ax.set_ylabel("Ratio")
            ax.set_title(label, fontsize=11)
            continue

        # Plot line
        local_x = list(range(len(all_indices)))
        ax.plot(local_x, ratios, color=line_color, linewidth=1.0,
                alpha=0.4, zorder=2)

        # Plot colored markers by config
        for lx, ratio, tag in zip(local_x, ratios, tags):
            if ratio > 0:
                c = tag_colors.get(tag, "gray")
                ax.scatter(lx, ratio, color=c, marker=marker, s=40,
                           edgecolors="white", linewidths=0.3, zorder=3)

        # Config-change annotations
        runs = []
        for i, tag in enumerate(tags):
            if runs and runs[-1][2] == tag:
                runs[-1] = (runs[-1][0], i, tag)
            else:
                runs.append((i, i, tag))

        for start, end, tag in runs:
            if not tag or tag == "unknown":
                continue
            mid = (start + end) / 2.0
            peak = max(ratios[start:end + 1]) if ratios[start:end + 1] else 0
            if peak > 0:
                ax.annotate(tag, (mid, peak), fontsize=6,
                            ha="center", va="bottom", rotation=45,
                            textcoords="offset points", xytext=(0, 5),
                            color=tag_colors.get(tag, "gray"),
                            fontweight="bold", zorder=4)

        avg = np.mean([r for r in ratios if r > 0])
        ax.set_ylabel("Ratio")
        ax.set_title(f"{label}  (avg={avg:.2f}x)", fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

        valid = [r for r in ratios if r > 0]
        if valid and max(valid) / max(min(valid), 0.01) > 5:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(_ratio_formatter))

    axes[-1].set_xlabel("Chunk Index")

    # Config color legend
    legend_handles = [plt.Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=tag_colors.get(t, "gray"),
                      markersize=8, label=t)
                      for t in tag_list if t]
    if legend_handles:
        fig.legend(handles=legend_handles, loc="upper right",
                   fontsize=7, ncol=min(len(legend_handles), 4),
                   title="Config", title_fontsize=8,
                   bbox_to_anchor=(0.99, 0.99))

    fig.suptitle("Per-Chunk Config & Ratio: Gray-Scott Data", fontsize=14,
                 fontweight="bold", y=1.03)
    fig.text(0.5, 1.01,
             "Per-chunk algorithm selection and compression ratio. Markers colored by chosen config.\n"
             "Reveals which algorithm each method picks per chunk and how that affects ratio.",
             ha="center", fontsize=8.5, color="#555", va="top", style="italic")
    fig.tight_layout()
    path = os.path.join(output_dir, "per_chunk_config.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Plot 6: Upper Bound Configs Bar Chart -------------------------------------

def plot_upper_bound_configs(ub_rows, chunks_rows, sgd_rows, output_dir):
    """Side-by-side bars: oracle best vs NN baseline vs best SGD per chunk."""
    ub_best = _get_ub_best_per_chunk(ub_rows)
    if not ub_best:
        return

    bl = _get_chunks_by_phase(chunks_rows, "baseline")

    best_sgd = {}
    sgd_label = ""
    if sgd_rows:
        best = max(sgd_rows, key=lambda r: g(r, "ratio"))
        best_lr = g(best, "lr")
        best_mt = g(best, "mape_threshold")
        sgd_label = f"lr={best_lr:.2f} mt={best_mt:.2f}"
        best_sgd = _get_sgd_phase_chunks(chunks_rows, best_lr, best_mt)

    chunk_indices = sorted(ub_best.keys())
    if not chunk_indices:
        return

    has_sgd = len(best_sgd) > 0
    n_series = 3 if has_sgd else 2
    bar_width = 0.8 / n_series

    fig, ax = plt.subplots(figsize=(max(10, len(chunk_indices) * 0.5), 7))

    UB_COLOR = "#e74c3c"
    NN_COLOR = "#2980b9"
    SGD_COLOR = "#27ae60"

    for ci in chunk_indices:
        ratio = ub_best[ci][0]
        offset = -bar_width if has_sgd else -bar_width / 2
        ax.bar(ci + offset, ratio, width=bar_width, color=UB_COLOR,
               edgecolor="white", linewidth=0.3, alpha=0.85, zorder=3)

    for ci in chunk_indices:
        if ci not in bl:
            continue
        ratio = bl[ci][0]
        offset = 0 if has_sgd else bar_width / 2
        ax.bar(ci + offset, ratio, width=bar_width, color=NN_COLOR,
               edgecolor="white", linewidth=0.3, alpha=0.85, zorder=3)

    if has_sgd:
        for ci in chunk_indices:
            if ci not in best_sgd:
                continue
            ratio = best_sgd[ci][0]
            ax.bar(ci + bar_width, ratio, width=bar_width, color=SGD_COLOR,
                   edgecolor="white", linewidth=0.3, alpha=0.85, zorder=3)

    # Legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=UB_COLOR, alpha=0.85,
                       edgecolor="white", label="Exhaustive (best-static)"),
        plt.Rectangle((0, 0), 1, 1, fc=NN_COLOR, alpha=0.85,
                       edgecolor="white", label="Baseline (NN inference-only)"),
    ]
    if has_sgd:
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc=SGD_COLOR, alpha=0.85,
                           edgecolor="white",
                           label=f"Best SGD ({sgd_label})"))
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right",
              framealpha=0.9)

    # Config annotations on UB bars
    runs = []
    for ci in chunk_indices:
        ratio, tag, _ = ub_best[ci]
        if runs and runs[-1][2] == tag and ci == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], ci, tag, max(runs[-1][3], ratio))
        else:
            runs.append((ci, ci, tag, ratio))

    for start, end, tag, peak_ratio in runs:
        mid = (start + end) / 2.0
        label = f"[{start}] {tag}" if start == end else f"[{start}-{end}] {tag}"
        ax.annotate(label, (mid, peak_ratio), fontsize=5, fontweight="bold",
                    ha="center", va="bottom", rotation=45,
                    textcoords="offset points", xytext=(0, 4),
                    color="#222", zorder=4)

    ax.set_xlabel("Chunk Index")
    ax.set_ylabel("Compression Ratio")
    title = "Exhaustive vs Baseline"
    if has_sgd:
        title += f" vs Best SGD ({sgd_label})"
    title += ": Per Chunk"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.text(0.5, 0.99,
             "Side-by-side compression ratio per chunk. Annotations show the exhaustive search's best-static\n"
             "algorithm config. Compares how close baseline and SGD get to the exhaustive upper bound.",
             ha="center", fontsize=8.5, color="#555", va="top", style="italic")
    ax.set_title("")

    all_ratios = [ub_best[ci][0] for ci in chunk_indices]
    if max(all_ratios) / max(min(all_ratios), 0.01) > 5:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_ratio_formatter))

    ax.set_xticks(chunk_indices)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(output_dir, "upper_bound_configs.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Plot 7: Config Cross-Check -----------------------------------------------

def plot_config_crosscheck(ub_rows, chunks_rows, sgd_rows, output_dir):
    """Heatmap showing algorithm choice per chunk for Exhaustive vs Baseline vs SGD."""
    ub_best = _get_ub_best_per_chunk(ub_rows)
    if not ub_best:
        return

    bl = _get_chunks_by_phase(chunks_rows, "baseline")

    best_sgd = {}
    sgd_label = "Best SGD"
    if sgd_rows:
        best = max(sgd_rows, key=lambda r: g(r, "ratio"))
        best_lr = g(best, "lr")
        best_mt = g(best, "mape_threshold")
        sgd_label = f"Best SGD\n(lr={best_lr:.2f}, mt={best_mt:.2f})"
        best_sgd = _get_sgd_phase_chunks(chunks_rows, best_lr, best_mt)

    chunk_indices = sorted(ub_best.keys())
    if not chunk_indices:
        return
    n_chunks = len(chunk_indices)

    has_sgd = bool(best_sgd)

    rows_data = [("Baseline", bl)]
    if has_sgd:
        rows_data.append((sgd_label, best_sgd))
    rows_data.append(("Exhaustive", ub_best))

    n_rows = len(rows_data)
    row_ub = n_rows - 1

    fig_h = 2.5 + n_rows * 0.9
    fig, ax = plt.subplots(figsize=(max(14, n_chunks * 0.14), fig_h))

    for row_y, (label, per_chunk) in enumerate(rows_data):
        for xi, ci in enumerate(chunk_indices):
            if ci not in per_chunk:
                continue
            _, tag, algo = per_chunk[ci]
            color = ALGO_COLORS.get(algo, "#ccc")
            ax.barh(row_y, 1, left=xi, height=0.8, color=color,
                    edgecolor="white", linewidth=0.5)
            if n_chunks <= 40 or xi % 5 == 0:
                ax.text(xi + 0.5, row_y, tag, ha="center", va="center",
                        fontsize=4, rotation=90, color="white", fontweight="bold")

    # Mark mismatches vs oracle
    for row_y, (label, per_chunk) in enumerate(rows_data):
        if row_y == row_ub:
            continue
        for xi, ci in enumerate(chunk_indices):
            ub_tag = ub_best.get(ci, (0, "", ""))[1]
            row_tag = per_chunk.get(ci, (0, "", ""))[1]
            if ub_tag and row_tag and ub_tag != row_tag:
                ax.plot(xi + 0.5, row_y - 0.5, marker="x", color="#e74c3c",
                        markersize=4, markeredgewidth=1.5, zorder=5)

    ax.set_yticks(list(range(n_rows)))
    ax.set_yticklabels([label for label, _ in rows_data], fontsize=8)
    ax.set_xlim(0, n_chunks)
    ax.set_ylim(-0.8, row_ub + 1.2)

    if n_chunks > 40:
        step = max(1, n_chunks // 25)
        ax.set_xticks(range(0, n_chunks, step))
        ax.set_xticklabels([str(chunk_indices[i]) for i in range(0, n_chunks, step)],
                           fontsize=7)
    else:
        ax.set_xticks(range(n_chunks))
        ax.set_xticklabels([str(ci) for ci in chunk_indices], fontsize=6)
    ax.set_xlabel("Chunk Index", fontsize=10)

    fig.suptitle("Config Cross-Check: Per-Chunk Algorithm Selection",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.text(0.5, 1.01,
             "Each row shows the algorithm chosen per chunk. Red 'x' marks chunks where the method\n"
             "disagrees with the exhaustive search. Fewer mismatches = better NN prediction accuracy.",
             ha="center", fontsize=8.5, color="#555", va="top", style="italic")
    ax.set_title("")

    # Algorithm color legend
    legend_handles = [plt.Rectangle((0, 0), 1, 1, fc=c, edgecolor="white", label=a)
                      for a, c in ALGO_COLORS.items()]
    ax.legend(handles=legend_handles, fontsize=6, loc="upper right", ncol=4,
              title="Algorithm", framealpha=0.9, title_fontsize=7)

    fig.tight_layout()
    path = os.path.join(output_dir, "config_crosscheck.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # Print summary
    n_total = len(chunk_indices)
    nn_match = sum(1 for ci in chunk_indices
                   if ub_best.get(ci, (0, "", ""))[1] ==
                      bl.get(ci, (0, "", ""))[1])
    print(f"    Baseline vs Exhaustive match: {nn_match}/{n_total} "
          f"({100 * nn_match / n_total:.1f}%)")
    if has_sgd:
        sgd_match = sum(1 for ci in chunk_indices
                        if ub_best.get(ci, (0, "", ""))[1] ==
                           best_sgd.get(ci, (0, "", ""))[1])
        print(f"    Best SGD vs Exhaustive match: {sgd_match}/{n_total} "
              f"({100 * sgd_match / n_total:.1f}%)")


# -- Plot 8: Per-Chunk Prediction Error (Avg MAPE) ----------------------------

def plot_fig1_avg_mape(chunks_rows, sgd_rows, output_dir):
    """Per-chunk prediction error: Baseline vs Best SGD MAPE per chunk."""
    # Collect baseline per-chunk MAPE
    bl_mape = {}
    for r in chunks_rows:
        if r.get("phase") == "baseline":
            ci = int(g(r, "chunk"))
            pred = g(r, "predicted_ratio")
            actual = g(r, "actual_ratio")
            if actual > 0:
                bl_mape[ci] = abs(pred - actual) / actual * 100.0

    if not bl_mape:
        return

    # Find best SGD config by max ratio
    best_lr, best_mt = 0, 0
    if sgd_rows:
        best = max(sgd_rows, key=lambda r: g(r, "ratio"))
        best_lr = g(best, "lr")
        best_mt = g(best, "mape_threshold")

    # Collect best SGD per-chunk MAPE
    sgd_mape = {}
    if best_lr > 0:
        phase = f"sgd_{best_lr:.2f}_{best_mt:.2f}"
        for r in chunks_rows:
            if r.get("phase") == phase:
                ci = int(g(r, "chunk"))
                pred = g(r, "predicted_ratio")
                actual = g(r, "actual_ratio")
                if actual > 0:
                    sgd_mape[ci] = abs(pred - actual) / actual * 100.0

    all_indices = sorted(set(list(bl_mape.keys()) + list(sgd_mape.keys())))
    if not all_indices:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.array(all_indices, dtype=float)

    # Oracle line (perfect prediction = 0 MAPE, show as small constant for log)
    oracle_val = 0.01
    ax.axhline(y=oracle_val, color="#e74c3c", linestyle="--", linewidth=1.5,
               alpha=0.7, label="Exhaustive (perfect prediction)", zorder=2)

    # Baseline MAPE
    bl_vals = [bl_mape.get(i, np.nan) for i in all_indices]
    ax.plot(x, bl_vals, color="#2980b9", linewidth=1.5, alpha=0.9,
            label="Baseline (NN inference-only)", marker="s", markersize=3, zorder=3)
    for xi, val in zip(x, bl_vals):
        if not np.isnan(val):
            ax.annotate(f"{val:.1f}", (xi, val), fontsize=6,
                        ha="center", va="bottom", textcoords="offset points",
                        xytext=(0, 3), color="#2980b9")

    # Best SGD MAPE
    if sgd_mape:
        sgd_vals = [sgd_mape.get(i, np.nan) for i in all_indices]
        ax.plot(x, sgd_vals, color="#27ae60", linewidth=1.5, alpha=0.9,
                label=f"Best SGD (lr={best_lr:.2f}, mt={best_mt:.2f})",
                marker="^", markersize=3, zorder=3)
        for xi, val in zip(x, sgd_vals):
            if not np.isnan(val):
                ax.annotate(f"{val:.1f}", (xi, val), fontsize=6,
                            ha="center", va="bottom", textcoords="offset points",
                            xytext=(0, 3), color="#27ae60")

        # Running average line for SGD convergence
        valid_sgd = [(i, sgd_mape[i]) for i in all_indices if i in sgd_mape]
        if len(valid_sgd) > 1:
            running_avg = []
            cumsum = 0.0
            for idx, (ci, val) in enumerate(valid_sgd):
                cumsum += val
                running_avg.append(cumsum / (idx + 1))
            ra_x = [ci for ci, _ in valid_sgd]
            ax.plot(ra_x, running_avg, color="#27ae60", linewidth=2.0,
                    alpha=0.4, linestyle="--", label="SGD Running Avg MAPE",
                    zorder=2)

    ax.set_xlabel("Chunk Index")
    ax.set_ylabel("MAPE (%)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_log_percent_formatter))

    fig.suptitle("Per-Chunk Prediction Error: Baseline vs Best SGD",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.text(0.5, 0.99,
             "MAPE (|predicted - actual| / actual) per chunk on log scale.\n"
             "Shows how online SGD reduces prediction error relative to the static baseline.",
             ha="center", fontsize=8.5, color="#555", va="top", style="italic")
    ax.set_title("")

    if len(all_indices) > 40:
        step = max(1, len(all_indices) // 25)
        ax.set_xticks(all_indices[::step])
    else:
        ax.set_xticks(all_indices)

    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    path = os.path.join(output_dir, "fig1_avg_mape.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Plot 9: Chunk MAPE Comparison (Side-by-Side Bars) -------------------------

def plot_chunk_mape_comparison(chunks_rows, sgd_rows, output_dir):
    """MAPE heatmap: config (y-axis) vs chunk (x-axis), like VPIC style."""
    # Gather all phases present in chunks data
    phases = sorted(set(r.get("phase", "") for r in chunks_rows if r.get("phase")))
    if not phases:
        return

    all_chunks = sorted(set(int(g(r, "chunk")) for r in chunks_rows))
    if not all_chunks:
        return

    # Order: baseline first, then SGD configs sorted naturally
    baseline_phases = [p for p in phases if p == "baseline"]
    sgd_phases = sorted([p for p in phases if p.startswith("sgd_")])
    ordered_phases = baseline_phases + sgd_phases
    if not ordered_phases:
        return

    # Build 2D grid: rows = configs, cols = chunks
    grid = np.full((len(ordered_phases), len(all_chunks)), np.nan)
    ch_idx = {c: i for i, c in enumerate(all_chunks)}

    for r in chunks_rows:
        phase = r.get("phase", "")
        if phase not in ordered_phases:
            continue
        ci = int(g(r, "chunk"))
        pred = g(r, "predicted_ratio")
        actual = g(r, "actual_ratio")
        if actual > 0 and pred > 0:
            mape = abs(pred - actual) / actual * 100.0
            row_i = ordered_phases.index(phase)
            grid[row_i, ch_idx[ci]] = mape

    if np.all(np.isnan(grid)):
        return

    # Pretty labels
    def _label(p):
        if p == "baseline":
            return "Baseline (no SGD)"
        parts = p.split("_")
        if len(parts) == 3:
            return f"SGD lr={parts[1]} mt={parts[2]}"
        return p

    labels = [_label(p) for p in ordered_phases]

    vmax = min(np.nanpercentile(grid, 95), 5000)
    vmax = max(vmax, 2)  # avoid degenerate range
    norm = mcolors.LogNorm(vmin=1, vmax=vmax, clip=True)

    fig, ax = plt.subplots(figsize=(max(12, len(all_chunks) * 0.25),
                                    max(4, len(ordered_phases) * 0.6 + 2)))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn_r", norm=norm,
                   interpolation="nearest")

    # X-axis: chunks
    if len(all_chunks) > 40:
        step = max(1, len(all_chunks) // 25)
        ax.set_xticks(range(0, len(all_chunks), step))
        ax.set_xticklabels([str(all_chunks[i]) for i in range(0, len(all_chunks), step)],
                           fontsize=7)
    else:
        ax.set_xticks(range(len(all_chunks)))
        ax.set_xticklabels([str(c) for c in all_chunks], fontsize=7)
    ax.set_xlabel("Chunk Index", fontsize=10)

    # Y-axis: configs
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_ylabel("Configuration", fontsize=10)

    # Annotate cells if grid is small enough
    if grid.size <= 600:
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid[i, j]
                if not np.isnan(val):
                    txt = f"{val:.0f}" if val < 10000 else f"{val/1000:.0f}K"
                    color = "white" if val > 500 else "black"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=5, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("MAPE (%)", fontsize=10)

    fig.suptitle("Per-Chunk Prediction Error: All Configurations",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.97,
             "Heatmap of MAPE (%) per chunk (x-axis) vs configuration (y-axis).\n"
             "Green = low error, Red = high error. Log scale.",
             ha="center", fontsize=9, color="#555", va="top", style="italic")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, "chunk_mape_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Plot 10: Compression Ratio Aggregate Comparison ---------------------------

def plot_fig4_compression_ratio(agg_rows, sgd_rows, output_dir):
    """Bar chart: Exhaustive vs Baseline vs Best SGD aggregate compression ratio."""
    oracle_ratio, oracle_file, oracle_orig = 0, 0, 0
    baseline_ratio, baseline_file = 0, 0
    best_sgd_ratio, best_sgd_file = 0, 0

    for r in agg_rows:
        phase = r.get("phase")
        if phase in ("oracle", "exhaustive"):
            oracle_ratio = g(r, "ratio")
            oracle_file = g(r, "file_mib")
            oracle_orig = g(r, "orig_mib")
        elif phase == "baseline":
            baseline_ratio = g(r, "ratio")
            baseline_file = g(r, "file_mib")
        elif phase == "best_sgd":
            best_sgd_ratio = g(r, "ratio")
            best_sgd_file = g(r, "file_mib")

    # If best_sgd not in agg, derive from sgd_rows
    if best_sgd_ratio == 0 and sgd_rows:
        best = max(sgd_rows, key=lambda r: g(r, "ratio"))
        best_sgd_ratio = g(best, "ratio")

    if oracle_ratio == 0 and baseline_ratio == 0:
        return

    labels = []
    values = []
    colors = []
    file_sizes = []

    labels.append("Exhaustive\n(best-static)")
    values.append(oracle_ratio)
    colors.append("#e74c3c")
    file_sizes.append(oracle_file)

    labels.append("Baseline\n(NN only)")
    values.append(baseline_ratio)
    colors.append("#2980b9")
    file_sizes.append(baseline_file)

    if best_sgd_ratio > 0:
        labels.append("Best SGD")
        values.append(best_sgd_ratio)
        colors.append("#27ae60")
        file_sizes.append(best_sgd_file)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white",
                  linewidth=1.5, alpha=0.85, width=0.5)

    # Annotate with ratio, file size, and % of exhaustive
    for bar, val, fsize in zip(bars, values, file_sizes):
        x_center = bar.get_x() + bar.get_width() / 2
        y_top = bar.get_height()

        # Ratio value
        ax.text(x_center, y_top + 0.05, f"{val:.2f}x",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

        # File size (if available)
        if fsize > 0:
            ax.text(x_center, y_top * 0.5, f"{fsize:.1f} MiB",
                    ha="center", va="center", fontsize=9, color="white",
                    fontweight="bold")

        # % of exhaustive
        if oracle_ratio > 0 and val > 0:
            pct = val / oracle_ratio * 100
            ax.text(x_center, y_top + 0.35,
                    f"({pct:.1f}% of exhaustive)",
                    ha="center", va="bottom", fontsize=8,
                    color="#555", style="italic")

    # Original data size reference line
    if oracle_orig > 0 and oracle_ratio > 0:
        # Show as a text annotation since units differ
        ax.text(0.98, 0.05,
                f"Original data: {oracle_orig:.1f} MiB",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color="#555", style="italic",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#ccc", alpha=0.9))

    ax.set_ylabel("Compression Ratio", fontsize=12)
    fig.suptitle("Compression Ratio: Exhaustive vs Baseline vs Best SGD",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.text(0.5, 0.99,
             "Aggregate compression ratio comparison across the full Gray-Scott dataset.\n"
             "Shows how close the NN-based methods get to the exhaustive exhaustive upper bound.",
             ha="center", fontsize=8.5, color="#555", va="top", style="italic")
    ax.set_title("")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(output_dir, "fig4_compression_ratio.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gray-Scott Adaptiveness Benchmark Visualizer")
    parser.add_argument("--sgd-csv", default=None)
    parser.add_argument("--chunks-csv", default=None)
    parser.add_argument("--ub-csv", default=None)
    parser.add_argument("--agg-csv", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_DIR)
    args = parser.parse_args()

    sgd_csv = args.sgd_csv or os.path.join(DEFAULT_DIR, "sgd_study.csv")
    chunks_csv = args.chunks_csv or os.path.join(
        DEFAULT_DIR, "benchmark_grayscott_vol_chunks.csv")
    ub_csv = args.ub_csv or os.path.join(DEFAULT_DIR, "upper_bound.csv")
    agg_csv = args.agg_csv or os.path.join(
        DEFAULT_DIR, "benchmark_grayscott_vol.csv")
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    sgd_rows, chunks_rows, ub_rows, agg_rows = [], [], [], []

    if os.path.exists(agg_csv):
        agg_rows = parse_csv(agg_csv)
        print(f"Loaded {len(agg_rows)} aggregate rows from {agg_csv}")
    else:
        print(f"WARNING: Aggregate CSV not found: {agg_csv}")

    if os.path.exists(sgd_csv):
        sgd_rows = parse_csv(sgd_csv)
        print(f"Loaded {len(sgd_rows)} SGD rows from {sgd_csv}")
    else:
        print(f"NOTE: SGD CSV not found: {sgd_csv} (run without --no-sgd)")

    if os.path.exists(chunks_csv):
        chunks_rows = parse_csv(chunks_csv)
        print(f"Loaded {len(chunks_rows)} chunk detail rows from {chunks_csv}")
    else:
        print(f"WARNING: Chunks CSV not found: {chunks_csv}")

    if os.path.exists(ub_csv):
        ub_rows = parse_csv(ub_csv)
        print(f"Loaded {len(ub_rows)} upper bound rows from {ub_csv}")
    else:
        print(f"WARNING: Upper bound CSV not found: {ub_csv}")

    if not agg_rows and not sgd_rows and not chunks_rows and not ub_rows:
        print("ERROR: No data to visualize. Run the benchmark first.")
        sys.exit(1)

    print("\nGenerating plots...")

    # 1. Summary bar chart
    if agg_rows:
        plot_summary(agg_rows, sgd_rows, output_dir)

    # 2. SGD heatmap
    if sgd_rows:
        plot_sgd_heatmap(sgd_rows, output_dir)

    # 3. SGD convergence
    if sgd_rows and chunks_rows:
        plot_sgd_convergence(chunks_rows, sgd_rows, output_dir)

    # 4. Per-chunk ratio comparison
    if chunks_rows or ub_rows:
        plot_per_chunk_comparison(chunks_rows, sgd_rows, ub_rows, output_dir)

    # 5. Per-chunk config & ratio breakdown
    if chunks_rows or ub_rows:
        plot_per_chunk_config(chunks_rows, sgd_rows, ub_rows, output_dir)

    # 6. Upper bound configs bar chart
    if ub_rows:
        plot_upper_bound_configs(ub_rows, chunks_rows, sgd_rows, output_dir)

    # 7. Config cross-check
    if ub_rows and chunks_rows:
        plot_config_crosscheck(ub_rows, chunks_rows, sgd_rows, output_dir)

    # 8. Per-chunk prediction error (avg MAPE)
    if chunks_rows:
        plot_fig1_avg_mape(chunks_rows, sgd_rows, output_dir)

    # 9. Chunk MAPE comparison (side-by-side bars)
    if chunks_rows:
        plot_chunk_mape_comparison(chunks_rows, sgd_rows, output_dir)

    # 10. Compression ratio aggregate comparison
    if agg_rows:
        plot_fig4_compression_ratio(agg_rows, sgd_rows, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
