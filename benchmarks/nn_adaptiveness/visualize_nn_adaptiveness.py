#!/usr/bin/env python3
"""
NN Adaptiveness Benchmark Visualizer.

Reads CSV results from the NN adaptiveness benchmark and produces:
  1. SGD heatmap: LR vs MAPE threshold, color = final MAPE
  2. SGD convergence: Cumulative MAPE vs chunk index with pattern-region bands
  3. Exploration heatmap: K vs threshold, color = ratio vs upper bound
  4. Exploration time-quality scatter: overhead vs ratio improvement
  5. Per-chunk ratio comparison: upper bound vs baseline vs best SGD
  6. Best configs bar chart: upper bound vs baseline vs best SGD vs best SGD+expl

Usage:
  python3 benchmarks/nn_adaptiveness/visualize_nn_adaptiveness.py [options]

  --sgd-csv PATH     SGD study CSV (default: auto-detect)
  --expl-csv PATH    Exploration study CSV (default: auto-detect)
  --chunks-csv PATH  Chunks detail CSV (default: auto-detect)
  --output-dir DIR   Directory for output PNGs (default: benchmarks/nn_adaptiveness/)
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# -- CSV parsing (no pandas dependency) --------------------------------------

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


# -- Constants ---------------------------------------------------------------

PATTERN_NAMES = [
    "ocean_waves",
    "heating_surface",
    "turbulent_flow",
    "geological_strata",
    "particle_shower",
]

PATTERN_LABELS = {
    "ocean_waves":       "Ocean Waves + Spikes",
    "heating_surface":   "Heating Surface + Hotspots",
    "turbulent_flow":    "Turbulent Flow (Gray-Scott)",
    "geological_strata": "Geological Strata",
    "particle_shower":   "Particle Shower",
}

PATTERN_COLORS = {
    "ocean_waves":       "#3498db",
    "heating_surface":   "#e74c3c",
    "turbulent_flow":    "#2ecc71",
    "geological_strata": "#e67e22",
    "particle_shower":   "#9b59b6",
}

MAPE_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#e67e22", "#9b59b6",
               "#1abc9c", "#34495e", "#f39c12"]

DEFAULT_DIR = "benchmarks/nn_adaptiveness"


# -- Helpers -----------------------------------------------------------------

def _compute_per_chunk_mape(chunk_rows):
    """Compute per-chunk MAPE (%) from a sorted list of chunk rows."""
    indices, values = [], []
    for r in chunk_rows:
        pred = g(r, "predicted_ratio")
        actual = g(r, "actual_ratio")
        if pred > 0 and actual > 0:
            ape = abs(pred - actual) / actual * 100.0
            indices.append(int(g(r, "chunk_idx")))
            values.append(ape)
    return indices, values


def _draw_pattern_bands(ax, n_chunks):
    """Draw colored vertical bands for the 5 pattern regions."""
    slab = n_chunks // 5 if n_chunks >= 5 else n_chunks
    for p in range(5):
        z0 = p * slab
        z1 = (p + 1) * slab if p < 4 else n_chunks
        color = PATTERN_COLORS.get(PATTERN_NAMES[p], "#ddd")
        ax.axvspan(z0, z1, alpha=0.10, color=color)
        mid = (z0 + z1) / 2
        ax.text(mid, 1.02, PATTERN_NAMES[p].replace("_", "\n"),
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=5, color=color, fontweight="bold")


def _log_percent_formatter(val, _):
    """Format log-scale MAPE ticks: <1000 as plain int, else compact."""
    if val >= 1e9:
        return f"{val:.0e}%"
    if val >= 1e6:
        return f"{val / 1e6:.0f}M%"
    if val >= 1e3:
        return f"{val / 1e3:.0f}K%"
    return f"{val:.0f}%"


def _get_upper_bound_ratio(ub_rows):
    """Compute overall upper bound ratio from upper_bound.csv rows.

    Each row is one (chunk, config) result. Find best ratio per chunk,
    then compute harmonic mean across chunks.
    """
    if not ub_rows:
        return 0.0
    best = {}  # chunk_idx -> best ratio
    for r in ub_rows:
        idx = int(g(r, "chunk_idx"))
        ratio = g(r, "compression_ratio")
        if ratio > best.get(idx, 0):
            best[idx] = ratio
    if not best:
        return 0.0
    inv_sum = sum(1.0 / r for r in best.values() if r > 0)
    n = sum(1 for r in best.values() if r > 0)
    return n / inv_sum if inv_sum > 0 else 0.0


def _get_upper_bound_per_chunk(ub_rows):
    """Return dict: chunk_idx -> best ratio from upper_bound.csv rows."""
    best = {}
    for r in ub_rows:
        idx = int(g(r, "chunk_idx"))
        ratio = g(r, "compression_ratio")
        if ratio > best.get(idx, 0):
            best[idx] = ratio
    return best


def _get_baseline_per_chunk(chunks_rows):
    """Return dict: chunk_idx -> ratio from baseline rows."""
    bl = {}
    for r in chunks_rows:
        if r.get("study") == "baseline":
            idx = int(g(r, "chunk_idx"))
            bl[idx] = g(r, "actual_ratio")
    return bl


# -- Plot 1: SGD Heatmap ----------------------------------------------------

def plot_sgd_heatmap(sgd_rows, output_dir):
    """Single heatmap: LR vs MAPE threshold, color = final MAPE."""
    lrs = sorted(set(g(r, "lr") for r in sgd_rows))
    mapes = sorted(set(g(r, "mape_threshold") for r in sgd_rows))
    if not lrs or not mapes:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle("SGD Study: Final MAPE (%) by LR x MAPE Threshold", fontsize=13)

    grid = np.full((len(lrs), len(mapes)), np.nan)
    for r in sgd_rows:
        li = lrs.index(g(r, "lr"))
        mi = mapes.index(g(r, "mape_threshold"))
        grid[li, mi] = g(r, "final_mape")

    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn_r", origin="lower")
    ax.set_xticks(range(len(mapes)))
    ax.set_xticklabels([f"{m:.2f}" for m in mapes], fontsize=9)
    ax.set_yticks(range(len(lrs)))
    ax.set_yticklabels([f"{lr:.3f}" for lr in lrs], fontsize=9)
    ax.set_xlabel("MAPE Threshold")
    ax.set_ylabel("Learning Rate")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Final MAPE (%)")

    for i in range(len(lrs)):
        for j in range(len(mapes)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i,j]:.1f}", ha="center", va="center",
                        fontsize=8, color="white" if grid[i, j] > 30 else "black")

    fig.tight_layout()
    path = os.path.join(output_dir, "sgd_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Plot 2: SGD Convergence ------------------------------------------------

def plot_sgd_convergence(chunks_rows, sgd_rows, output_dir):
    """One subplot per learning rate. Each subplot shows all MAPE thresholds
    as separate lines, with pattern-region bands in the background."""
    sgd_chunks = [r for r in chunks_rows if r.get("study") == "sgd"]
    if not sgd_chunks:
        return

    lrs = sorted(set(g(r, "lr") for r in sgd_rows))
    mapes = sorted(set(g(r, "mape_threshold") for r in sgd_rows))
    if not lrs or not mapes:
        return

    max_chunk = max(int(g(r, "chunk_idx")) for r in sgd_chunks)
    n_chunks = max_chunk + 1

    n_lr = len(lrs)
    cols = min(n_lr, 3)
    rows = (n_lr + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    fig.suptitle("SGD Convergence: Per-Chunk MAPE Across Pattern Transitions",
                 fontsize=14, y=1.01)

    for li, lr in enumerate(lrs):
        ax = axes[li // cols][li % cols]
        _draw_pattern_bands(ax, n_chunks)

        for mi, mt in enumerate(mapes):
            cr = [r for r in sgd_chunks
                  if abs(g(r, "lr") - lr) < 1e-6
                  and abs(g(r, "mape_threshold") - mt) < 1e-6]
            cr.sort(key=lambda r: g(r, "chunk_idx"))
            if not cr:
                continue

            indices, running = _compute_per_chunk_mape(cr)
            if indices:
                ax.plot(indices, running,
                        color=MAPE_COLORS[mi % len(MAPE_COLORS)],
                        label=f"mt={mt:.2f}", linewidth=1.3, alpha=0.85)

        ax.set_title(f"LR = {lr:.3f}", fontsize=11)
        ax.set_xlabel("Chunk Index")
        ax.set_ylabel("Per-Chunk MAPE (%)")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_log_percent_formatter))
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        ax.grid(True, alpha=0.3, which="both")

    for i in range(n_lr, rows * cols):
        axes[i // cols][i % cols].set_visible(False)

    fig.tight_layout()
    path = os.path.join(output_dir, "sgd_convergence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Plot 3: Exploration Heatmap ---------------------------------------------

def plot_exploration_heatmap(expl_rows, output_dir):
    """Single heatmap: K vs threshold, color = ratio vs upper bound."""
    thresholds = sorted(set(g(r, "expl_threshold") for r in expl_rows))
    ks = sorted(set(int(g(r, "K")) for r in expl_rows))
    if not thresholds or not ks:
        return

    # Use ratio_vs_upper_bound if available, else fall back to ratio_vs_baseline
    has_ub = any("ratio_vs_upper_bound" in r for r in expl_rows)
    col = "ratio_vs_upper_bound" if has_ub else "ratio_vs_baseline"
    label = "Ratio vs Upper Bound" if has_ub else "Ratio vs Baseline"

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle(f"Exploration Study: {label} by Threshold x K", fontsize=13)

    grid = np.full((len(ks), len(thresholds)), np.nan)
    for r in expl_rows:
        ki = ks.index(int(g(r, "K")))
        ti = thresholds.index(g(r, "expl_threshold"))
        grid[ki, ti] = g(r, col)

    vmax = max(1.3, np.nanmax(grid)) if not np.all(np.isnan(grid)) else 1.3
    vmin = min(0.95, np.nanmin(grid)) if not np.all(np.isnan(grid)) else 0.95
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", origin="lower",
                    vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds], fontsize=9)
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([str(k) for k in ks], fontsize=9)
    ax.set_xlabel("Exploration Threshold")
    ax.set_ylabel("K (alternatives)")
    fig.colorbar(im, ax=ax, shrink=0.8, label=label)

    for i in range(len(ks)):
        for j in range(len(thresholds)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i,j]:.3f}", ha="center", va="center",
                        fontsize=8)

    fig.tight_layout()
    path = os.path.join(output_dir, "exploration_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Plot 4: Exploration Time-Quality Scatter --------------------------------

def plot_exploration_scatter(expl_rows, ub_rows, output_dir):
    """Overhead (ms) vs ratio improvement, Pareto frontier, upper bound ref."""
    if not expl_rows:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("Exploration: Overhead vs Ratio Improvement", fontsize=13)

    x = [g(r, "total_exploration_ms") for r in expl_rows]
    y = [g(r, "ratio_vs_baseline") for r in expl_rows]
    labels = [f"t={g(r,'expl_threshold'):.2f} K={int(g(r,'K'))}" for r in expl_rows]

    ax.scatter(x, y, color="#3498db", alpha=0.7, s=50, edgecolors="white", linewidths=0.5)

    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), fontsize=6, alpha=0.7,
                    textcoords="offset points", xytext=(5, 3))

    # Pareto frontier
    if x and y:
        points = sorted(zip(x, y))
        pareto_x, pareto_y = [], []
        best_y = -float("inf")
        for px, py in points:
            if py > best_y:
                pareto_x.append(px)
                pareto_y.append(py)
                best_y = py
        if pareto_x:
            ax.plot(pareto_x, pareto_y, "k--", alpha=0.5, label="Pareto frontier")

    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, label="Baseline")

    # Upper bound reference line
    ub_ratio = _get_upper_bound_ratio(ub_rows)
    if ub_ratio > 0:
        best_expl = max(expl_rows, key=lambda r: g(r, "ratio"))
        rvb = g(best_expl, "ratio_vs_baseline")
        if rvb > 0:
            baseline_ratio = g(best_expl, "ratio") / rvb
            if baseline_ratio > 0:
                ub_vs_baseline = ub_ratio / baseline_ratio
                ax.axhline(y=ub_vs_baseline, color="#2ecc71", linestyle="--",
                           alpha=0.7, label=f"Upper Bound ({ub_vs_baseline:.2f}x)")

    ax.set_xlabel("Exploration Overhead (ms)")
    ax.set_ylabel("Ratio vs Baseline")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "exploration_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Plot 5: Per-Chunk Ratio Comparison --------------------------------------

def plot_per_chunk_comparison(chunks_rows, sgd_rows, ub_rows, output_dir):
    """Per-chunk ratio: upper bound vs baseline vs best SGD config."""
    ub = _get_upper_bound_per_chunk(ub_rows)
    bl = _get_baseline_per_chunk(chunks_rows)
    if not ub and not bl:
        return

    # Find the best SGD config (highest overall ratio)
    best_sgd_chunks = {}
    if sgd_rows:
        best_sgd = max(sgd_rows, key=lambda r: g(r, "ratio"))
        best_lr = g(best_sgd, "lr")
        best_mt = g(best_sgd, "mape_threshold")
        for r in chunks_rows:
            if (r.get("study") == "sgd"
                    and abs(g(r, "lr") - best_lr) < 1e-6
                    and abs(g(r, "mape_threshold") - best_mt) < 1e-6):
                idx = int(g(r, "chunk_idx"))
                best_sgd_chunks[idx] = g(r, "actual_ratio")

    all_indices = sorted(set(list(ub.keys()) + list(bl.keys()) + list(best_sgd_chunks.keys())))
    if not all_indices:
        return

    n_chunks = max(all_indices) + 1

    fig, ax = plt.subplots(figsize=(14, 6))
    _draw_pattern_bands(ax, n_chunks)

    x = np.array(all_indices, dtype=float)

    if ub:
        ub_vals = [ub.get(i, 0) for i in all_indices]
        ax.plot(x, ub_vals, color="#e74c3c", linewidth=2.0, alpha=0.9,
                label="Upper Bound (best-static)", marker="o", markersize=3, zorder=4)

    if bl:
        bl_vals = [bl.get(i, 0) for i in all_indices]
        ax.plot(x, bl_vals, color="#f39c12", linewidth=2.0, alpha=0.9,
                label="Baseline (NN inference-only)", marker="s", markersize=3, zorder=3)

    if best_sgd_chunks:
        sgd_vals = [best_sgd_chunks.get(i, 0) for i in all_indices]
        ax.plot(x, sgd_vals, color="#2c3e50", linewidth=2.0, alpha=0.9,
                label=f"Best SGD (lr={best_lr:.3f}, mt={best_mt:.2f})",
                marker="^", markersize=3, zorder=2)

    ax.set_xlabel("Chunk Index")
    ax.set_ylabel("Compression Ratio (log)")
    ax.set_title("Per-Chunk Compression Ratio Comparison", fontsize=13)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:.1f}x" if v < 10 else f"{v:.0f}x"))
    # Only label every Nth tick if too many chunks
    if len(all_indices) > 40:
        step = max(1, len(all_indices) // 25)
        ax.set_xticks(all_indices[::step])
    else:
        ax.set_xticks(all_indices)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y", which="both")

    fig.tight_layout()
    path = os.path.join(output_dir, "per_chunk_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Plot 6: Upper Bound All Configs Per Chunk -------------------------------

# One color per algorithm
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


def _decode_nn_action(action):
    """Decode NN action int -> (algo_name, shuffle, quant) using modular encoding."""
    action = int(action)
    algo_id = action % 8
    quant = (action // 8) % 2
    shuffle = (action // 16) % 2
    algo = ALGO_NAMES_BY_ID.get(algo_id, f"unk{algo_id}")
    tag = algo
    if shuffle:
        tag += "+s4"
    if quant:
        tag += "+q"
    return algo, tag


def plot_upper_bound_configs(ub_rows, chunks_rows, output_dir):
    """Side-by-side bars: upper bound best vs NN baseline pick per chunk."""
    if not ub_rows:
        return

    ok_rows = [r for r in ub_rows
               if r.get("status") == "ok" and g(r, "compression_ratio") > 0]
    if not ok_rows:
        return

    # Find best config per chunk from upper bound
    best_per_chunk = {}
    for r in ok_rows:
        ci = int(g(r, "chunk_idx"))
        ratio = g(r, "compression_ratio")
        if ratio > best_per_chunk.get(ci, (0, None, None))[0]:
            algo = r.get("algorithm", "?")
            shuf = int(g(r, "shuffle_bytes"))
            quant = int(g(r, "quantization"))
            tag = algo
            if shuf:
                tag += f"+s{shuf}"
            if quant:
                tag += "+q"
            best_per_chunk[ci] = (ratio, tag, algo)

    # Get NN baseline picks per chunk
    nn_per_chunk = {}  # chunk_idx -> (ratio, tag, algo)
    for r in chunks_rows:
        if r.get("study") == "baseline":
            ci = int(g(r, "chunk_idx"))
            action = int(g(r, "nn_action"))
            algo, tag = _decode_nn_action(action)
            nn_per_chunk[ci] = (g(r, "actual_ratio"), tag, algo)

    chunk_indices = sorted(best_per_chunk.keys())
    if not chunk_indices:
        return
    n_chunks = max(chunk_indices) + 1

    fig, ax = plt.subplots(figsize=(14, 6))
    _draw_pattern_bands(ax, n_chunks)

    bar_width = 0.38

    UB_COLOR = "#e74c3c"   # red
    NN_COLOR = "#2980b9"   # blue

    # Draw upper bound bars (left)
    for ci in chunk_indices:
        ratio, tag, algo = best_per_chunk[ci]
        ax.bar(ci - bar_width / 2, ratio, width=bar_width, color=UB_COLOR,
               edgecolor="white", linewidth=0.3, alpha=0.85, zorder=3)

    # Draw NN baseline bars (right)
    for ci in chunk_indices:
        if ci not in nn_per_chunk:
            continue
        ratio, tag, algo = nn_per_chunk[ci]
        ax.bar(ci + bar_width / 2, ratio, width=bar_width, color=NN_COLOR,
               edgecolor="white", linewidth=0.3, alpha=0.85, zorder=3)

    # Legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=UB_COLOR, alpha=0.85,
                       edgecolor="white", label="Upper Bound (best-static)"),
        plt.Rectangle((0, 0), 1, 1, fc=NN_COLOR, alpha=0.85,
                       edgecolor="white", label="NN Baseline (inference-only)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right",
              framealpha=0.9)

    # Range-collapsed labels for upper bound best
    runs = []
    for ci in chunk_indices:
        ratio, tag, _ = best_per_chunk[ci]
        if runs and runs[-1][2] == tag and ci == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], ci, tag, max(runs[-1][3], ratio))
        else:
            runs.append((ci, ci, tag, ratio))

    for start, end, tag, peak_ratio in runs:
        mid = (start + end) / 2.0
        if start == end:
            label = f"[{start}] {tag}"
        else:
            label = f"[{start}-{end}] {tag}"
        ax.annotate(label, (mid, peak_ratio), fontsize=5, fontweight="bold",
                    ha="center", va="bottom", rotation=45,
                    textcoords="offset points", xytext=(0, 4),
                    color="#222", zorder=4)

    # Range-collapsed labels for NN baseline (below bars)
    nn_runs = []
    for ci in chunk_indices:
        if ci not in nn_per_chunk:
            continue
        ratio, tag, _ = nn_per_chunk[ci]
        if nn_runs and nn_runs[-1][2] == tag and ci == nn_runs[-1][1] + 1:
            nn_runs[-1] = (nn_runs[-1][0], ci, tag, min(nn_runs[-1][3], ratio))
        else:
            nn_runs.append((ci, ci, tag, ratio))

    for start, end, tag, min_ratio in nn_runs:
        mid = (start + end) / 2.0
        if start == end:
            label = f"[{start}] {tag}"
        else:
            label = f"[{start}-{end}] {tag}"
        ax.annotate(label, (mid, min_ratio), fontsize=5, fontstyle="italic",
                    ha="center", va="top", rotation=-45,
                    textcoords="offset points", xytext=(0, -4),
                    color="#555", zorder=4)

    ax.set_xlabel("Chunk Index")
    ax.set_ylabel("Compression Ratio (log)")
    ax.set_title("Upper Bound vs NN Baseline: Best Config Per Chunk", fontsize=13)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:.1f}x" if v < 10 else f"{v:.0f}x"))

    if len(chunk_indices) > 40:
        step = max(1, len(chunk_indices) // 25)
        ax.set_xticks(chunk_indices[::step])
    else:
        ax.set_xticks(chunk_indices)

    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(output_dir, "upper_bound_configs.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Plot 7: Best Configs Bar Chart ------------------------------------------

# -- Plot 7: Config Cross-Check (NN vs Upper Bound) -------------------------

def plot_config_crosscheck(ub_rows, chunks_rows, sgd_rows, expl_rows, output_dir):
    """Four-row heatmap: Upper Bound vs Best Exploration vs Best SGD vs NN Baseline
    per chunk, colored by algorithm. Mismatched chunks marked."""
    if not ub_rows or not chunks_rows:
        return

    ok_rows = [r for r in ub_rows
               if r.get("status") == "ok" and g(r, "compression_ratio") > 0]
    if not ok_rows:
        return

    # Best upper bound config per chunk
    ub_per_chunk = {}
    for r in ok_rows:
        ci = int(g(r, "chunk_idx"))
        ratio = g(r, "compression_ratio")
        if ratio > ub_per_chunk.get(ci, (0, None, None))[0]:
            algo = r.get("algorithm", "?")
            shuf = int(g(r, "shuffle_bytes"))
            quant = int(g(r, "quantization"))
            tag = algo
            if shuf:
                tag += "+s4"
            if quant:
                tag += "+q"
            ub_per_chunk[ci] = (ratio, tag, algo)

    # NN baseline config per chunk
    nn_per_chunk = {}
    for r in chunks_rows:
        if r.get("study") == "baseline":
            ci = int(g(r, "chunk_idx"))
            action = int(g(r, "nn_action"))
            algo, tag = _decode_nn_action(action)
            nn_per_chunk[ci] = (g(r, "actual_ratio"), tag, algo)

    # Best SGD config per chunk
    sgd_per_chunk = {}
    sgd_label = "Best SGD"
    if sgd_rows:
        best_sgd = max(sgd_rows, key=lambda r: g(r, "ratio"))
        best_lr = g(best_sgd, "lr")
        best_mt = g(best_sgd, "mape_threshold")
        sgd_label = f"Best SGD\n(lr={best_lr:.3f}, mt={best_mt:.2f})"
        for r in chunks_rows:
            if (r.get("study") == "sgd"
                    and abs(g(r, "lr") - best_lr) < 1e-6
                    and abs(g(r, "mape_threshold") - best_mt) < 1e-6):
                ci = int(g(r, "chunk_idx"))
                action = int(g(r, "nn_action"))
                algo, tag = _decode_nn_action(action)
                sgd_per_chunk[ci] = (g(r, "actual_ratio"), tag, algo)

    # Best Exploration config per chunk
    expl_per_chunk = {}
    expl_label = "Best Exploration"
    if expl_rows:
        best_expl = max(expl_rows, key=lambda r: g(r, "ratio"))
        best_et = g(best_expl, "expl_threshold")
        best_k = int(g(best_expl, "K"))
        expl_label = f"Best Expl\n(t={best_et:.2f}, K={best_k})"
        for r in chunks_rows:
            if (r.get("study") == "exploration"
                    and abs(g(r, "expl_threshold") - best_et) < 1e-6
                    and int(g(r, "K")) == best_k):
                ci = int(g(r, "chunk_idx"))
                action = int(g(r, "nn_action"))
                algo, tag = _decode_nn_action(action)
                expl_per_chunk[ci] = (g(r, "actual_ratio"), tag, algo)

    chunk_indices = sorted(ub_per_chunk.keys())
    if not chunk_indices:
        return
    n_chunks = len(chunk_indices)

    has_sgd = bool(sgd_per_chunk)
    has_expl = bool(expl_per_chunk)

    # Build rows bottom-to-top: NN Baseline, Best SGD, Best Expl, Upper Bound
    rows = []  # list of (label, per_chunk_dict)
    rows.append(("NN Baseline", nn_per_chunk))
    if has_sgd:
        rows.append((sgd_label, sgd_per_chunk))
    if has_expl:
        rows.append((expl_label, expl_per_chunk))
    rows.append(("Upper Bound", ub_per_chunk))

    n_rows = len(rows)
    row_ub = n_rows - 1

    fig_h = 2.5 + n_rows * 0.9
    fig, ax = plt.subplots(figsize=(max(14, n_chunks * 0.14), fig_h))

    # Draw each row
    for row_y, (label, per_chunk) in enumerate(rows):
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

    # Mark mismatches vs upper bound for each non-UB row
    for row_y, (label, per_chunk) in enumerate(rows):
        if row_y == row_ub:
            continue
        for xi, ci in enumerate(chunk_indices):
            ub_tag = ub_per_chunk.get(ci, (0, "", ""))[1]
            row_tag = per_chunk.get(ci, (0, "", ""))[1]
            if ub_tag and row_tag and ub_tag != row_tag:
                ax.plot(xi + 0.5, row_y - 0.5, marker="x", color="#e74c3c",
                        markersize=4, markeredgewidth=1.5, zorder=5)

    # Pattern region separators
    slab = n_chunks // 5 if n_chunks >= 5 else n_chunks
    for p in range(5):
        x0 = p * slab
        ax.axvline(x=x0, color="#888", linewidth=0.8, linestyle="--", alpha=0.5)
        mid = x0 + slab / 2
        ax.text(mid, row_ub + 0.7, PATTERN_NAMES[p].replace("_", " "),
                ha="center", va="bottom", fontsize=6,
                color=PATTERN_COLORS.get(PATTERN_NAMES[p], "#888"),
                fontweight="bold")

    ax.set_yticks(list(range(n_rows)))
    ax.set_yticklabels([label for label, _ in rows], fontsize=7)
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

    ax.set_title("Config Cross-Check: Per-Chunk Algorithm Selection\n"
                 "(x = differs from Upper Bound)", fontsize=12)

    # Algorithm color legend
    all_algos = list(ALGO_COLORS.keys())
    legend_handles = []
    for algo in all_algos:
        color = ALGO_COLORS.get(algo, "#ccc")
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=color,
                                             edgecolor="white", label=algo))
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
                   if ub_per_chunk.get(ci, (0, "", ""))[1] ==
                      nn_per_chunk.get(ci, (0, "", ""))[1])
    print(f"    NN Baseline vs UB match: {nn_match}/{n_total} "
          f"({100*nn_match/n_total:.1f}%)")
    if has_sgd:
        sgd_match = sum(1 for ci in chunk_indices
                        if ub_per_chunk.get(ci, (0, "", ""))[1] ==
                           sgd_per_chunk.get(ci, (0, "", ""))[1])
        print(f"    Best SGD vs UB match:    {sgd_match}/{n_total} "
              f"({100*sgd_match/n_total:.1f}%)")
    if has_expl:
        expl_match = sum(1 for ci in chunk_indices
                         if ub_per_chunk.get(ci, (0, "", ""))[1] ==
                            expl_per_chunk.get(ci, (0, "", ""))[1])
        print(f"    Best Expl vs UB match:   {expl_match}/{n_total} "
              f"({100*expl_match/n_total:.1f}%)")


# -- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NN Adaptiveness Benchmark Visualizer")
    parser.add_argument("--sgd-csv", default=None)
    parser.add_argument("--expl-csv", default=None)
    parser.add_argument("--chunks-csv", default=None)
    parser.add_argument("--ub-csv", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_DIR)
    args = parser.parse_args()

    sgd_csv = args.sgd_csv or os.path.join(DEFAULT_DIR, "sgd_study.csv")
    expl_csv = args.expl_csv or os.path.join(DEFAULT_DIR, "exploration_study.csv")
    chunks_csv = args.chunks_csv or os.path.join(DEFAULT_DIR, "chunks_detail.csv")
    ub_csv = args.ub_csv or os.path.join(DEFAULT_DIR, "upper_bound.csv")
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    sgd_rows, expl_rows, chunks_rows, ub_rows = [], [], [], []

    if os.path.exists(sgd_csv):
        sgd_rows = parse_csv(sgd_csv)
        print(f"Loaded {len(sgd_rows)} SGD rows from {sgd_csv}")
    else:
        print(f"WARNING: SGD CSV not found: {sgd_csv}")

    if os.path.exists(expl_csv):
        expl_rows = parse_csv(expl_csv)
        print(f"Loaded {len(expl_rows)} exploration rows from {expl_csv}")
    else:
        print(f"WARNING: Exploration CSV not found: {expl_csv}")

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

    if not sgd_rows and not expl_rows and not chunks_rows and not ub_rows:
        print("ERROR: No data to visualize. Run the benchmark first.")
        sys.exit(1)

    print("\nGenerating plots...")

    if sgd_rows:
        plot_sgd_heatmap(sgd_rows, output_dir)
        if chunks_rows:
            plot_sgd_convergence(chunks_rows, sgd_rows, output_dir)

    if expl_rows:
        plot_exploration_heatmap(expl_rows, output_dir)
        plot_exploration_scatter(expl_rows, ub_rows, output_dir)

    if chunks_rows or ub_rows:
        plot_per_chunk_comparison(chunks_rows, sgd_rows, ub_rows, output_dir)

    if ub_rows:
        plot_upper_bound_configs(ub_rows, chunks_rows, output_dir)

    if ub_rows and chunks_rows:
        plot_config_crosscheck(ub_rows, chunks_rows, sgd_rows, expl_rows, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
