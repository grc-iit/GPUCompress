#!/usr/bin/env python3
"""
NN Adaptiveness Benchmark Visualizer.

Reads CSV results from the NN adaptiveness benchmark and produces:
  1. SGD heatmap: LR vs MAPE threshold, color = final MAPE
  2. SGD convergence: Cumulative MAPE vs chunk index with pattern-region bands
  3. Per-chunk ratio comparison: upper bound vs baseline vs best SGD
  4. Best configs bar chart: upper bound vs baseline vs best SGD

Usage:
  python3 benchmarks/nn_adaptiveness/visualize_nn_adaptiveness.py [options]

  --sgd-csv PATH     SGD study CSV (default: auto-detect)
  --chunks-csv PATH  Chunks detail CSV (default: auto-detect)
  --output-dir DIR   Directory for output PNGs (default: benchmarks/nn_adaptiveness/)
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

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(_SCRIPT_DIR, "results")


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
    fig.suptitle("SGD Study: Final MAPE (%) by LR x MAPE Threshold", fontsize=13,
                 fontweight="bold")
    fig.text(0.5, 0.95,
             "Hyperparameter grid search on 5 synthetic data patterns. Each cell shows the final\n"
             "prediction error after SGD converges. Lower MAPE = better prediction accuracy.",
             ha="center", fontsize=8.5, color="#555", va="top", style="italic")

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
    """One PNG per learning rate, each with one subplot per MAPE threshold.
    Shows per-chunk MAPE with pattern-region bands."""
    sgd_chunks = [r for r in chunks_rows if r.get("study") == "sgd"]
    if not sgd_chunks:
        return

    lrs = sorted(set(g(r, "lr") for r in sgd_rows))
    mapes = sorted(set(g(r, "mape_threshold") for r in sgd_rows))
    if not lrs or not mapes:
        return

    max_chunk = max(int(g(r, "chunk_idx")) for r in sgd_chunks)
    n_chunks = max_chunk + 1

    for li, lr in enumerate(lrs):
        n_mt = len(mapes)
        fig, axes = plt.subplots(n_mt, 1, figsize=(14, 4 * n_mt), sharex=True)
        if n_mt == 1:
            axes = [axes]
        fig.suptitle(f"SGD Convergence — LR = {lr:.3f}", fontsize=14,
                     fontweight="bold", y=1.03)
        fig.text(0.5, 1.01,
                 "Per-chunk prediction error as SGD processes chunks sequentially across 5 data patterns.\n"
                 "Colored bands show pattern boundaries. Vertical line marks convergence point.",
                 ha="center", fontsize=8.5, color="#555", va="top", style="italic")

        for mi, mt in enumerate(mapes):
            ax = axes[mi]
            _draw_pattern_bands(ax, n_chunks)

            cr = [r for r in sgd_chunks
                  if abs(g(r, "lr") - lr) < 1e-6
                  and abs(g(r, "mape_threshold") - mt) < 1e-6]
            cr.sort(key=lambda r: g(r, "chunk_idx"))
            if not cr:
                ax.set_title(f"MT = {mt:.2f}  (no data)", fontsize=11)
                continue

            indices, running = _compute_per_chunk_mape(cr)
            if indices:
                ax.plot(indices, running,
                        color=MAPE_COLORS[mi % len(MAPE_COLORS)],
                        linewidth=1.5, alpha=0.9)
                # Convergence threshold line
                ax.axhline(y=10, color="gray", linestyle="--",
                           alpha=0.4, linewidth=0.8)
                ax.text(n_chunks * 0.98, 10, "10%", fontsize=7,
                        ha="right", va="bottom", color="gray", alpha=0.6)

            # Find convergence chunk from sgd_rows
            matching = [r for r in sgd_rows
                        if abs(g(r, "lr") - lr) < 1e-6
                        and abs(g(r, "mape_threshold") - mt) < 1e-6]
            conv_chunk = int(g(matching[0], "convergence_chunks")) if matching else 0
            final_mape = g(matching[0], "final_mape") if matching else 0
            ratio = g(matching[0], "ratio") if matching else 0

            title = f"MT = {mt:.2f}  |  ratio={ratio:.2f}x  mape={final_mape:.1f}%"
            if conv_chunk > 0:
                title += f"  conv={conv_chunk}"
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


# -- Plot 3: Per-Chunk Ratio Comparison --------------------------------------

def plot_per_chunk_comparison(chunks_rows, sgd_rows, ub_rows, output_dir):
    """Per-chunk ratio comparison: one PNG per pattern + one combined.
    Each shows upper bound vs baseline vs best SGD as line plots."""
    ub = _get_upper_bound_per_chunk(ub_rows)
    bl = _get_baseline_per_chunk(chunks_rows)
    if not ub and not bl:
        return

    # Find the best SGD config (highest overall ratio)
    best_sgd_chunks = {}
    sgd_label = ""
    if sgd_rows:
        best_sgd = max(sgd_rows, key=lambda r: g(r, "ratio"))
        best_lr = g(best_sgd, "lr")
        best_mt = g(best_sgd, "mape_threshold")
        sgd_label = f"lr={best_lr:.3f}, mt={best_mt:.2f}"
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

    # Build pattern mapping: chunk_idx -> pattern_name
    chunk_patterns = {}
    for r in chunks_rows:
        if r.get("study") == "baseline":
            ci = int(g(r, "chunk_idx"))
            chunk_patterns[ci] = r.get("pattern", "unknown")

    # Group chunks by pattern (preserving order)
    pattern_chunks = {}  # pattern -> [chunk_indices]
    seen_patterns = []
    for ci in all_indices:
        pat = chunk_patterns.get(ci, "unknown")
        if pat not in pattern_chunks:
            pattern_chunks[pat] = []
            seen_patterns.append(pat)
        pattern_chunks[pat].append(ci)

    # --- Combined plot (all patterns) ---
    fig, ax = plt.subplots(figsize=(14, 6))
    _draw_pattern_bands(ax, n_chunks)
    x = np.array(all_indices, dtype=float)

    if ub:
        ax.plot(x, [ub.get(i, 0) for i in all_indices],
                color="#e74c3c", linewidth=2.0, alpha=0.9,
                label="Upper Bound (best-static)", marker="o", markersize=3, zorder=4)
    if bl:
        ax.plot(x, [bl.get(i, 0) for i in all_indices],
                color="#f39c12", linewidth=2.0, alpha=0.9,
                label="Baseline (NN inference-only)", marker="s", markersize=3, zorder=3)
    if best_sgd_chunks:
        ax.plot(x, [best_sgd_chunks.get(i, 0) for i in all_indices],
                color="#2c3e50", linewidth=2.0, alpha=0.9,
                label=f"Best SGD ({sgd_label})", marker="^", markersize=3, zorder=2)

    ax.set_xlabel("Chunk Index")
    ax.set_ylabel("Compression Ratio (log)")
    fig.suptitle("Per-Chunk Compression Ratio Comparison (All Patterns)", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.text(0.5, 0.99,
             "Compression ratio per chunk across all 5 synthetic patterns. Compares upper bound\n"
             "(exhaustive search), NN baseline (inference-only), and best SGD (online learning).",
             ha="center", fontsize=8.5, color="#555", va="top", style="italic")
    ax.set_title("")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:.1f}x" if v < 10 else f"{v:.0f}x"))
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

    # --- Build per-chunk config info for baseline and best SGD ---
    bl_configs = {}   # chunk_idx -> (ratio, config_tag)
    sgd_configs = {}
    for r in chunks_rows:
        ci = int(g(r, "chunk_idx"))
        _, tag = _parse_nn_action(r.get("nn_action", ""))
        if r.get("study") == "baseline":
            bl_configs[ci] = (g(r, "actual_ratio"), tag)
        elif (r.get("study") == "sgd" and sgd_label
              and abs(g(r, "lr") - best_lr) < 1e-6
              and abs(g(r, "mape_threshold") - best_mt) < 1e-6):
            sgd_configs[ci] = (g(r, "actual_ratio"), tag)

    # Build per-chunk config for upper bound
    ub_configs = {}
    for r in (ub_rows or []):
        if r.get("status") != "ok" or g(r, "compression_ratio") <= 0:
            continue
        ci = int(g(r, "chunk_idx"))
        ratio = g(r, "compression_ratio")
        if ratio > ub_configs.get(ci, (0, ""))[0]:
            algo = r.get("algorithm", "?")
            shuf = int(g(r, "shuffle_bytes"))
            quant = int(g(r, "quantization"))
            tag = algo
            if shuf:
                tag += f"+s{shuf}"
            if quant:
                tag += "+q"
            ub_configs[ci] = (ratio, tag)

    # Collect all unique config tags for color mapping
    all_tags = set()
    for ci in all_indices:
        if ci in ub_configs:
            all_tags.add(ub_configs[ci][1])
        if ci in bl_configs:
            all_tags.add(bl_configs[ci][1])
        if ci in sgd_configs:
            all_tags.add(sgd_configs[ci][1])
    tag_list = sorted(all_tags)
    config_cmap = plt.cm.tab10
    tag_colors = {t: config_cmap(i % 10) for i, t in enumerate(tag_list)}

    # --- Per-pattern plots ---
    for pat in seen_patterns:
        indices = pattern_chunks[pat]
        if not indices:
            continue

        local_x = list(range(len(indices)))

        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        series = [
            ("Upper Bound (best-static)", ub_configs, "#e74c3c", "o"),
            ("Baseline (NN inference-only)", bl_configs, "#2980b9", "s"),
            (f"Best SGD ({sgd_label})" if sgd_label else "Best SGD", sgd_configs, "#27ae60", "^"),
        ]

        for ax, (label, configs, line_color, marker) in zip(axes, series):
            ratios = [configs.get(ci, (0, ""))[0] for ci in indices]
            tags   = [configs.get(ci, (0, "unknown"))[1] for ci in indices]

            if not any(r > 0 for r in ratios):
                ax.text(0.5, 0.5, f"{label}: no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=12, color="gray")
                ax.set_ylabel("Ratio")
                ax.set_title(label, fontsize=11)
                continue

            # Plot line
            ax.plot(local_x, ratios, color=line_color, linewidth=1.0,
                    alpha=0.4, zorder=2)

            # Plot colored markers by config
            for i, (lx, ratio, tag) in enumerate(zip(local_x, ratios, tags)):
                if ratio > 0:
                    c = tag_colors.get(tag, "gray")
                    ax.scatter(lx, ratio, color=c, marker=marker, s=40,
                               edgecolors="white", linewidths=0.3, zorder=3)

            # Config-change annotations: label runs of same config
            runs = []
            for i, tag in enumerate(tags):
                if runs and runs[-1][2] == tag:
                    runs[-1] = (runs[-1][0], i, tag)
                else:
                    runs.append((i, i, tag))

            for start, end, tag in runs:
                if tag == "" or tag == "unknown":
                    continue
                mid = (start + end) / 2.0
                peak = max(ratios[start:end+1]) if ratios[start:end+1] else 0
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

            # Use log scale if range is large
            valid = [r for r in ratios if r > 0]
            if valid and max(valid) / max(min(valid), 0.01) > 5:
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(
                    lambda v, _: f"{v:.1f}x" if v < 10 else f"{v:.0f}x"))

        axes[-1].set_xlabel(f"Chunk (within {pat})")

        # Config color legend
        used_tags = set()
        for ci in indices:
            for configs in [ub_configs, bl_configs, sgd_configs]:
                if ci in configs:
                    used_tags.add(configs[ci][1])
        legend_handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=tag_colors.get(t, "gray"),
                          markersize=8, label=t)
                          for t in sorted(used_tags) if t]
        if legend_handles:
            fig.legend(handles=legend_handles, loc="upper right",
                       fontsize=7, ncol=min(len(legend_handles), 4),
                       title="Config", title_fontsize=8,
                       bbox_to_anchor=(0.99, 0.99))

        fig.suptitle(f"Per-Chunk Config & Ratio: {pat}", fontsize=14,
                     fontweight="bold", y=1.03)
        fig.text(0.5, 1.01,
                 "Algorithm selection and ratio per chunk. Markers colored by config.\n"
                 "Shows which algorithm each method picks and the resulting compression ratio.",
                 ha="center", fontsize=8.5, color="#555", va="top", style="italic")
        fig.tight_layout()

        safe_name = pat.replace(" ", "_").lower()
        path = os.path.join(output_dir, f"per_chunk_{safe_name}.png")
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


def _parse_nn_action(action_str):
    """Parse nn_action string from CSV -> (algo_name, tag_string).

    CSV now stores human-readable strings like 'lz4', 'snappy+shuf4',
    'zstd+shuf4+quant'.
    """
    s = str(action_str).strip()
    algo = s.split("+")[0] if "+" in s else s
    return algo, s


def plot_upper_bound_configs(ub_rows, chunks_rows, sgd_rows, output_dir):
    """Side-by-side bars: upper bound best vs NN baseline vs best SGD per chunk."""
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
            algo, tag = _parse_nn_action(r.get("nn_action", ""))
            nn_per_chunk[ci] = (g(r, "actual_ratio"), tag, algo)

    # Get best SGD run's picks per chunk
    sgd_per_chunk = {}
    sgd_label = ""
    if sgd_rows:
        best_sgd = max(sgd_rows, key=lambda r: g(r, "ratio"))
        best_lr = g(best_sgd, "lr")
        best_mt = g(best_sgd, "mape_threshold")
        sgd_label = f"lr={best_lr:.2f} mt={best_mt:.2f}"
        for r in chunks_rows:
            if (r.get("study") == "sgd"
                    and abs(g(r, "lr") - best_lr) < 1e-6
                    and abs(g(r, "mape_threshold") - best_mt) < 1e-6):
                ci = int(g(r, "chunk_idx"))
                algo, tag = _parse_nn_action(r.get("nn_action", ""))
                sgd_per_chunk[ci] = (g(r, "actual_ratio"), tag, algo)

    chunk_indices = sorted(best_per_chunk.keys())
    if not chunk_indices:
        return
    n_chunks = max(chunk_indices) + 1

    has_sgd = len(sgd_per_chunk) > 0
    n_series = 3 if has_sgd else 2
    bar_width = 0.8 / n_series

    fig, ax = plt.subplots(figsize=(16, 7))
    _draw_pattern_bands(ax, n_chunks)

    UB_COLOR  = "#e74c3c"   # red
    NN_COLOR  = "#2980b9"   # blue
    SGD_COLOR = "#27ae60"   # green

    # Draw upper bound bars
    for ci in chunk_indices:
        ratio, tag, algo = best_per_chunk[ci]
        offset = -bar_width if has_sgd else -bar_width / 2
        ax.bar(ci + offset, ratio, width=bar_width, color=UB_COLOR,
               edgecolor="white", linewidth=0.3, alpha=0.85, zorder=3)

    # Draw NN baseline bars
    for ci in chunk_indices:
        if ci not in nn_per_chunk:
            continue
        ratio, tag, algo = nn_per_chunk[ci]
        offset = 0 if has_sgd else bar_width / 2
        ax.bar(ci + offset, ratio, width=bar_width, color=NN_COLOR,
               edgecolor="white", linewidth=0.3, alpha=0.85, zorder=3)

    # Draw best SGD bars
    if has_sgd:
        for ci in chunk_indices:
            if ci not in sgd_per_chunk:
                continue
            ratio, tag, algo = sgd_per_chunk[ci]
            ax.bar(ci + bar_width, ratio, width=bar_width, color=SGD_COLOR,
                   edgecolor="white", linewidth=0.3, alpha=0.85, zorder=3)

    # Legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=UB_COLOR, alpha=0.85,
                       edgecolor="white", label="Upper Bound (best-static)"),
        plt.Rectangle((0, 0), 1, 1, fc=NN_COLOR, alpha=0.85,
                       edgecolor="white", label="NN Baseline (inference-only)"),
    ]
    if has_sgd:
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc=SGD_COLOR, alpha=0.85,
                           edgecolor="white",
                           label=f"Best SGD ({sgd_label})"))
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
    title = "Upper Bound vs NN Baseline"
    if has_sgd:
        title += f" vs Best SGD ({sgd_label})"
    title += ": Per Chunk"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.text(0.5, 0.99,
             "Side-by-side ratio per chunk with exhaustive annotations. Pattern bands show data transitions.\n"
             "Compares how close NN baseline and SGD approach the exhaustive-search upper bound.",
             ha="center", fontsize=8.5, color="#555", va="top", style="italic")
    ax.set_title("")
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

def plot_config_crosscheck(ub_rows, chunks_rows, sgd_rows, output_dir):
    """Multi-row heatmap: Upper Bound vs Best SGD vs NN Baseline
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
            algo, tag = _parse_nn_action(r.get("nn_action", ""))
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
                algo, tag = _parse_nn_action(r.get("nn_action", ""))
                sgd_per_chunk[ci] = (g(r, "actual_ratio"), tag, algo)

    chunk_indices = sorted(ub_per_chunk.keys())
    if not chunk_indices:
        return
    n_chunks = len(chunk_indices)

    has_sgd = bool(sgd_per_chunk)

    # Build rows bottom-to-top: NN Baseline, Best SGD, Upper Bound
    rows = []  # list of (label, per_chunk_dict)
    rows.append(("NN Baseline", nn_per_chunk))
    if has_sgd:
        rows.append((sgd_label, sgd_per_chunk))
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

    fig.suptitle("Config Cross-Check: Per-Chunk Algorithm Selection",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.text(0.5, 1.01,
             "Each row shows the algorithm chosen per chunk. Red 'x' marks disagreements with upper bound.\n"
             "Fewer mismatches = better NN prediction. Pattern regions separated by dashed lines.",
             ha="center", fontsize=8.5, color="#555", va="top", style="italic")
    ax.set_title("")

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


# -- Plot 8: Summary Bar Chart -----------------------------------------------

def plot_summary(sgd_rows, ub_rows, chunks_rows, output_dir):
    """Aggregate bar chart: Upper Bound vs Baseline vs Best SGD."""
    ub_ratio = _get_upper_bound_ratio(ub_rows)

    # Baseline: harmonic mean of actual_ratio from baseline chunks
    bl_ratios = [g(r, "actual_ratio") for r in chunks_rows
                 if r.get("study") == "baseline" and g(r, "actual_ratio") > 0]
    if bl_ratios:
        bl_ratio = len(bl_ratios) / sum(1.0 / v for v in bl_ratios)
    else:
        bl_ratio = 0.0

    # Best SGD ratio
    sgd_ratio = max((g(r, "ratio") for r in sgd_rows), default=0.0) if sgd_rows else 0.0

    labels, values, colors = [], [], []
    if ub_ratio > 0:
        labels.append("Upper Bound")
        values.append(ub_ratio)
        colors.append("#e74c3c")
    if bl_ratio > 0:
        labels.append("Baseline")
        values.append(bl_ratio)
        colors.append("#2980b9")
    if sgd_ratio > 0:
        labels.append("Best SGD")
        values.append(sgd_ratio)
        colors.append("#27ae60")

    if not values:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("NN Adaptiveness: Aggregate Compression Ratio", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.text(0.5, 0.99,
             "Harmonic mean compression ratio across all chunks and patterns.\n"
             "Compares upper bound, NN baseline, and best SGD configs.",
             ha="center", fontsize=8.5, color="#555", va="top", style="italic")

    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="white",
                  linewidth=0.5, alpha=0.85, zorder=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Compression Ratio")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2f}x", ha="center", va="bottom", fontsize=11,
                fontweight="bold")

    fig.tight_layout()
    path = os.path.join(output_dir, "summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Plot 9: Per-Chunk Prediction Error (Baseline vs Best SGD) ---------------

def plot_fig1_avg_mape(chunks_rows, sgd_rows, output_dir):
    """Per-chunk prediction error: baseline vs best SGD, with pattern bands."""
    bl_chunks = [r for r in chunks_rows if r.get("study") == "baseline"]
    if not bl_chunks:
        return

    max_chunk = max(int(g(r, "chunk_idx")) for r in chunks_rows)
    n_chunks = max_chunk + 1

    # Baseline MAPE per chunk
    bl_idx, bl_mape = _compute_per_chunk_mape(bl_chunks)

    # Best SGD MAPE per chunk
    sgd_idx, sgd_mape = [], []
    sgd_label = ""
    if sgd_rows:
        best_sgd = max(sgd_rows, key=lambda r: g(r, "ratio"))
        best_lr = g(best_sgd, "lr")
        best_mt = g(best_sgd, "mape_threshold")
        sgd_label = f"lr={best_lr:.3f}, mt={best_mt:.2f}"
        sgd_chunks = [r for r in chunks_rows
                      if r.get("study") == "sgd"
                      and abs(g(r, "lr") - best_lr) < 1e-6
                      and abs(g(r, "mape_threshold") - best_mt) < 1e-6]
        sgd_chunks.sort(key=lambda r: g(r, "chunk_idx"))
        sgd_idx, sgd_mape = _compute_per_chunk_mape(sgd_chunks)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Per-Chunk Prediction Error: Baseline vs Best SGD", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.text(0.5, 0.99,
             "MAPE (%) per chunk on log scale. Lower = better prediction accuracy.\n"
             "Running average shows SGD convergence trend across pattern transitions.",
             ha="center", fontsize=8.5, color="#555", va="top", style="italic")

    _draw_pattern_bands(ax, n_chunks)

    # Exhaustive reference line
    ax.axhline(y=0.1, color="gray", linestyle="--", alpha=0.5, linewidth=1.0,
               label="Exhaustive (~0% error)")

    # Baseline
    if bl_idx:
        ax.plot(bl_idx, bl_mape, color="#2980b9", linewidth=1.2, alpha=0.8,
                label="Baseline", marker=".", markersize=2, zorder=3)

    # Best SGD
    if sgd_idx:
        ax.plot(sgd_idx, sgd_mape, color="#27ae60", linewidth=1.2, alpha=0.8,
                label=f"Best SGD ({sgd_label})", marker=".", markersize=2, zorder=3)
        # Running average
        if len(sgd_mape) >= 5:
            window = max(5, len(sgd_mape) // 20)
            running_avg = np.convolve(sgd_mape, np.ones(window) / window, mode="valid")
            offset = window // 2
            ax.plot(sgd_idx[offset:offset + len(running_avg)], running_avg,
                    color="#2c3e50", linewidth=2.0, alpha=0.7, linestyle="--",
                    label=f"SGD Running Avg (w={window})", zorder=4)

    ax.set_xlabel("Chunk Index")
    ax.set_ylabel("MAPE (%)")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.01)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_log_percent_formatter))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    path = os.path.join(output_dir, "fig1_avg_mape.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Plot 10: Chunk MAPE Comparison (Side-by-Side Bars) ----------------------

def plot_chunk_mape_comparison(chunks_rows, sgd_rows, output_dir):
    """MAPE heatmap: config (y-axis) vs chunk (x-axis), like VPIC style."""
    all_chunks = sorted(set(int(g(r, "chunk_idx")) for r in chunks_rows))
    if not all_chunks:
        return
    n_chunks = max(all_chunks) + 1

    # Gather unique configs: baseline + each SGD (lr, mt) combo
    configs = []  # list of (label, filter_func)
    configs.append(("Baseline (no SGD)",
                    lambda r: r.get("study") == "baseline"))

    # Collect unique SGD configs
    sgd_configs = set()
    for r in chunks_rows:
        if r.get("study") == "sgd":
            lr = g(r, "lr")
            mt = g(r, "mape_threshold")
            if lr > 0:
                sgd_configs.add((lr, mt))

    for lr, mt in sorted(sgd_configs):
        label = f"SGD lr={lr:.3f} mt={mt:.2f}"
        configs.append((label,
                        lambda r, _lr=lr, _mt=mt: (
                            r.get("study") == "sgd"
                            and abs(g(r, "lr") - _lr) < 1e-6
                            and abs(g(r, "mape_threshold") - _mt) < 1e-6)))

    if not configs:
        return

    # Build 2D grid: rows = configs, cols = chunks
    ch_idx = {c: i for i, c in enumerate(all_chunks)}
    grid = np.full((len(configs), len(all_chunks)), np.nan)

    for r in chunks_rows:
        pred = g(r, "predicted_ratio")
        actual = g(r, "actual_ratio")
        if not (pred > 0 and actual > 0):
            continue
        ci = int(g(r, "chunk_idx"))
        if ci not in ch_idx:
            continue
        mape = abs(pred - actual) / actual * 100.0
        for row_i, (_, filt) in enumerate(configs):
            if filt(r):
                grid[row_i, ch_idx[ci]] = mape
                break

    if np.all(np.isnan(grid)):
        return

    labels = [lbl for lbl, _ in configs]

    vmax = min(np.nanpercentile(grid, 95), 5000)
    vmax = max(vmax, 2)
    norm = mcolors.LogNorm(vmin=1, vmax=vmax, clip=True)

    fig, ax = plt.subplots(figsize=(max(12, len(all_chunks) * 0.25),
                                    max(4, len(configs) * 0.6 + 2)))

    # Draw pattern bands behind the heatmap
    _draw_pattern_bands(ax, n_chunks)

    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn_r", norm=norm,
                   interpolation="nearest", zorder=2)

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
             "Green = low error, Red = high error. Log scale. Pattern regions shown as background bands.",
             ha="center", fontsize=9, color="#555", va="top", style="italic")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, "chunk_mape_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Plot 11: Compression Ratio Comparison Bars ------------------------------

def plot_fig4_compression_ratio(sgd_rows, ub_rows, chunks_rows, output_dir):
    """Aggregate compression ratio comparison: Upper Bound vs Baseline vs Best SGD."""
    ub_ratio = _get_upper_bound_ratio(ub_rows)

    # Baseline: harmonic mean
    bl_ratios = [g(r, "actual_ratio") for r in chunks_rows
                 if r.get("study") == "baseline" and g(r, "actual_ratio") > 0]
    if bl_ratios:
        bl_ratio = len(bl_ratios) / sum(1.0 / v for v in bl_ratios)
    else:
        bl_ratio = 0.0

    # Best SGD ratio
    sgd_ratio = max((g(r, "ratio") for r in sgd_rows), default=0.0) if sgd_rows else 0.0

    labels, values, colors = [], [], []
    if ub_ratio > 0:
        labels.append("Upper Bound")
        values.append(ub_ratio)
        colors.append("#e74c3c")
    if bl_ratio > 0:
        labels.append("Baseline")
        values.append(bl_ratio)
        colors.append("#2980b9")
    if sgd_ratio > 0:
        labels.append("Best SGD")
        values.append(sgd_ratio)
        colors.append("#27ae60")

    if not values:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Compression Ratio: Upper Bound vs Baseline vs Best SGD", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.text(0.5, 0.99,
             "Aggregate compression ratio (harmonic mean across chunks).\n"
             "Shows how close each method gets to the exhaustive-search upper bound.",
             ha="center", fontsize=8.5, color="#555", va="top", style="italic")

    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="white",
                  linewidth=0.5, alpha=0.85, zorder=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Compression Ratio")
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate with ratio values and % of upper bound
    for i, (bar, val) in enumerate(zip(bars, values)):
        text = f"{val:.2f}x"
        if ub_ratio > 0 and labels[i] != "Upper Bound":
            pct = val / ub_ratio * 100
            text += f"\n({pct:.1f}% of UB)"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                text, ha="center", va="bottom", fontsize=10,
                fontweight="bold")

    fig.tight_layout()
    path = os.path.join(output_dir, "fig4_compression_ratio.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Plot 12: Per-Chunk Config (3-Panel) -------------------------------------

def plot_per_chunk_config(chunks_rows, sgd_rows, ub_rows, output_dir):
    """3-panel plot showing all chunks with markers colored by algorithm config.
    Panel 1: Upper Bound, Panel 2: Baseline, Panel 3: Best SGD."""
    ub_best = _get_upper_bound_per_chunk(ub_rows)
    bl = _get_baseline_per_chunk(chunks_rows)
    if not ub_best and not bl:
        return

    # Build per-chunk config dicts: chunk_idx -> (ratio, tag, algo)
    ub_configs = {}
    for r in (ub_rows or []):
        if r.get("status") != "ok" or g(r, "compression_ratio") <= 0:
            continue
        ci = int(g(r, "chunk_idx"))
        ratio = g(r, "compression_ratio")
        if ratio > ub_configs.get(ci, (0, "", ""))[0]:
            algo = r.get("algorithm", "?")
            shuf = int(g(r, "shuffle_bytes"))
            quant = int(g(r, "quantization"))
            tag = algo
            if shuf:
                tag += f"+s{shuf}"
            if quant:
                tag += "+q"
            ub_configs[ci] = (ratio, tag, algo)

    bl_configs = {}
    for r in chunks_rows:
        if r.get("study") == "baseline":
            ci = int(g(r, "chunk_idx"))
            algo, tag = _parse_nn_action(r.get("nn_action", ""))
            bl_configs[ci] = (g(r, "actual_ratio"), tag, algo)

    sgd_configs = {}
    sgd_label = ""
    if sgd_rows:
        best_sgd = max(sgd_rows, key=lambda r: g(r, "ratio"))
        best_lr = g(best_sgd, "lr")
        best_mt = g(best_sgd, "mape_threshold")
        sgd_label = f"lr={best_lr:.3f}, mt={best_mt:.2f}"
        for r in chunks_rows:
            if (r.get("study") == "sgd"
                    and abs(g(r, "lr") - best_lr) < 1e-6
                    and abs(g(r, "mape_threshold") - best_mt) < 1e-6):
                ci = int(g(r, "chunk_idx"))
                algo, tag = _parse_nn_action(r.get("nn_action", ""))
                sgd_configs[ci] = (g(r, "actual_ratio"), tag, algo)

    all_indices = sorted(set(list(ub_configs.keys()) + list(bl_configs.keys())
                             + list(sgd_configs.keys())))
    if not all_indices:
        return
    n_chunks = max(all_indices) + 1

    # Collect all unique config tags for color mapping
    all_tags = set()
    for configs in [ub_configs, bl_configs, sgd_configs]:
        for ci in all_indices:
            if ci in configs:
                all_tags.add(configs[ci][1])
    tag_list = sorted(all_tags)
    config_cmap = plt.cm.tab10
    tag_colors = {t: config_cmap(i % 10) for i, t in enumerate(tag_list)}

    series = [
        ("Upper Bound (best-static)", ub_configs, "#e74c3c", "o"),
        ("Baseline (NN inference-only)", bl_configs, "#2980b9", "s"),
        (f"Best SGD ({sgd_label})" if sgd_label else "Best SGD", sgd_configs, "#27ae60", "^"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Per-Chunk Config & Ratio (All Patterns)", fontsize=14,
                 fontweight="bold", y=1.03)
    fig.text(0.5, 1.01,
             "Algorithm selection and compression ratio per chunk. Markers colored by config.\n"
             "Shows which algorithm each method picks across all pattern regions.",
             ha="center", fontsize=8.5, color="#555", va="top", style="italic")

    for ax, (label, configs, line_color, marker) in zip(axes, series):
        _draw_pattern_bands(ax, n_chunks)

        ratios = [configs.get(ci, (0, "", ""))[0] for ci in all_indices]
        tags = [configs.get(ci, (0, "unknown", ""))[1] for ci in all_indices]

        if not any(r > 0 for r in ratios):
            ax.text(0.5, 0.5, f"{label}: no data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="gray")
            ax.set_ylabel("Ratio")
            ax.set_title(label, fontsize=11)
            continue

        # Plot connecting line
        ax.plot(all_indices, ratios, color=line_color, linewidth=1.0,
                alpha=0.4, zorder=2)

        # Plot colored markers by config
        for ci, ratio, tag in zip(all_indices, ratios, tags):
            if ratio > 0:
                c = tag_colors.get(tag, "gray")
                ax.scatter(ci, ratio, color=c, marker=marker, s=40,
                           edgecolors="white", linewidths=0.3, zorder=3)

        # Config-change annotations: label runs of same config
        runs = []
        for i, tag in enumerate(tags):
            if runs and runs[-1][2] == tag:
                runs[-1] = (runs[-1][0], i, tag)
            else:
                runs.append((i, i, tag))

        for start, end, tag in runs:
            if tag == "" or tag == "unknown":
                continue
            mid_idx = (start + end) // 2
            mid_x = all_indices[mid_idx] if mid_idx < len(all_indices) else all_indices[-1]
            peak = max(ratios[start:end + 1]) if ratios[start:end + 1] else 0
            if peak > 0:
                ax.annotate(tag, (mid_x, peak), fontsize=6,
                            ha="center", va="bottom", rotation=45,
                            textcoords="offset points", xytext=(0, 5),
                            color=tag_colors.get(tag, "gray"),
                            fontweight="bold", zorder=4)

        avg = np.mean([r for r in ratios if r > 0])
        ax.set_ylabel("Ratio")
        ax.set_title(f"{label}  (avg={avg:.2f}x)", fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

        # Use log scale if range is large
        valid = [r for r in ratios if r > 0]
        if valid and max(valid) / max(min(valid), 0.01) > 5:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(
                lambda v, _: f"{v:.1f}x" if v < 10 else f"{v:.0f}x"))

    axes[-1].set_xlabel("Chunk Index")

    if len(all_indices) > 40:
        step = max(1, len(all_indices) // 25)
        axes[-1].set_xticks(all_indices[::step])
    else:
        axes[-1].set_xticks(all_indices)

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

    fig.tight_layout()
    path = os.path.join(output_dir, "per_chunk_config.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NN Adaptiveness Benchmark Visualizer")
    parser.add_argument("--sgd-csv", default=None)
    parser.add_argument("--chunks-csv", default=None)
    parser.add_argument("--ub-csv", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_DIR)
    args = parser.parse_args()

    sgd_csv = args.sgd_csv or os.path.join(DEFAULT_DIR, "sgd_study.csv")
    chunks_csv = args.chunks_csv or os.path.join(DEFAULT_DIR, "chunks_detail.csv")
    ub_csv = args.ub_csv or os.path.join(DEFAULT_DIR, "upper_bound.csv")
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    sgd_rows, chunks_rows, ub_rows = [], [], []

    if os.path.exists(sgd_csv):
        sgd_rows = parse_csv(sgd_csv)
        print(f"Loaded {len(sgd_rows)} SGD rows from {sgd_csv}")
    else:
        print(f"WARNING: SGD CSV not found: {sgd_csv}")

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

    if not sgd_rows and not chunks_rows and not ub_rows:
        print("ERROR: No data to visualize. Run the benchmark first.")
        sys.exit(1)

    print("\nGenerating plots...")

    if sgd_rows:
        plot_sgd_heatmap(sgd_rows, output_dir)
        if chunks_rows:
            plot_sgd_convergence(chunks_rows, sgd_rows, output_dir)

    if chunks_rows or ub_rows:
        plot_per_chunk_comparison(chunks_rows, sgd_rows, ub_rows, output_dir)

    if ub_rows:
        plot_upper_bound_configs(ub_rows, chunks_rows, sgd_rows, output_dir)

    if ub_rows and chunks_rows:
        plot_config_crosscheck(ub_rows, chunks_rows, sgd_rows, output_dir)

    if sgd_rows or ub_rows or chunks_rows:
        plot_summary(sgd_rows, ub_rows, chunks_rows, output_dir)

    if chunks_rows:
        plot_fig1_avg_mape(chunks_rows, sgd_rows, output_dir)
        plot_chunk_mape_comparison(chunks_rows, sgd_rows, output_dir)

    if ub_rows or chunks_rows:
        plot_fig4_compression_ratio(sgd_rows, ub_rows, chunks_rows, output_dir)

    if chunks_rows or ub_rows:
        plot_per_chunk_config(chunks_rows, sgd_rows, ub_rows, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
