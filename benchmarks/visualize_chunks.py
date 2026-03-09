#!/usr/bin/env python3
"""
Visualize per-chunk algorithm selection across benchmark phases and timesteps.

Usage:
    .venv/bin/python3 benchmarks/visualize_chunks.py <chunks.csv> [--out fig.png]

Both VPIC and Gray-Scott benchmarks use a unified CSV schema with action_final as a
human-readable string.  When multiple timesteps are present, generates a heatmap
(x=chunk, y=timestep) for the adaptive phase (nn-rl+exp50) showing algorithm selection
over time.
"""
import argparse
import csv
import os
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Action decoding (matches gpucompress action encoding) ────────
ALGO_NAMES = ["lz4", "snappy", "deflate", "gdeflate", "zstd", "ans", "cascaded", "bitcomp"]

def decode_action(action_int):
    algo = action_int % 8
    quant = (action_int // 8) % 2
    shuf = (action_int // 16) % 2
    s = ALGO_NAMES[algo]
    if shuf: s += "+shuf"
    if quant: s += "+quant"
    return s

# ── Color palette for algorithms ─────────────────────────────────
ALGO_COLORS = OrderedDict([
    ("lz4",              "#1f77b4"),
    ("lz4+shuf",         "#aec7e8"),
    ("lz4+quant",        "#08519c"),
    ("lz4+shuf+quant",   "#6baed6"),
    ("snappy",           "#ff7f0e"),
    ("snappy+shuf",      "#ffbb78"),
    ("snappy+quant",     "#e6550d"),
    ("snappy+shuf+quant","#fdae6b"),
    ("deflate",          "#2ca02c"),
    ("deflate+shuf",     "#98df8a"),
    ("deflate+quant",    "#006d2c"),
    ("deflate+shuf+quant","#74c476"),
    ("gdeflate",         "#d62728"),
    ("gdeflate+shuf",    "#ff9896"),
    ("gdeflate+quant",   "#a50f15"),
    ("gdeflate+shuf+quant","#fc9272"),
    ("zstd",             "#9467bd"),
    ("zstd+shuf",        "#c5b0d5"),
    ("zstd+quant",       "#6a3d9a"),
    ("zstd+shuf+quant",  "#b294c7"),
    ("ans",              "#8c564b"),
    ("ans+shuf",         "#c49c94"),
    ("ans+quant",        "#5b3a29"),
    ("ans+shuf+quant",   "#a97e6e"),
    ("cascaded",         "#e377c2"),
    ("cascaded+shuf",    "#f7b6d2"),
    ("cascaded+quant",   "#c51b8a"),
    ("cascaded+shuf+quant","#f768a1"),
    ("bitcomp",          "#7f7f7f"),
    ("bitcomp+shuf",     "#c7c7c7"),
    ("bitcomp+quant",    "#525252"),
    ("bitcomp+shuf+quant","#969696"),
])

def get_color(algo_str):
    return ALGO_COLORS.get(algo_str, "#333333")

# ── Parse CSV ────────────────────────────────────────────────────
Row = tuple  # (timestep, chunk, algo, ratio, pred, sgd, expl)

def load_csv(path):
    """Returns {(timestep, phase): [Row, ...]} and set of all timesteps."""
    data = OrderedDict()
    timesteps = set()
    with open(path) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        has_timestep = "timestep" in fields
        for row in reader:
            ts = int(row["timestep"]) if has_timestep else 0
            phase = row["phase"]
            if phase == "oracle":
                phase = "exhaustive"
            chunk = int(row["chunk"])
            algo = row.get("action_final", "")
            if not algo:
                # Legacy fallback: integer nn_action column
                action = int(row.get("nn_action", 0))
                algo = decode_action(action)
            ratio = float(row.get("actual_ratio", 0))
            pred  = float(row.get("predicted_ratio", 0))
            sgd   = int(row.get("sgd_fired", 0))
            expl  = int(row.get("exploration_triggered", row.get("explored", 0)))
            key = (ts, phase)
            timesteps.add(ts)
            if key not in data:
                data[key] = []
            data[key].append((ts, chunk, algo, ratio, pred, sgd, expl))
    return data, sorted(timesteps)


# ── Single-timestep plot (original style) ────────────────────────
def plot_single_timestep(data, timesteps, out_path, title=None):
    """Plot for single-timestep data: one row per phase."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    ts = timesteps[0]
    # Collect phases at this timestep
    phases = OrderedDict()
    for (t, phase), recs in data.items():
        if t == ts:
            phases[phase] = recs

    phase_names = list(phases.keys())
    n_phases = len(phase_names)
    has_ratio = any(r[3] > 0 for recs in phases.values() for r in recs)

    exhaustive_ratios = None
    if "exhaustive" in phases:
        exhaustive_ratios = [r[3] for r in phases["exhaustive"]]

    if has_ratio:
        fig, axes = plt.subplots(n_phases, 2, figsize=(18, 3.0 * n_phases + 1.5),
                                 gridspec_kw={"width_ratios": [3, 1]})
        if n_phases == 1:
            axes = [axes]
    else:
        fig, axes_raw = plt.subplots(n_phases, 1, figsize=(14, 2.5 * n_phases + 1.5))
        if n_phases == 1:
            axes_raw = [axes_raw]
        axes = [(ax, None) for ax in axes_raw]

    all_algos = OrderedDict()
    for recs in phases.values():
        for r in recs:
            if r[2] not in all_algos:
                all_algos[r[2]] = get_color(r[2])

    for idx, phase in enumerate(phase_names):
        recs = phases[phase]
        n_chunks = len(recs)
        algos  = [r[2] for r in recs]
        ratios = [r[3] for r in recs]
        preds  = [r[4] for r in recs]
        sgds   = [r[5] for r in recs]
        expls  = [r[6] for r in recs]
        colors = [all_algos[a] for a in algos]

        if has_ratio:
            ax_bar, ax_ratio = axes[idx]
        else:
            ax_bar, ax_ratio = axes[idx]

        for i, color in enumerate(colors):
            ax_bar.barh(0, 1, left=i, color=color, edgecolor="white", linewidth=0.3)
            if sgds[i]:
                ax_bar.plot(i + 0.5, 0.35, marker="v", color="black", markersize=4)
            if expls[i]:
                ax_bar.plot(i + 0.5, -0.35, marker="*", color="red", markersize=5)

        ax_bar.set_xlim(0, n_chunks)
        ax_bar.set_ylim(-0.5, 0.5)
        ax_bar.set_yticks([])
        ax_bar.set_xlabel("Chunk index")
        ax_bar.set_title(f"Phase: {phase}  ({n_chunks} chunks)", fontsize=11, fontweight="bold")

        if ax_ratio is not None and has_ratio:
            x = np.arange(n_chunks)
            ax_ratio.bar(x, ratios, color=colors, edgecolor="white", linewidth=0.3, width=1.0)
            if exhaustive_ratios is not None and phase != "exhaustive" and len(exhaustive_ratios) == n_chunks:
                ax_ratio.plot(x, exhaustive_ratios, color="goldenrod", linewidth=1.5,
                              alpha=0.85, label="exhaustive best", zorder=5)
            all_y = list(ratios)
            if exhaustive_ratios and phase != "exhaustive":
                all_y += list(exhaustive_ratios)
            all_y_sorted = sorted(all_y)
            p95 = all_y_sorted[min(len(all_y_sorted) - 1, int(0.95 * len(all_y_sorted)))]
            y_ceil = p95 * 1.5
            if any(p > 0 for p in preds):
                in_range = sum(1 for p in preds if 0 < p <= y_ceil) / max(len(preds), 1)
                if in_range > 0.3:
                    clipped = [min(p, y_ceil) for p in preds]
                    ax_ratio.plot(x, clipped, "k--", linewidth=0.8, alpha=0.5, label="predicted (clipped)")
                else:
                    med_pred = np.median([p for p in preds if p > 0])
                    ax_ratio.text(0.5, 0.95, f"predicted median: {med_pred:.0f}x (off-scale)",
                                  transform=ax_ratio.transAxes, fontsize=7, ha="center", va="top",
                                  bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", alpha=0.8))
            ax_ratio.set_ylim(0, y_ceil)
            handles, labels = ax_ratio.get_legend_handles_labels()
            if handles:
                ax_ratio.legend(fontsize=7, loc="upper right")
            ax_ratio.set_xlabel("Chunk index")
            ax_ratio.set_ylabel("Compression ratio")
            ax_ratio.set_title("Per-chunk ratio", fontsize=10)
            ax_ratio.set_xlim(-0.5, n_chunks - 0.5)

    legend_elements = [Patch(facecolor=c, edgecolor="gray", label=a) for a, c in all_algos.items()]
    legend_elements.append(Line2D([0], [0], marker="v", color="w", markerfacecolor="black",
                                  markersize=6, label="SGD fired"))
    legend_elements.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="red",
                                  markersize=8, label="Exploration"))
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=min(len(legend_elements), 8), fontsize=8,
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


# ── Multi-timestep heatmap ───────────────────────────────────────
def plot_multi_timestep(data, timesteps, out_path, title=None):
    """Heatmap: x=chunk, y=timestep. Top row = oracle (t=0), then nn-rl+exp50 per timestep.
    Right panel: per-chunk ratio over time."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.colors import ListedColormap, BoundaryNorm

    # Collect oracle data (t=0 baseline)
    exhaustive_key = (0, "exhaustive")
    exhaustive_recs = data.get(exhaustive_key, [])
    n_chunks = len(exhaustive_recs) if exhaustive_recs else 0

    # Collect nn-rl+exp50 across all timesteps
    adapt_phase = "nn-rl+exp50"
    adapt_ts = []
    for ts in timesteps:
        key = (ts, adapt_phase)
        if key in data:
            adapt_ts.append(ts)

    if not adapt_ts:
        print("No multi-timestep nn-rl+exp50 data found, falling back to single-timestep plot",
              file=sys.stderr)
        return plot_single_timestep(data, timesteps, out_path, title)

    # Also collect t=0 phases for the initial comparison
    t0_phases = OrderedDict()
    for (t, phase), recs in data.items():
        if t == 0:
            t0_phases[phase] = recs

    # Build algorithm → integer mapping for heatmap
    all_algos = OrderedDict()
    for recs in data.values():
        for r in recs:
            if r[2] not in all_algos:
                all_algos[r[2]] = get_color(r[2])
    algo_list = list(all_algos.keys())
    algo_to_idx = {a: i for i, a in enumerate(algo_list)}

    # Build heatmap matrix: rows = [oracle, nn(t=0), nn-rl+exp50(t=0), nn-rl+exp50(t=1), ...]
    row_labels = []
    heatmap_rows = []
    ratio_rows = []

    # Oracle row
    if exhaustive_recs:
        row_labels.append("exhaustive (t=0)")
        heatmap_rows.append([algo_to_idx.get(r[2], 0) for r in exhaustive_recs])
        ratio_rows.append([r[3] for r in exhaustive_recs])

    # nn at t=0
    nn_key = (0, "nn")
    if nn_key in data:
        recs = data[nn_key]
        row_labels.append("nn (t=0)")
        heatmap_rows.append([algo_to_idx.get(r[2], 0) for r in recs])
        ratio_rows.append([r[3] for r in recs])

    # nn-rl+exp50 across all timesteps
    for ts in adapt_ts:
        recs = data[(ts, adapt_phase)]
        row_labels.append(f"nn-rl+exp50 (t={ts})")
        heatmap_rows.append([algo_to_idx.get(r[2], 0) for r in recs])
        ratio_rows.append([r[3] for r in recs])

    n_rows = len(heatmap_rows)
    if n_chunks == 0:
        n_chunks = max(len(r) for r in heatmap_rows)

    # Pad rows to n_chunks if needed (partial last chunk)
    for i in range(n_rows):
        while len(heatmap_rows[i]) < n_chunks:
            heatmap_rows[i].append(0)
        while len(ratio_rows[i]) < n_chunks:
            ratio_rows[i].append(0)

    hmap = np.array(heatmap_rows)
    rmap = np.array(ratio_rows)

    # Build colormap from algo colors
    cmap_colors = [all_algos[a] for a in algo_list]
    cmap = ListedColormap(cmap_colors)
    bounds = np.arange(len(algo_list) + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)

    # ── Figure: heatmap (left) + ratio heatmap (right) ──
    fig_h = max(4, 0.4 * n_rows + 2.5)
    fig, (ax_heat, ax_rat) = plt.subplots(1, 2, figsize=(18, fig_h),
                                           gridspec_kw={"width_ratios": [2, 1]})

    # Algorithm heatmap
    im = ax_heat.imshow(hmap, aspect="auto", cmap=cmap, norm=norm,
                        interpolation="nearest", origin="upper")
    ax_heat.set_yticks(np.arange(n_rows))
    ax_heat.set_yticklabels(row_labels, fontsize=8)
    ax_heat.set_xlabel("Chunk index", fontsize=10)
    ax_heat.set_title("Algorithm selection per chunk", fontsize=12, fontweight="bold")

    # Add a horizontal line separating baseline rows from timestep rows
    n_baseline = 1 + (1 if nn_key in data else 0)  # oracle + nn
    if n_baseline < n_rows:
        ax_heat.axhline(n_baseline - 0.5, color="white", linewidth=2)

    # Ratio heatmap
    exhaustive_rats = np.array(ratio_rows[0]) if exhaustive_recs else None
    # Use p95 for vmax to avoid outlier tail chunks
    all_rats = rmap[rmap > 0]
    if len(all_rats) > 0:
        vmax = np.percentile(all_rats, 95) * 1.3
    else:
        vmax = 1
    im2 = ax_rat.imshow(rmap, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax,
                         interpolation="nearest", origin="upper")
    ax_rat.set_yticks(np.arange(n_rows))
    ax_rat.set_yticklabels(row_labels, fontsize=8)
    ax_rat.set_xlabel("Chunk index", fontsize=10)
    ax_rat.set_title("Compression ratio per chunk", fontsize=12, fontweight="bold")
    if n_baseline < n_rows:
        ax_rat.axhline(n_baseline - 0.5, color="white", linewidth=2)
    fig.colorbar(im2, ax=ax_rat, label="Compression ratio", shrink=0.8)

    # Algorithm legend
    legend_elements = [Patch(facecolor=c, edgecolor="gray", label=a) for a, c in all_algos.items()]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=min(len(legend_elements), 8), fontsize=8,
               bbox_to_anchor=(0.5, -0.04), frameon=True)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Also generate per-timestep ratio trend line
    out_dir = os.path.dirname(out_path)
    trend_path = os.path.join(out_dir, "chunks_ratio_trend.png")
    plot_ratio_trend(data, adapt_ts, exhaustive_recs, n_chunks, trend_path, title)


def plot_ratio_trend(data, adapt_ts, exhaustive_recs, n_chunks, out_path, title=None):
    """Line plot: mean compression ratio per timestep vs oracle baseline."""
    adapt_phase = "nn-rl+exp50"
    mean_ratios = []
    sgd_counts = []
    expl_counts = []
    for ts in adapt_ts:
        recs = data[(ts, adapt_phase)]
        rats = [r[3] for r in recs if r[3] > 0]
        mean_ratios.append(np.mean(rats) if rats else 0)
        sgd_counts.append(sum(1 for r in recs if r[5]))
        expl_counts.append(sum(1 for r in recs if r[6]))

    exhaustive_mean = 0
    if exhaustive_recs:
        exhaustive_mean = np.mean([r[3] for r in exhaustive_recs if r[3] > 0])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(adapt_ts, mean_ratios, "o-", color="#9467bd", linewidth=2, label="nn-rl+exp50")
    if exhaustive_mean > 0:
        ax1.axhline(exhaustive_mean, color="goldenrod", linewidth=1.5, linestyle="--", label=f"exhaustive ({exhaustive_mean:.1f}x)")
    ax1.set_ylabel("Mean compression ratio")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_title("Compression ratio over timesteps" + (f" — {title}" if title else ""),
                   fontsize=11, fontweight="bold")

    ax2.bar(adapt_ts, sgd_counts, color="#2ca02c", alpha=0.7, label="SGD fires")
    ax2.bar(adapt_ts, expl_counts, color="#d62728", alpha=0.7, bottom=sgd_counts, label="Explorations")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Chunks affected")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


# ── Auto-detection ────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_CHUNK_PATHS = [
    os.path.join(PROJECT_ROOT, "benchmarks/grayscott/results/benchmark_grayscott_vol_chunks.csv"),
    os.path.join(PROJECT_ROOT, "benchmarks/vpic-kokkos/results/benchmark_vpic_deck_chunks.csv"),
    os.path.join(PROJECT_ROOT, "benchmarks/grayscott/benchmark_grayscott_vol_chunks.csv"),
    os.path.join(PROJECT_ROOT, "benchmarks/vpic-kokkos/benchmark_vpic_deck_chunks.csv"),
]


def find_chunk_csv():
    for p in DEFAULT_CHUNK_PATHS:
        if os.path.exists(p):
            return p
    return None


# ── Main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Visualize per-chunk algorithm selection")
    parser.add_argument("csv", nargs="?", default=None,
                        help="Path to benchmark chunks CSV (auto-detected if omitted)")
    parser.add_argument("--out", default=None, help="Output image path (default: <csv_dir>/chunks_viz.png)")
    parser.add_argument("--title", default=None, help="Figure title")
    args = parser.parse_args()

    if args.csv is None:
        args.csv = find_chunk_csv()
        if args.csv is None:
            print("ERROR: No chunk CSV found. Expected locations:", file=sys.stderr)
            for p in DEFAULT_CHUNK_PATHS:
                print(f"  {p}", file=sys.stderr)
            print("\nRun benchmarks first, or pass a CSV path as argument.", file=sys.stderr)
            sys.exit(1)
        print(f"Auto-detected: {args.csv}")

    if args.out is None:
        csv_dir = os.path.dirname(os.path.abspath(args.csv))
        args.out = os.path.join(csv_dir, "chunks_viz.png")

    data, timesteps = load_csv(args.csv)
    if not data:
        print("No data found in CSV", file=sys.stderr)
        sys.exit(1)

    if len(timesteps) > 1:
        plot_multi_timestep(data, timesteps, args.out, title=args.title)
    else:
        plot_single_timestep(data, timesteps, args.out, title=args.title)

if __name__ == "__main__":
    main()
