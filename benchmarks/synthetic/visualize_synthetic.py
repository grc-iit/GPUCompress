#!/usr/bin/env python3
"""
Synthetic workloads benchmark visualizer for GPUCompress.

Generates:
  1. Summary bar charts       — ratio, unique configs, throughput, MAPE
  2. Pattern config heatmap   — which config per pattern per phase
  3. Feature scatter           — entropy/MAD/deriv colored by config
  4. Chunk timeline            — config strip across timesteps + T=0 detail
  5. All-timesteps detail      — per-chunk config+ratio grid for every timestep
  6. Timestep adaptation       — MAPE + unique configs over timesteps

Usage:
  # Auto-detect CSVs in default path
  python3 benchmarks/synthetic/visualize_synthetic.py

  # Explicit aggregate CSV
  python3 benchmarks/synthetic/visualize_synthetic.py \\
      --agg-csv benchmarks/synthetic/results/synthetic_workloads_aggregate.csv

  # Custom output directory
  python3 benchmarks/synthetic/visualize_synthetic.py --output-dir results/plots
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects
import matplotlib.patches
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# Constants & Utilities
# ═══════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

PHASE_COLORS = {
    "nn":          "#e49444",
    "nn-rl":       "#6a9f58",
    "nn-rl+exp50": "#c85a5a",
    "nn-rl+exp":   "#c85a5a",
}

# Canonical algorithm names
ALGO_NAMES = ["lz4", "snappy", "deflate", "gdeflate",
              "zstd", "ans", "cascaded", "bitcomp"]

# 32-config color palette
_CONFIG_PALETTE = {
    "lz4": "#2166ac", "lz4+shuf": "#67a9cf",
    "lz4+quant": "#053061", "lz4+shuf+quant": "#a6cee3",
    "snappy": "#e08214", "snappy+shuf": "#fdb863",
    "snappy+quant": "#b35806", "snappy+shuf+quant": "#fee0b6",
    "deflate": "#d6301d", "deflate+shuf": "#fc8d59",
    "deflate+quant": "#7f0000", "deflate+shuf+quant": "#fddbc7",
    "gdeflate": "#1b7837", "gdeflate+shuf": "#7fbf7b",
    "gdeflate+quant": "#00441b", "gdeflate+shuf+quant": "#d9f0d3",
    "zstd": "#762a83", "zstd+shuf": "#af8dc3",
    "zstd+quant": "#40004b", "zstd+shuf+quant": "#e7d4e8",
    "ans": "#b8860b", "ans+shuf": "#daa520",
    "ans+quant": "#8b6914", "ans+shuf+quant": "#ffd700",
    "cascaded": "#c51b7d", "cascaded+shuf": "#e9a3c9",
    "cascaded+quant": "#8e0152", "cascaded+shuf+quant": "#fde0ef",
    "bitcomp": "#01665e", "bitcomp+shuf": "#5ab4ac",
    "bitcomp+quant": "#003c30", "bitcomp+shuf+quant": "#c7eae5",
}

CONFIG_ORDER = []
CONFIG_COLORS = {}
for algo in ALGO_NAMES:
    for suffix in ["", "+shuf", "+quant", "+shuf+quant"]:
        name = algo + suffix
        CONFIG_ORDER.append(name)
        CONFIG_COLORS[name] = _CONFIG_PALETTE.get(name, "#cccccc")

# Auto-detection paths
DEFAULT_AGG = [
    os.path.join(PROJECT_ROOT, "benchmarks/synthetic/results/synthetic_workloads_aggregate.csv"),
    os.path.join(PROJECT_ROOT, "results/synthetic_workloads_aggregate.csv"),
]
DEFAULT_CHUNKS = [
    os.path.join(PROJECT_ROOT, "benchmarks/synthetic/results/synthetic_workloads_chunks.csv"),
    os.path.join(PROJECT_ROOT, "results/synthetic_workloads_chunks.csv"),
]
DEFAULT_TIMESTEPS = [
    os.path.join(PROJECT_ROOT, "benchmarks/synthetic/results/synthetic_workloads_timesteps.csv"),
    os.path.join(PROJECT_ROOT, "results/synthetic_workloads_timesteps.csv"),
]


def find_csv(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def parse_csv(path):
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
    for k in keys:
        if k in row:
            return row[k]
    return default


def _normalize_action(action_str):
    if not action_str or action_str == "none":
        return "none"
    parts = action_str.lower().split("+")
    algo = parts[0]
    shuf = "shuf" in parts
    quant = "quant" in parts
    name = algo
    if shuf:
        name += "+shuf"
    if quant:
        name += "+quant"
    return name


def _build_config_cmap():
    from matplotlib.colors import ListedColormap, BoundaryNorm
    colors = [CONFIG_COLORS.get(c, "#cccccc") for c in CONFIG_ORDER]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(CONFIG_ORDER) + 0.5, 1), cmap.N)
    return cmap, norm


def _config_to_idx(action_str):
    name = _normalize_action(str(action_str))
    try:
        return CONFIG_ORDER.index(name)
    except ValueError:
        return -1


# ═══════════════════════════════════════════════════════════════════════
# Plot 1: Summary bar charts
# ═══════════════════════════════════════════════════════════════════════

def make_summary_figure(rows, output_path):
    if not rows:
        return
    combos = []
    by_combo = {}
    for r in rows:
        key = (r.get("phase", "?"), r.get("mode", "?"))
        if key not in by_combo:
            combos.append(key)
        by_combo[key] = r
    n = len(combos)
    if n == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="white")
    fig.suptitle("Synthetic Scientific Workloads Benchmark",
                 fontsize=14, fontweight="bold", y=0.98)
    labels = [f"{ph}\n({mode})" for ph, mode in combos]
    x = np.arange(n)
    colors = [PHASE_COLORS.get(c[0], "#bdc3c7") for c in combos]

    # Ratio
    ax = axes[0, 0]
    vals = [g(by_combo[c], "ratio") for c in combos]
    bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5, width=0.6)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.2f}x", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Ratio"); ax.set_title("Compression Ratio", fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--"); ax.set_axisbelow(True)

    # Unique configs
    ax = axes[0, 1]
    vals = [g(by_combo[c], "n_unique_configs") for c in combos]
    bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5, width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{int(v)}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Count"); ax.set_title("Unique Configs Selected (out of 32)", fontweight="bold")
    ax.set_ylim(0, 32)
    ax.axhline(1, color="red", linewidth=1, linestyle="--", alpha=0.5, label="1 = no differentiation")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle="--"); ax.set_axisbelow(True)

    # Write throughput
    ax = axes[1, 0]
    vals = [g(by_combo[c], "write_mibps", "write_mbps") for c in combos]
    bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5, width=0.6)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("MiB/s"); ax.set_title("Write Throughput", fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--"); ax.set_axisbelow(True)

    # MAPE
    ax = axes[1, 1]
    w = 0.25
    for i, (mkey, color, label) in enumerate([
        ("mape_ratio_pct", "#2c3e50", "Ratio"),
        ("mape_comp_pct", "#3498db", "Comp Time"),
        ("mape_decomp_pct", "#e74c3c", "Decomp Time"),
    ]):
        vals = [g(by_combo[c], mkey) for c in combos]
        ax.bar(x + (i - 1) * w, vals, w, color=color, edgecolor="black",
               linewidth=0.5, label=label)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("MAPE (%)"); ax.set_title("Prediction Accuracy (MAPE)", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3, linestyle="--"); ax.set_axisbelow(True)

    for row in axes:
        for ax in row:
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 2: Pattern config heatmap
# ═══════════════════════════════════════════════════════════════════════

def make_pattern_heatmap(chunk_csv_path, output_path):
    rows = parse_csv(chunk_csv_path)
    if not rows:
        return
    ss_rows = [r for r in rows if int(g(r, "timestep", default=-1)) == -1]
    if not ss_rows:
        ss_rows = rows

    by_phase_mode = {}
    for r in ss_rows:
        key = (r.get("phase", "?"), r.get("mode", "?"))
        pattern = r.get("pattern", f"chunk_{int(g(r, 'chunk'))}")
        by_phase_mode.setdefault(key, {}).setdefault(pattern, []).append(r)

    if not by_phase_mode:
        return

    pattern_order = []
    seen = set()
    for r in ss_rows:
        p = r.get("pattern", f"chunk_{int(g(r, 'chunk'))}")
        if p not in seen:
            pattern_order.append(p)
            seen.add(p)

    combo_order = []
    seen_combo = set()
    for r in ss_rows:
        key = (r.get("phase", "?"), r.get("mode", "?"))
        if key not in seen_combo:
            combo_order.append(key)
            seen_combo.add(key)

    n_patterns = len(pattern_order)
    n_combos = len(combo_order)
    cmap, norm = _build_config_cmap()

    heatmap = np.full((n_patterns, n_combos), np.nan)
    config_labels = {}

    for ci, combo in enumerate(combo_order):
        pat_data = by_phase_mode.get(combo, {})
        for pi, pat in enumerate(pattern_order):
            chunk_rows = pat_data.get(pat, [])
            if not chunk_rows:
                continue
            config_counts = {}
            for r in chunk_rows:
                cfg = _normalize_action(r.get("action_final", r.get("action", "none")))
                config_counts[cfg] = config_counts.get(cfg, 0) + 1
            best = max(config_counts, key=config_counts.get)
            heatmap[pi, ci] = _config_to_idx(best)
            config_labels[(pi, ci)] = best

    fig_height = max(8, 0.4 * n_patterns + 3)
    fig, ax = plt.subplots(figsize=(max(8, 2.5 * n_combos + 3), fig_height),
                            facecolor="white")
    fig.suptitle("NN Config Selection by Data Pattern\n"
                 "(Does the NN differentiate between patterns?)",
                 fontsize=14, fontweight="bold", y=0.99)

    ax.imshow(heatmap, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    for pi in range(n_patterns):
        for ci in range(n_combos):
            label = config_labels.get((pi, ci), "")
            if label and label != "none":
                short = label.replace("deflate", "defl").replace("cascaded", "casc") \
                             .replace("gdeflate", "gdefl").replace("snappy", "snap") \
                             .replace("bitcomp", "bitc")
                ax.text(ci, pi, short, ha="center", va="center",
                        fontsize=7, fontweight="bold", color="white",
                        path_effects=[matplotlib.patheffects.withStroke(
                            linewidth=2, foreground="black")])

    ax.set_xticks(range(n_combos))
    ax.set_xticklabels([f"{ph}\n({mode})" for ph, mode in combo_order],
                        fontsize=9, fontweight="bold")
    ax.set_yticks(range(n_patterns))
    ax.set_yticklabels(pattern_order, fontsize=9)
    ax.set_ylabel("Data Pattern", fontsize=11, fontweight="bold")
    ax.set_xlabel("Phase (Mode)", fontsize=11, fontweight="bold")

    for ci, combo in enumerate(combo_order):
        configs_in_col = set()
        for pi in range(n_patterns):
            label = config_labels.get((pi, ci), "")
            if label and label != "none":
                configs_in_col.add(label)
        ax.text(ci, -0.8, f"{len(configs_in_col)} unique",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color="#c0392b")

    all_cfgs = set(config_labels.values()) - {"", "none"}
    legend_patches = [matplotlib.patches.Patch(facecolor=CONFIG_COLORS[cfg],
                       edgecolor="black", linewidth=0.5, label=cfg)
                      for cfg in CONFIG_ORDER if cfg in all_cfgs]
    ax.legend(handles=legend_patches, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), fontsize=8, framealpha=0.9)

    fig.tight_layout(rect=[0, 0, 0.85, 0.95])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 3: Feature scatter
# ═══════════════════════════════════════════════════════════════════════

def make_feature_scatter(chunk_csv_path, output_path):
    rows = parse_csv(chunk_csv_path)
    if not rows:
        return
    ss_rows = [r for r in rows
               if r.get("phase", "") == "nn"
               and int(g(r, "timestep", default=-1)) == -1]
    if not ss_rows:
        ss_rows = [r for r in rows if int(g(r, "timestep", default=-1)) == -1]
    if not ss_rows:
        ss_rows = rows[:200]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="white")
    fig.suptitle("NN Input Features vs Config Selection (nn inference)",
                 fontsize=14, fontweight="bold", y=0.98)

    pairs = [
        ("feat_entropy", "feat_mad", "Entropy", "MAD"),
        ("feat_entropy", "feat_deriv", "Entropy", "2nd Derivative"),
        ("feat_mad", "feat_deriv", "MAD", "2nd Derivative"),
    ]

    for ax, (xkey, ykey, xlabel, ylabel) in zip(axes, pairs):
        for r in ss_rows:
            cfg = _normalize_action(r.get("action_final", r.get("action", "none")))
            color = CONFIG_COLORS.get(cfg, "#999999")
            ax.scatter(g(r, xkey), g(r, ykey), c=color, s=30, alpha=0.7,
                       edgecolor="black", linewidth=0.3, zorder=3)
        ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.2, linestyle="-")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    all_cfgs = set()
    for r in ss_rows:
        cfg = _normalize_action(r.get("action_final", r.get("action", "none")))
        if cfg != "none":
            all_cfgs.add(cfg)
    legend_patches = [matplotlib.patches.Patch(facecolor=CONFIG_COLORS.get(c, "#999"),
                       edgecolor="black", linewidth=0.5, label=c)
                      for c in CONFIG_ORDER if c in all_cfgs]
    fig.legend(handles=legend_patches, loc="lower center", fontsize=8,
               ncol=min(len(legend_patches), 8), framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))

    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 4: Chunk timeline (strip + T=0 detail)
# ═══════════════════════════════════════════════════════════════════════

def make_chunk_timeline(chunk_csv_path, output_path):
    rows = parse_csv(chunk_csv_path)
    if not rows:
        return

    rl_rows = [r for r in rows
                if r.get("phase", "") == "nn-rl"
                and int(g(r, "timestep", default=-1)) >= 0]
    nn_rows = [r for r in rows
                if r.get("phase", "") == "nn"
                and int(g(r, "timestep", default=-1)) == -1]
    if not rl_rows and not nn_rows:
        rl_rows = [r for r in rows if int(g(r, "timestep", default=-1)) >= 0]
        nn_rows = [r for r in rows if int(g(r, "timestep", default=-1)) == -1]
    if not rl_rows and not nn_rows:
        return

    cmap, norm = _build_config_cmap()

    by_ts = {}
    for r in rl_rows:
        ts = int(g(r, "timestep"))
        by_ts.setdefault(ts, []).append(r)
    all_ts = sorted(by_ts.keys())
    if nn_rows:
        by_ts[-1] = sorted(nn_rows, key=lambda r: int(g(r, "chunk")))
        all_ts = [-1] + all_ts

    n_ts = len(all_ts)
    max_chunks = max(len(by_ts[ts]) for ts in all_ts) if all_ts else 0
    if n_ts == 0 or max_chunks == 0:
        return

    strip_data = []
    ts_labels = []
    for ts in all_ts:
        ts_rows = sorted(by_ts[ts], key=lambda r: int(g(r, "chunk")))
        row_vals = [_config_to_idx(r.get("action_final", r.get("action", "none")))
                    for r in ts_rows]
        while len(row_vals) < max_chunks:
            row_vals.append(-1)
        strip_data.append(row_vals)
        ts_labels.append("nn (static)" if ts == -1 else f"T={ts}")

    strip_arr = np.array(strip_data, dtype=float)
    strip_arr[strip_arr < 0] = np.nan

    fig_height = max(6, 0.35 * n_ts + 6)
    fig = plt.figure(figsize=(18, fig_height), facecolor="white")
    gs = gridspec.GridSpec(2, 1, height_ratios=[n_ts, 4], hspace=0.35,
                           top=0.93, bottom=0.06, left=0.08, right=0.88)
    fig.suptitle("Per-Chunk Config Selection Within Each Timestep (nn-rl)\n"
                 "Each row = one dataset write; each cell = one chunk's config",
                 fontsize=13, fontweight="bold", y=0.98)

    ax_strip = fig.add_subplot(gs[0])
    ax_strip.imshow(strip_arr, aspect="auto", cmap=cmap, norm=norm,
                    interpolation="nearest")
    ax_strip.set_yticks(range(n_ts))
    ax_strip.set_yticklabels(ts_labels, fontsize=8)
    ax_strip.set_xlabel("Chunk Index", fontsize=10)
    ax_strip.set_title("Config Evolution Across Timesteps", fontsize=11, fontweight="bold")

    for ri, ts in enumerate(all_ts):
        cfgs = set()
        for r in by_ts[ts]:
            cfg = _normalize_action(r.get("action_final", r.get("action", "none")))
            if cfg != "none":
                cfgs.add(cfg)
        ax_strip.text(max_chunks + 0.5, ri, f"{len(cfgs)}",
                      ha="left", va="center", fontsize=9, fontweight="bold",
                      color="#c0392b")
    ax_strip.text(max_chunks + 0.5, -0.7, "Unique", ha="left", va="bottom",
                  fontsize=8, color="#c0392b", fontweight="bold")

    # T=0 detail
    detail_ts = 0 if 0 in by_ts else (all_ts[1] if len(all_ts) > 1 else all_ts[0])
    detail_rows = sorted(by_ts[detail_ts], key=lambda r: int(g(r, "chunk")))
    n_detail = len(detail_rows)

    ax_bar = fig.add_subplot(gs[1])
    x = np.arange(n_detail)
    ratios = np.array([g(r, "actual_ratio", default=1.0) for r in detail_rows])
    colors = []
    labels = []
    for r in detail_rows:
        cfg = _normalize_action(r.get("action_final", r.get("action", "none")))
        colors.append(CONFIG_COLORS.get(cfg, "#cccccc"))
        labels.append(cfg)

    bars = ax_bar.bar(x, ratios, color=colors, edgecolor="black", linewidth=0.5,
                      width=0.8, zorder=3)
    for xi, (bar, label) in enumerate(zip(bars, labels)):
        short = label.replace("deflate", "defl").replace("cascaded", "casc") \
                     .replace("gdeflate", "gdefl").replace("snappy", "snap") \
                     .replace("bitcomp", "bitc")
        rot = 90 if n_detail > 16 else 45
        fs = 7 if n_detail > 20 else 8
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    short, ha="center", va="bottom", fontsize=fs,
                    fontweight="bold", rotation=rot)

    pattern_labels = []
    for r in detail_rows:
        p = r.get("pattern", f"C{int(g(r, 'chunk'))}")
        p = p.replace("_", "\n", 1).replace("_", " ")
        pattern_labels.append(p)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(pattern_labels, fontsize=6, rotation=60, ha="right")
    ax_bar.set_ylabel("Compression Ratio", fontsize=10)
    ax_bar.set_title(f"Timestep {detail_ts}: Per-Chunk Config and Ratio",
                     fontsize=11, fontweight="bold")
    ax_bar.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
    ax_bar.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax_bar.spines[spine].set_visible(False)

    for xi, r in enumerate(detail_rows):
        if int(g(r, "sgd_fired", default=0)):
            ax_bar.plot(xi, -0.15, marker="^", color="#e74c3c", markersize=6,
                        clip_on=False, zorder=5)
    ax_bar.text(-0.5, -0.15, "SGD", fontsize=7, color="#e74c3c", ha="right",
                va="center", fontweight="bold")

    all_cfgs = set()
    for ts in all_ts:
        for r in by_ts[ts]:
            cfg = _normalize_action(r.get("action_final", r.get("action", "none")))
            if cfg != "none":
                all_cfgs.add(cfg)
    legend_patches = [matplotlib.patches.Patch(facecolor=CONFIG_COLORS.get(c, "#999"),
                       edgecolor="black", linewidth=0.5, label=c)
                      for c in CONFIG_ORDER if c in all_cfgs]
    fig.legend(handles=legend_patches, loc="center right",
               bbox_to_anchor=(0.98, 0.5), fontsize=8, framealpha=0.9)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 5: All-timesteps detail grid
# ═══════════════════════════════════════════════════════════════════════

def make_all_timesteps_detail(chunk_csv_path, output_path):
    rows = parse_csv(chunk_csv_path)
    if not rows:
        return

    rl_rows = [r for r in rows
                if r.get("phase", "") == "nn-rl"
                and int(g(r, "timestep", default=-1)) >= 0]
    nn_rows = [r for r in rows
                if r.get("phase", "") == "nn"
                and int(g(r, "timestep", default=-1)) == -1]
    if not rl_rows and not nn_rows:
        return

    by_ts = {}
    for r in rl_rows:
        ts = int(g(r, "timestep"))
        by_ts.setdefault(ts, []).append(r)
    if nn_rows:
        by_ts[-1] = nn_rows

    all_ts = sorted(by_ts.keys())
    n_ts = len(all_ts)
    if n_ts == 0:
        return

    max_chunks = max(len(by_ts[ts]) for ts in all_ts)

    n_cols = min(4, n_ts)
    n_rows_grid = (n_ts + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows_grid, n_cols,
                              figsize=(5 * n_cols, 3 * n_rows_grid + 1),
                              facecolor="white", squeeze=False)
    fig.suptitle("Per-Chunk Config & Ratio — Every Timestep (nn-rl)\n"
                 "Bar color = config chosen, height = compression ratio",
                 fontsize=13, fontweight="bold", y=0.99)

    all_ratios = []
    for ts in all_ts:
        for r in by_ts[ts]:
            all_ratios.append(g(r, "actual_ratio", default=1.0))
    y_max = min(np.percentile(all_ratios, 98) * 1.3, max(all_ratios) * 1.1) if all_ratios else 5
    y_max = max(y_max, 2.0)

    for idx, ts in enumerate(all_ts):
        row_i = idx // n_cols
        col_i = idx % n_cols
        ax = axes[row_i, col_i]

        ts_rows = sorted(by_ts[ts], key=lambda r: int(g(r, "chunk")))
        n_ch = len(ts_rows)
        x = np.arange(n_ch)
        ratios = np.array([g(r, "actual_ratio", default=1.0) for r in ts_rows])
        colors = [CONFIG_COLORS.get(
            _normalize_action(r.get("action_final", r.get("action", "none"))), "#cccccc")
            for r in ts_rows]

        ax.bar(x, ratios, color=colors, edgecolor="black", linewidth=0.3,
               width=0.85, zorder=3)

        cfgs = set()
        for r in ts_rows:
            cfg = _normalize_action(r.get("action_final", r.get("action", "none")))
            if cfg != "none":
                cfgs.add(cfg)

        label = "nn (static)" if ts == -1 else f"T={ts}"
        avg_ratio = np.mean(ratios) if len(ratios) > 0 else 0
        ax.set_title(f"{label}  |  {len(cfgs)} configs  |  avg {avg_ratio:.2f}x",
                     fontsize=9, fontweight="bold")
        ax.set_ylim(0, y_max)
        ax.set_xlim(-0.5, max_chunks - 0.5)
        ax.grid(axis="y", alpha=0.2, linestyle="--", zorder=0)
        ax.set_axisbelow(True)
        if col_i == 0:
            ax.set_ylabel("Ratio", fontsize=9)
        if row_i == n_rows_grid - 1:
            ax.set_xlabel("Chunk", fontsize=9)
        ax.tick_params(labelsize=7)

        for xi, r in enumerate(ts_rows):
            if int(g(r, "sgd_fired", default=0)):
                ax.plot(xi, -0.08 * y_max, marker="^", color="#e74c3c",
                        markersize=3, clip_on=False, zorder=5)

    for idx in range(n_ts, n_rows_grid * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    all_cfgs = set()
    for ts in all_ts:
        for r in by_ts[ts]:
            cfg = _normalize_action(r.get("action_final", r.get("action", "none")))
            if cfg != "none":
                all_cfgs.add(cfg)
    legend_patches = [matplotlib.patches.Patch(facecolor=CONFIG_COLORS.get(c, "#999"),
                       edgecolor="black", linewidth=0.5, label=c)
                      for c in CONFIG_ORDER if c in all_cfgs]
    fig.legend(handles=legend_patches, loc="lower center", fontsize=9,
               ncol=min(len(legend_patches), 8), framealpha=0.9,
               bbox_to_anchor=(0.5, 0.002))

    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 6: Timestep adaptation (MAPE + unique configs)
# ═══════════════════════════════════════════════════════════════════════

def make_timestep_figure(ts_csv_path, output_path):
    rows = parse_csv(ts_csv_path)
    if not rows:
        return

    by_key = {}
    for r in rows:
        key = (r.get("phase", "?"), r.get("mode", "?"))
        by_key.setdefault(key, []).append(r)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("Synthetic Workloads: SGD Adaptation Over Timesteps",
                 fontsize=14, fontweight="bold", y=0.98)

    style_cycle = [
        {"color": "#6a9f58", "ls": "-",  "marker": "o", "lw": 2.0},
        {"color": "#c85a5a", "ls": "--", "marker": "D", "lw": 1.8},
        {"color": "#e49444", "ls": ":",  "marker": "s", "lw": 1.5},
        {"color": "#5778a4", "ls": "-.", "marker": "^", "lw": 1.5},
    ]

    ax = axes[0]
    for si, (key, key_rows) in enumerate(sorted(by_key.items())):
        ph, mode = key
        sty = style_cycle[si % len(style_cycle)]
        ts = np.array([int(g(r, "timestep")) for r in key_rows])
        mape = np.array([g(r, "mape_ratio") for r in key_rows])
        ax.plot(ts, mape, label=f"{ph} ({mode})", **sty, markersize=5, alpha=0.9)
    ax.axhline(20, color="#e67e22", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_ylabel("Ratio MAPE (%)", fontsize=11, fontweight="bold")
    ax.set_title("Prediction Accuracy Over Timesteps", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, linestyle="--")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    for si, (key, key_rows) in enumerate(sorted(by_key.items())):
        ph, mode = key
        sty = style_cycle[si % len(style_cycle)]
        ts = np.array([int(g(r, "timestep")) for r in key_rows])
        ucfg = np.array([g(r, "n_unique_configs") for r in key_rows])
        ax.plot(ts, ucfg, label=f"{ph} ({mode})", **sty, markersize=5, alpha=0.9)
    ax.axhline(1, color="red", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_ylabel("Unique Configs", fontsize=11, fontweight="bold")
    ax.set_xlabel("Timestep", fontsize=11, fontweight="bold")
    ax.set_title("Config Diversity Over Timesteps", fontweight="bold")
    ax.set_ylim(0, 32)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, linestyle="--")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Synthetic workloads benchmark visualizer for GPUCompress")
    parser.add_argument("csvs", nargs="*", help="Positional CSV paths")
    parser.add_argument("--agg-csv", help="Aggregate CSV path")
    parser.add_argument("--chunks-csv", help="Per-chunk CSV path")
    parser.add_argument("--timesteps-csv", help="Timestep CSV path")
    parser.add_argument("--output-dir", help="Output directory (default: alongside CSV)")
    args = parser.parse_args()

    # Classify positional CSVs
    for csv_path in args.csvs:
        low = csv_path.lower()
        if "chunk" in low and not args.chunks_csv:
            args.chunks_csv = csv_path
        elif "timestep" in low and not args.timesteps_csv:
            args.timesteps_csv = csv_path
        elif not args.agg_csv:
            args.agg_csv = csv_path

    # Auto-detect
    agg_csv = args.agg_csv or find_csv(DEFAULT_AGG)
    chunks_csv = args.chunks_csv
    tsteps_csv = args.timesteps_csv

    # Resolve output dir
    out_dir = args.output_dir
    if not out_dir and agg_csv and os.path.exists(agg_csv):
        out_dir = os.path.dirname(os.path.abspath(agg_csv))
    if not out_dir:
        out_dir = os.path.join(SCRIPT_DIR, "results")

    # Auto-detect chunks/timesteps alongside aggregate
    if not chunks_csv:
        chunks_csv = find_csv(
            [os.path.join(out_dir, "synthetic_workloads_chunks.csv")] + DEFAULT_CHUNKS)
    if not tsteps_csv:
        tsteps_csv = find_csv(
            [os.path.join(out_dir, "synthetic_workloads_timesteps.csv")] + DEFAULT_TIMESTEPS)

    found_any = False

    if agg_csv and os.path.exists(agg_csv):
        found_any = True
        print(f"Loading aggregate: {agg_csv}")
        make_summary_figure(parse_csv(agg_csv),
            os.path.join(out_dir, "synthetic_workloads_summary.png"))

    if chunks_csv and os.path.exists(chunks_csv):
        found_any = True
        print(f"Loading chunk data: {chunks_csv}")
        try:
            make_pattern_heatmap(chunks_csv,
                os.path.join(out_dir, "synthetic_pattern_config_heatmap.png"))
        except Exception as e:
            print(f"  Warning: pattern heatmap failed: {e}")
        try:
            make_feature_scatter(chunks_csv,
                os.path.join(out_dir, "synthetic_feature_scatter.png"))
        except Exception as e:
            print(f"  Warning: feature scatter failed: {e}")
        try:
            make_chunk_timeline(chunks_csv,
                os.path.join(out_dir, "synthetic_chunk_timeline.png"))
        except Exception as e:
            print(f"  Warning: chunk timeline failed: {e}")
        try:
            make_all_timesteps_detail(chunks_csv,
                os.path.join(out_dir, "synthetic_all_timesteps_detail.png"))
        except Exception as e:
            print(f"  Warning: all-timesteps detail failed: {e}")

    if tsteps_csv and os.path.exists(tsteps_csv):
        found_any = True
        print(f"Loading timesteps: {tsteps_csv}")
        make_timestep_figure(tsteps_csv,
            os.path.join(out_dir, "synthetic_workloads_timesteps.png"))

    if not found_any:
        print("ERROR: No synthetic benchmark CSV files found.")
        print("Expected locations:")
        for p in DEFAULT_AGG:
            print(f"  {p}")
        print("\nRun the benchmark first, or specify paths explicitly.")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
