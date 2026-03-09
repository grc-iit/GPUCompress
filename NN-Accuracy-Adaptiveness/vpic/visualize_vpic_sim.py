#!/usr/bin/env python3
"""
VPIC Simulation Benchmark — Executive Adaptation Figures.

Produces 10 paper-quality figures demonstrating online SGD adaptation.

Reads:
  sim_chunk_metrics.csv    — per-chunk per-timestep predicted/actual ratios
  sim_timestep_metrics.csv — per-timestep aggregated metrics
  sim_upper_bound.csv      — oracle per-chunk best config/ratio
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


# -- CSV parsing ---------------------------------------------------------------

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


# -- Helpers -------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(_SCRIPT_DIR, "results")
STABLE_MEDIAN_MAPE = 200  # chunks with median APE < 200% are "stable"


def _build_chunk_mape(chunk_rows, phase):
    """Return {chunk_id: [(timestep, mape), ...]} for a given phase."""
    series = {}
    for r in chunk_rows:
        if r.get("phase") != phase:
            continue
        pred = g(r, "predicted_ratio")
        actual = g(r, "actual_ratio")
        if actual <= 0:
            continue
        ts = int(g(r, "timestep"))
        ci = int(g(r, "chunk"))
        if pred > 0:
            ape = abs(pred - actual) / actual * 100.0
        else:
            ape = 0.0  # no prediction available
        series.setdefault(ci, []).append((ts, ape))
    for ci in series:
        series[ci].sort()
    return series


def _get_timesteps(series):
    return sorted(set(ts for ci in series for ts, _ in series[ci]))


def _mape_at_timestep(series, chunk_ids, ts):
    vals = []
    for ci in chunk_ids:
        for t, m in series.get(ci, []):
            if t == ts:
                vals.append(m)
    return vals


def _classify_stable(series):
    stable, unstable = [], []
    for ci, data in series.items():
        median_mape = np.median([m for _, m in data])
        if median_mape < STABLE_MEDIAN_MAPE:
            stable.append(ci)
        else:
            unstable.append(ci)
    return sorted(stable), sorted(unstable)


def _pct_fmt(val, _):
    if val >= 1e6:
        return f"{val / 1e6:.0f}M%"
    if val >= 1e3:
        return f"{val / 1e3:.1f}K%"
    return f"{val:.0f}%"


# -- Algorithm config decoding ------------------------------------------------

ALGO_NAMES_BY_ID = {
    0: "lz4", 1: "snappy", 2: "deflate", 3: "gdeflate",
    4: "zstd", 5: "ans", 6: "cascaded", 7: "bitcomp",
}

ALGO_COLORS = {
    "lz4": "#e74c3c", "snappy": "#3498db", "deflate": "#2ecc71",
    "gdeflate": "#1abc9c", "zstd": "#9b59b6", "ans": "#e67e22",
    "cascaded": "#f39c12", "bitcomp": "#34495e",
}

PHASE_COLORS = {"oracle": "#e74c3c", "nn_baseline": "#2980b9", "nn_sgd": "#27ae60"}


def _parse_nn_action(action_str):
    """Parse nn_action string from CSV -> (algo_name, tag_string).

    CSV now stores human-readable strings like 'lz4', 'snappy+shuf',
    'zstd+shuf+quant'.
    """
    s = str(action_str).strip()
    algo = s.split("+")[0] if "+" in s else s
    return algo, s


def _config_label(algo, quant, shuffle):
    """Short human-readable config label (for oracle rows)."""
    parts = [algo]
    if quant:
        parts.append("q")
    if shuffle:
        parts.append("s")
    return "+".join(parts)


# -- Figure 1: Average MAPE Over Simulation Time ------------------------------

def fig1_avg_mape(chunk_rows, output_dir):
    """Average MAPE for Oracle, Baseline, SGD over timesteps."""
    sgd = _build_chunk_mape(chunk_rows, "nn_sgd")
    bl = _build_chunk_mape(chunk_rows, "nn_baseline")

    if not sgd:
        return

    timesteps = _get_timesteps(sgd)
    all_chunks = sorted(sgd.keys())
    n = len(all_chunks)

    fig, ax = plt.subplots(figsize=(14, 6))

    # Oracle: MAPE = 0
    ax.axhline(y=0, color="#e74c3c", linewidth=2.5, linestyle="-", alpha=0.8,
               label="Oracle (MAPE = 0%)", zorder=2)

    # Baseline
    bl_avgs = [np.mean(_mape_at_timestep(bl, all_chunks, ts))
               for ts in timesteps]
    ax.plot(timesteps, bl_avgs, color="#2980b9", linewidth=2.5,
            marker="s", markersize=5, label="NN Baseline (no learning)",
            zorder=3, alpha=0.9)
    for i, (ts, val) in enumerate(zip(timesteps, bl_avgs)):
        va = "bottom" if i % 2 == 0 else "top"
        offset = 8 if i % 2 == 0 else -8
        ax.annotate(f"{val:.0f}%", (ts, val), fontsize=6,
                    color="#1f618d", fontweight="bold", ha="center", va=va,
                    textcoords="offset points", xytext=(0, offset))

    # SGD
    sgd_avgs = [np.mean(_mape_at_timestep(sgd, all_chunks, ts))
                for ts in timesteps]
    ax.plot(timesteps, sgd_avgs, color="#27ae60", linewidth=2.5,
            marker="^", markersize=5, label="NN + SGD (online learning)",
            zorder=4, alpha=0.9)
    for i, (ts, val) in enumerate(zip(timesteps, sgd_avgs)):
        va = "top" if i % 2 == 0 else "bottom"
        offset = -8 if i % 2 == 0 else 8
        ax.annotate(f"{val:.0f}%", (ts, val), fontsize=6,
                    color="#1a7a3a", fontweight="bold", ha="center", va=va,
                    textcoords="offset points", xytext=(0, offset))

    ax.set_xlabel("Simulation Timestep", fontsize=12)
    ax.set_ylabel("Average Prediction Error (MAPE %)", fontsize=12)
    fig.suptitle("Online SGD Reduces Prediction Error Over Simulation Time",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.text(0.5, 1.01,
             f"Average MAPE across {n} chunks per timestep on live VPIC Harris sheet simulation.\n"
             "Shows how online SGD progressively reduces NN prediction error vs static baseline.",
             ha="center", fontsize=8.5, color="#555", va="top", style="italic")
    ax.set_title("")
    ax.legend(fontsize=10, loc="upper right", framealpha=0.95)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    path = os.path.join(output_dir, "fig1_avg_mape.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Figure 2: All Chunks vs Stable Chunks ------------------------------------

def fig2_all_vs_stable(chunk_rows, output_dir):
    """Two panels with average MAPE and per-point annotations."""
    sgd = _build_chunk_mape(chunk_rows, "nn_sgd")
    bl = _build_chunk_mape(chunk_rows, "nn_baseline")

    if not sgd:
        return

    timesteps = _get_timesteps(sgd)
    all_chunks = sorted(sgd.keys())
    stable, unstable = _classify_stable(sgd)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    panels = [
        (axes[0], all_chunks, f"All {len(all_chunks)} Chunks"),
        (axes[1], stable, f"{len(stable)} Stable Chunks "
                          f"(excluding boundary chunks {unstable})"),
    ]

    for ax, chunk_set, title in panels:
        if not chunk_set:
            ax.text(0.5, 0.5, "No chunks in this category",
                    transform=ax.transAxes, ha="center", va="center")
            continue

        # Baseline
        bl_avgs = [np.mean(_mape_at_timestep(bl, chunk_set, ts))
                   for ts in timesteps]
        ax.plot(timesteps, bl_avgs, color="#2980b9", linewidth=2.5,
                marker="s", markersize=4, label="NN Baseline", zorder=3)

        # SGD
        sgd_avgs = [np.mean(_mape_at_timestep(sgd, chunk_set, ts))
                    for ts in timesteps]
        ax.plot(timesteps, sgd_avgs, color="#27ae60", linewidth=2.5,
                marker="^", markersize=4, label="NN + SGD", zorder=4)

        # Oracle
        ax.axhline(y=0, color="#e74c3c", linewidth=1.5, linestyle="--",
                   alpha=0.6, label="Oracle (0%)", zorder=2)

        # Annotate each data point
        for i, ts in enumerate(timesteps):
            va_bl = "bottom" if i % 2 == 0 else "top"
            off_bl = 6 if i % 2 == 0 else -6
            ax.annotate(f"{bl_avgs[i]:.0f}", (ts, bl_avgs[i]), fontsize=5,
                        color="#1f618d", fontweight="bold", ha="center",
                        va=va_bl, textcoords="offset points",
                        xytext=(0, off_bl))

            va_sgd = "top" if i % 2 == 0 else "bottom"
            off_sgd = -6 if i % 2 == 0 else 6
            ax.annotate(f"{sgd_avgs[i]:.0f}", (ts, sgd_avgs[i]), fontsize=5,
                        color="#1a7a3a", fontweight="bold", ha="center",
                        va=va_sgd, textcoords="offset points",
                        xytext=(0, off_sgd))

        ax.set_ylabel("Average MAPE (%)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9, loc="upper right", framealpha=0.95)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
        ax.grid(True, alpha=0.3, which="both")

    axes[-1].set_xlabel("Simulation Timestep", fontsize=12)
    fig.suptitle("Model Adaptation: All Chunks vs Stable Chunks",
                 fontsize=14, fontweight="bold", y=1.04)
    fig.text(0.5, 1.02,
             "Separates boundary/unstable chunks from stable interior chunks.\n"
             "Shows that SGD improvement is consistent even when excluding noisy boundary chunks.",
             ha="center", fontsize=8.5, color="#555", va="top", style="italic")
    fig.tight_layout()
    path = os.path.join(output_dir, "fig2_all_vs_stable.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Figure 3: Per-Chunk MAPE Reduction (first vs last) -----------------------

def fig3_per_chunk_reduction(chunk_rows, output_dir):
    """Bar chart: first vs last MAPE per chunk."""
    sgd = _build_chunk_mape(chunk_rows, "nn_sgd")

    if not sgd:
        return

    chunk_ids = sorted(sgd.keys())
    n = len(chunk_ids)

    first_mapes, last_mapes = [], []
    for ci in chunk_ids:
        data = sgd[ci]
        first_mapes.append(data[0][1])
        last_mapes.append(data[-1][1])

    fig, ax = plt.subplots(figsize=(max(10, n * 0.7), 6))

    x = np.arange(n)
    bar_w = 0.38

    ax.bar(x - bar_w / 2, first_mapes, bar_w, color="#e74c3c", alpha=0.85,
           edgecolor="white", linewidth=0.5, label="First diagnostic",
           zorder=3)
    ax.bar(x + bar_w / 2, last_mapes, bar_w, color="#27ae60", alpha=0.85,
           edgecolor="white", linewidth=0.5, label="Last diagnostic",
           zorder=3)

    # Annotate reduction %
    for i, (f_m, l_m) in enumerate(zip(first_mapes, last_mapes)):
        if f_m > 0:
            reduction = (1 - l_m / f_m) * 100
            y_pos = max(f_m, l_m) * 1.15
            ax.text(x[i], y_pos, f"{reduction:.0f}%",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    color="#27ae60" if reduction > 0 else "#e74c3c",
                    zorder=5)

    # Summary stats
    reductions = [(1 - l / f) * 100 for f, l in zip(first_mapes, last_mapes)
                  if f > 0]
    mean_red = np.mean(reductions)
    median_red = np.median(reductions)

    n_improved = sum(1 for r in reductions if r > 0)
    summary = (f"{n_improved}/{n} chunks improved\n"
               f"Mean reduction: {mean_red:.0f}%\n"
               f"Median reduction: {median_red:.0f}%")
    ax.text(0.98, 0.95, summary, transform=ax.transAxes, fontsize=9,
            ha="right", va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#d5f5e3",
                      edgecolor="#27ae60", alpha=0.95))

    ax.set_xlabel("Chunk Index", fontsize=12)
    ax.set_ylabel("MAPE (%)", fontsize=12)
    fig.suptitle("Per-Chunk Prediction Error: Before vs After Online Learning",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.text(0.5, 0.99,
             "MAPE at first vs last diagnostic timestep for each chunk. Percentage labels show\n"
             "error reduction. Demonstrates per-chunk adaptation effectiveness of online SGD.",
             ha="center", fontsize=8.5, color="#555", va="top", style="italic")
    ax.set_title("")
    ax.set_xticks(x)
    ax.set_xticklabels([str(ci) for ci in chunk_ids], fontsize=9)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y", which="both")

    fig.tight_layout()
    path = os.path.join(output_dir, "fig3_per_chunk_reduction.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Figure 4: Compression Ratio Over Simulation Time -------------------------

def fig4_compression_ratio(ts_rows, output_dir):
    """Per-timestep compression ratio for Oracle, Baseline, SGD."""
    fig, ax = plt.subplots(figsize=(14, 6))

    phases = [
        ("oracle",      "#e74c3c", "D", "Oracle (best-static)", 2.5),
        ("nn_baseline", "#2980b9", "s", "NN Baseline",          2.5),
        ("nn_sgd",      "#27ae60", "^", "NN + SGD (online)",    2.5),
    ]

    for phase, color, marker, label, lw in phases:
        data = [r for r in ts_rows if r.get("phase") == phase]
        data.sort(key=lambda r: g(r, "timestep"))
        if not data:
            continue
        ts = [g(r, "timestep") for r in data]
        ratios = [g(r, "ratio") for r in data]

        ax.plot(ts, ratios, marker=marker, color=color, linewidth=lw,
                alpha=0.9, label=label, markersize=5, zorder=3)

        # Annotate each point
        for i, (t, ratio) in enumerate(zip(ts, ratios)):
            va = "bottom" if i % 2 == 0 else "top"
            offset = 7 if i % 2 == 0 else -7
            ax.annotate(f"{ratio:.2f}", (t, ratio), fontsize=5,
                        color=color, fontweight="bold", ha="center",
                        va=va, textcoords="offset points",
                        xytext=(0, offset))

    # Overall averages in text box
    avgs = {}
    for phase, _, _, label, _ in phases:
        vals = [g(r, "ratio") for r in ts_rows if r.get("phase") == phase]
        if vals:
            # Use total_orig / total_compressed for correct aggregate
            orig = [g(r, "orig_mib") for r in ts_rows if r.get("phase") == phase]
            compressed = [g(r, "file_mib") for r in ts_rows if r.get("phase") == phase]
            total_orig = sum(orig)
            total_comp = sum(compressed)
            if total_comp > 0:
                avgs[label] = total_orig / total_comp
            else:
                # Oracle has file_mib=0, use harmonic mean
                avgs[label] = len(vals) / sum(1.0/r for r in vals if r > 0)

    if avgs:
        lines = [f"{k}: {v:.2f}x" for k, v in avgs.items()]
        ax.text(0.02, 0.02, "Overall:\n" + "\n".join(lines),
                transform=ax.transAxes, fontsize=8, va="bottom",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#ccc", alpha=0.95))

    ax.set_xlabel("Simulation Timestep", fontsize=12)
    ax.set_ylabel("Compression Ratio", fontsize=12)
    fig.suptitle("Compression Ratio Over Simulation Time",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.text(0.5, 0.99,
             "Actual compression ratio at each simulation timestep for oracle, NN baseline, and NN + SGD.\n"
             "Shows whether SGD's improved predictions translate to better end-to-end compression.",
             ha="center", fontsize=8.5, color="#555", va="top", style="italic")
    ax.set_title("")
    ax.legend(fontsize=10, loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig4_compression_ratio.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Plot 5: Aggregate Summary Bar Chart --------------------------------------

def plot_summary(ts_rows, output_dir):
    """Aggregate bar chart: Oracle vs Baseline vs SGD compression ratios."""
    phases = [
        ("oracle", "Oracle", PHASE_COLORS["oracle"]),
        ("nn_baseline", "Baseline", PHASE_COLORS["nn_baseline"]),
        ("nn_sgd", "SGD", PHASE_COLORS["nn_sgd"]),
    ]

    means = {}
    for phase, label, _ in phases:
        vals = [g(r, "ratio") for r in ts_rows if r.get("phase") == phase]
        if vals:
            means[label] = np.mean(vals)

    if not means:
        return

    labels = list(means.keys())
    values = [means[l] for l in labels]
    colors = [c for _, l, c in phases if l in means]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, values, color=colors, edgecolor="white",
                  linewidth=1.5, width=0.55, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}x", ha="center", va="bottom", fontsize=12,
                fontweight="bold")

    ax.set_ylabel("Mean Compression Ratio", fontsize=12)
    fig.suptitle("VPIC Simulation: Aggregate Compression Ratio",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.text(0.5, 0.99,
             "Mean compression ratio across all simulation timesteps for each phase.\n"
             "Higher is better. Oracle uses exhaustive per-chunk search.",
             ha="center", fontsize=8.5, color="#555", va="top", style="italic")
    ax.set_title("")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(output_dir, "summary.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Plot 6: Per-Chunk Comparison (Oracle/Baseline/SGD) -----------------------

def plot_per_chunk_comparison(chunk_rows, ub_rows, output_dir):
    """Line plot: Oracle vs Baseline vs SGD per-chunk ratios at last timestep."""
    if not chunk_rows:
        return

    last_ts = int(max(g(r, "timestep") for r in chunk_rows))

    # Oracle per-chunk: best ratio from upper_bound at last timestep
    oracle_by_chunk = {}
    for r in ub_rows:
        if int(g(r, "timestep")) != last_ts:
            continue
        ch = int(g(r, "chunk"))
        ratio = g(r, "ratio")
        if ch not in oracle_by_chunk or ratio > oracle_by_chunk[ch]:
            oracle_by_chunk[ch] = ratio

    # Baseline and SGD from chunk_rows
    bl_by_chunk = {}
    sgd_by_chunk = {}
    for r in chunk_rows:
        if int(g(r, "timestep")) != last_ts:
            continue
        ch = int(g(r, "chunk"))
        ratio = g(r, "actual_ratio")
        if r.get("phase") == "nn_baseline":
            bl_by_chunk[ch] = ratio
        elif r.get("phase") == "nn_sgd":
            sgd_by_chunk[ch] = ratio

    all_chunks = sorted(set(oracle_by_chunk) | set(bl_by_chunk) | set(sgd_by_chunk))
    if not all_chunks:
        return

    fig, ax = plt.subplots(figsize=(max(12, len(all_chunks) * 0.6), 6))

    series = [
        (oracle_by_chunk, PHASE_COLORS["oracle"], "D", "Oracle"),
        (bl_by_chunk, PHASE_COLORS["nn_baseline"], "s", "Baseline"),
        (sgd_by_chunk, PHASE_COLORS["nn_sgd"], "^", "SGD"),
    ]
    for data, color, marker, label in series:
        xs = [c for c in all_chunks if c in data]
        ys = [data[c] for c in xs]
        ax.plot(xs, ys, marker=marker, color=color, linewidth=2,
                markersize=6, label=label, alpha=0.9, zorder=3)

    # Use log scale if range is large
    all_vals = []
    for data, _, _, _ in series:
        all_vals.extend(data.values())
    if all_vals and max(all_vals) / max(min(all_vals), 1e-6) > 20:
        ax.set_yscale("log")

    ax.set_xlabel("Chunk Index", fontsize=12)
    ax.set_ylabel("Compression Ratio", fontsize=12)
    ax.set_xticks(all_chunks)
    ax.set_xticklabels([str(c) for c in all_chunks], fontsize=8)
    fig.suptitle("Per-Chunk Compression Ratio: VPIC Data",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.text(0.5, 0.99,
             f"Oracle, Baseline, and SGD compression ratios per chunk at timestep {last_ts}.\n"
             "Compares how closely NN-chosen configs match oracle per-chunk performance.",
             ha="center", fontsize=8.5, color="#555", va="top", style="italic")
    ax.set_title("")
    ax.legend(fontsize=10, loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    path = os.path.join(output_dir, "per_chunk_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Plot 7: Per-Chunk Config Breakdown (3 panels) ----------------------------

def plot_per_chunk_config(chunk_rows, ub_rows, output_dir):
    """3-panel config breakdown (Oracle/Baseline/SGD) at last timestep."""
    if not chunk_rows:
        return

    last_ts = int(max(g(r, "timestep") for r in chunk_rows))

    # Oracle: best config per chunk from upper_bound
    oracle_configs = {}  # chunk -> (ratio, algo_name)
    for r in ub_rows:
        if int(g(r, "timestep")) != last_ts:
            continue
        ch = int(g(r, "chunk"))
        ratio = g(r, "ratio")
        algo = r.get("algorithm", "unknown")
        if ch not in oracle_configs or ratio > oracle_configs[ch][0]:
            oracle_configs[ch] = (ratio, algo)

    # Baseline & SGD: decode nn_action
    bl_configs = {}  # chunk -> (ratio, algo_name)
    sgd_configs = {}
    for r in chunk_rows:
        if int(g(r, "timestep")) != last_ts:
            continue
        ch = int(g(r, "chunk"))
        ratio = g(r, "actual_ratio")
        algo, _ = _parse_nn_action(r.get("nn_action", ""))
        if r.get("phase") == "nn_baseline":
            bl_configs[ch] = (ratio, algo)
        elif r.get("phase") == "nn_sgd":
            sgd_configs[ch] = (ratio, algo)

    all_chunks = sorted(
        set(oracle_configs) | set(bl_configs) | set(sgd_configs))
    if not all_chunks:
        return

    fig, axes = plt.subplots(3, 1, figsize=(max(12, len(all_chunks) * 0.6), 14),
                             sharex=True)

    panels = [
        (axes[0], oracle_configs, "Oracle (exhaustive search)"),
        (axes[1], bl_configs, "NN Baseline (no learning)"),
        (axes[2], sgd_configs, "NN + SGD (online learning)"),
    ]

    for ax, configs, title in panels:
        xs, ys, colors = [], [], []
        for ch in all_chunks:
            if ch in configs:
                ratio, algo = configs[ch]
                xs.append(ch)
                ys.append(ratio)
                colors.append(ALGO_COLORS.get(algo, "#888888"))

        ax.scatter(xs, ys, c=colors, s=80, edgecolors="white",
                   linewidth=0.8, zorder=3)
        ax.set_ylabel("Compression Ratio", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3, which="both")

    axes[-1].set_xlabel("Chunk Index", fontsize=12)
    axes[-1].set_xticks(all_chunks)
    axes[-1].set_xticklabels([str(c) for c in all_chunks], fontsize=8)

    # Legend for algorithm colors
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=color, markersize=10, label=name)
                       for name, color in ALGO_COLORS.items()]
    axes[0].legend(handles=legend_elements, fontsize=8, loc="upper right",
                   ncol=4, framealpha=0.95)

    fig.suptitle("Per-Chunk Config & Ratio: VPIC Data",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.text(0.5, 1.00,
             f"Algorithm selection per chunk at timestep {last_ts}.\n"
             "Marker color indicates the compression algorithm chosen by each method.",
             ha="center", fontsize=8.5, color="#555", va="top", style="italic")

    fig.tight_layout()
    path = os.path.join(output_dir, "per_chunk_config.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Plot 8: Upper-Bound Configs (side-by-side bars) --------------------------

def plot_upper_bound_configs(chunk_rows, ub_rows, output_dir):
    """Side-by-side bars: Oracle vs Baseline vs SGD per chunk at last timestep."""
    if not chunk_rows:
        return

    last_ts = int(max(g(r, "timestep") for r in chunk_rows))

    # Oracle per-chunk best
    oracle = {}
    oracle_labels = {}
    for r in ub_rows:
        if int(g(r, "timestep")) != last_ts:
            continue
        ch = int(g(r, "chunk"))
        ratio = g(r, "ratio")
        algo = r.get("algorithm", "?")
        if ch not in oracle or ratio > oracle[ch]:
            oracle[ch] = ratio
            shuf = int(g(r, "shuffle"))
            qt = int(g(r, "quant"))
            oracle_labels[ch] = _config_label(algo, qt, shuf)

    bl, sgd = {}, {}
    for r in chunk_rows:
        if int(g(r, "timestep")) != last_ts:
            continue
        ch = int(g(r, "chunk"))
        ratio = g(r, "actual_ratio")
        if r.get("phase") == "nn_baseline":
            bl[ch] = ratio
        elif r.get("phase") == "nn_sgd":
            sgd[ch] = ratio

    all_chunks = sorted(set(oracle) | set(bl) | set(sgd))
    if not all_chunks:
        return
    n = len(all_chunks)

    fig, ax = plt.subplots(figsize=(max(14, n * 1.0), 7))
    x = np.arange(n)
    w = 0.26

    or_vals = [oracle.get(c, 0) for c in all_chunks]
    bl_vals = [bl.get(c, 0) for c in all_chunks]
    sgd_vals = [sgd.get(c, 0) for c in all_chunks]

    ax.bar(x - w, or_vals, w, color=PHASE_COLORS["oracle"], alpha=0.85,
           edgecolor="white", linewidth=0.5, label="Oracle", zorder=3)
    ax.bar(x, bl_vals, w, color=PHASE_COLORS["nn_baseline"], alpha=0.85,
           edgecolor="white", linewidth=0.5, label="Baseline", zorder=3)
    ax.bar(x + w, sgd_vals, w, color=PHASE_COLORS["nn_sgd"], alpha=0.85,
           edgecolor="white", linewidth=0.5, label="SGD", zorder=3)

    # Annotate oracle config labels
    for i, c in enumerate(all_chunks):
        lbl = oracle_labels.get(c, "")
        if lbl:
            ax.text(x[i] - w, or_vals[i] + 0.02, lbl, ha="center",
                    va="bottom", fontsize=5, rotation=45, color="#555")

    ax.set_xlabel("Chunk Index", fontsize=12)
    ax.set_ylabel("Compression Ratio", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in all_chunks], fontsize=8)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Oracle vs Baseline vs SGD: Per Chunk",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.text(0.5, 0.99,
             f"Per-chunk compression ratios at timestep {last_ts} with oracle config annotations.\n"
             "Compares exhaustive search (oracle) against NN-driven selection.",
             ha="center", fontsize=8.5, color="#555", va="top", style="italic")
    ax.set_title("")

    fig.tight_layout()
    path = os.path.join(output_dir, "upper_bound_configs.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Plot 9: Config Cross-Check Heatmap ---------------------------------------

def plot_config_crosscheck(chunk_rows, ub_rows, output_dir):
    """Algorithm selection heatmap at last timestep."""
    if not chunk_rows:
        return

    last_ts = int(max(g(r, "timestep") for r in chunk_rows))

    # Oracle algo per chunk
    oracle_algo = {}
    oracle_best_ratio = {}
    for r in ub_rows:
        if int(g(r, "timestep")) != last_ts:
            continue
        ch = int(g(r, "chunk"))
        ratio = g(r, "ratio")
        algo = r.get("algorithm", "unknown")
        if ch not in oracle_best_ratio or ratio > oracle_best_ratio[ch]:
            oracle_best_ratio[ch] = ratio
            oracle_algo[ch] = algo

    # Baseline and SGD algo per chunk
    bl_algo, sgd_algo = {}, {}
    for r in chunk_rows:
        if int(g(r, "timestep")) != last_ts:
            continue
        ch = int(g(r, "chunk"))
        algo, _ = _parse_nn_action(r.get("nn_action", ""))
        if r.get("phase") == "nn_baseline":
            bl_algo[ch] = algo
        elif r.get("phase") == "nn_sgd":
            sgd_algo[ch] = algo

    all_chunks = sorted(set(oracle_algo) | set(bl_algo) | set(sgd_algo))
    if not all_chunks:
        return

    # Build algo -> numeric id mapping
    algo_list = list(ALGO_NAMES_BY_ID.values())
    algo_to_id = {name: i for i, name in enumerate(algo_list)}

    rows_label = ["Baseline", "SGD", "Oracle"]
    rows_data = [bl_algo, sgd_algo, oracle_algo]

    grid = np.full((3, len(all_chunks)), np.nan)
    for ri, data in enumerate(rows_data):
        for ci, ch in enumerate(all_chunks):
            algo = data.get(ch)
            if algo and algo in algo_to_id:
                grid[ri, ci] = algo_to_id[algo]

    # Custom colormap from algo colors
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap_colors = [ALGO_COLORS.get(a, "#888888") for a in algo_list]
    cmap = ListedColormap(cmap_colors)
    bounds = np.arange(-0.5, len(algo_list) + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(max(12, len(all_chunks) * 0.5), 4))
    im = ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm,
                   interpolation="nearest")

    # Mark mismatches vs oracle with red 'x'
    for ci, ch in enumerate(all_chunks):
        oracle_a = oracle_algo.get(ch)
        for ri, data in enumerate(rows_data[:2]):  # Baseline and SGD only
            if data.get(ch) and oracle_a and data[ch] != oracle_a:
                ax.text(ci, ri, "x", ha="center", va="center",
                        fontsize=12, fontweight="bold", color="red")

    ax.set_xticks(range(len(all_chunks)))
    ax.set_xticklabels([str(c) for c in all_chunks], fontsize=7)
    ax.set_xlabel("Chunk Index", fontsize=11)
    ax.set_yticks(range(3))
    ax.set_yticklabels(rows_label, fontsize=10)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='s', color='w',
                              markerfacecolor=ALGO_COLORS.get(a, "#888"),
                              markersize=10, label=a) for a in algo_list]
    legend_elements.append(Line2D([0], [0], marker='x', color='red',
                                  markersize=10, label='mismatch', linestyle='None'))
    ax.legend(handles=legend_elements, fontsize=7, loc="upper left",
              bbox_to_anchor=(1.01, 1), ncol=1, framealpha=0.95)

    fig.suptitle("Config Cross-Check: Per-Chunk Algorithm Selection",
                 fontsize=14, fontweight="bold", y=1.06)
    fig.text(0.5, 1.02,
             f"Algorithm chosen per chunk at timestep {last_ts}. Red 'x' marks mismatches vs oracle.\n"
             "Rows bottom-to-top: Baseline, SGD, Oracle.",
             ha="center", fontsize=8.5, color="#555", va="top", style="italic")
    ax.set_title("")

    fig.tight_layout()
    path = os.path.join(output_dir, "config_crosscheck.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Plot 10: Chunk MAPE Comparison Heatmap (merged from visualize_chunk_mape) -

def _build_mape_grid(chunk_rows, phase):
    """Return (timesteps, chunks, mape_2d) for a given phase."""
    phase_rows = [r for r in chunk_rows if r.get("phase") == phase]
    if not phase_rows:
        return None, None, None

    timesteps = sorted(set(int(g(r, "timestep")) for r in phase_rows))
    chunks = sorted(set(int(g(r, "chunk")) for r in phase_rows))

    grid = np.full((len(timesteps), len(chunks)), np.nan)
    ts_idx = {t: i for i, t in enumerate(timesteps)}
    ch_idx = {c: i for i, c in enumerate(chunks)}

    for r in phase_rows:
        ts = int(g(r, "timestep"))
        ch = int(g(r, "chunk"))
        pred = g(r, "predicted_ratio")
        actual = g(r, "actual_ratio")
        if pred > 0 and actual > 0:
            mape = abs(pred - actual) / actual * 100.0
            grid[ts_idx[ts], ch_idx[ch]] = mape

    return timesteps, chunks, grid


def plot_chunk_mape_comparison(chunk_rows, output_dir):
    """Side-by-side MAPE heatmap: Baseline vs SGD across all timesteps."""
    ts_bl, ch_bl, grid_bl = _build_mape_grid(chunk_rows, "nn_baseline")
    ts_sgd, ch_sgd, grid_sgd = _build_mape_grid(chunk_rows, "nn_sgd")

    if (grid_bl is None or np.all(np.isnan(grid_bl)) or
            grid_sgd is None or np.all(np.isnan(grid_sgd))):
        return

    vmax = min(max(np.nanpercentile(grid_bl, 95),
                   np.nanpercentile(grid_sgd, 95)), 5000)

    fig, axes = plt.subplots(2, 1, figsize=(14, 14))

    for ax, grid, timesteps, chunks, title in [
        (axes[0], grid_bl, ts_bl, ch_bl, "NN Baseline (no learning)"),
        (axes[1], grid_sgd, ts_sgd, ch_sgd, "NN + SGD (online learning)"),
    ]:
        norm = mcolors.LogNorm(vmin=1, vmax=vmax, clip=True)
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdYlGn_r",
                       norm=norm, interpolation="nearest")

        ax.set_xticks(range(len(chunks)))
        ax.set_xticklabels([str(c) for c in chunks], fontsize=7)
        ax.set_xlabel("Chunk Index", fontsize=10)

        step = max(1, len(timesteps) // 20)
        ax.set_yticks(range(0, len(timesteps), step))
        ax.set_yticklabels([str(timesteps[i]) for i in range(0, len(timesteps), step)],
                           fontsize=7)
        ax.set_ylabel("Simulation Timestep", fontsize=10)
        ax.set_title(title, fontsize=12)

        fig.colorbar(im, ax=ax, orientation="horizontal",
                     pad=0.08, shrink=0.6, label="MAPE (%)")

    fig.suptitle("Per-Chunk Prediction Error: Baseline vs SGD",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.97,
             "Heatmap of MAPE (%) per chunk (x-axis) vs simulation timestep (y-axis).\n"
             "Top: static NN baseline. Bottom: NN + online SGD. Green = low error, Red = high error.",
             ha="center", fontsize=9, color="#555", va="top", style="italic")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, "chunk_mape_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VPIC Simulation — Executive Adaptation Figures")
    parser.add_argument("--output-dir", default=DEFAULT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    chunk_csv = os.path.join(output_dir, "sim_chunk_metrics.csv")
    ts_csv = os.path.join(output_dir, "sim_timestep_metrics.csv")

    if not os.path.exists(chunk_csv):
        print(f"ERROR: {chunk_csv} not found")
        sys.exit(1)

    chunk_rows = parse_csv(chunk_csv)
    print(f"Loaded {len(chunk_rows)} chunk rows from {chunk_csv}")

    ts_rows = parse_csv(ts_csv) if os.path.exists(ts_csv) else []
    if ts_rows:
        print(f"Loaded {len(ts_rows)} timestep rows from {ts_csv}")

    ub_csv = os.path.join(output_dir, "sim_upper_bound.csv")
    ub_rows = parse_csv(ub_csv) if os.path.exists(ub_csv) else []
    if ub_rows:
        print(f"Loaded {len(ub_rows)} upper-bound rows from {ub_csv}")

    # Quick summary
    sgd = _build_chunk_mape(chunk_rows, "nn_sgd")
    timesteps = _get_timesteps(sgd)
    stable, unstable = _classify_stable(sgd)
    print(f"  Chunks: {len(sgd)} total, {len(stable)} stable, "
          f"{len(unstable)} boundary/unstable {unstable}")
    if timesteps:
        print(f"  Timesteps: {len(timesteps)} "
              f"({int(timesteps[0])} to {int(timesteps[-1])})")
    else:
        print("  Timesteps: 0")
    print()

    print("Generating figures...")
    fig1_avg_mape(chunk_rows, output_dir)
    fig2_all_vs_stable(chunk_rows, output_dir)
    fig3_per_chunk_reduction(chunk_rows, output_dir)
    if ts_rows:
        fig4_compression_ratio(ts_rows, output_dir)
        plot_summary(ts_rows, output_dir)
    if ub_rows and chunk_rows:
        plot_per_chunk_comparison(chunk_rows, ub_rows, output_dir)
        plot_per_chunk_config(chunk_rows, ub_rows, output_dir)
        plot_upper_bound_configs(chunk_rows, ub_rows, output_dir)
        plot_config_crosscheck(chunk_rows, ub_rows, output_dir)
    plot_chunk_mape_comparison(chunk_rows, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
