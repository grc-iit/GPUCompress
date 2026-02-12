#!/usr/bin/env python3
"""
Visualization 1: Aggregate Algorithm Comparison

Six separate figures comparing algorithms across all files,
showing shuffle=0 vs shuffle=4 side by side.

Usage:
    python3 scripts/viz_algo_comparison.py benchmark_results.csv
    python3 scripts/viz_algo_comparison.py benchmark_results.csv -o plots/
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ALGO_ORDER = ["lz4", "snappy", "deflate", "gdeflate", "zstd", "ans", "cascaded", "bitcomp"]
PALETTE = {
    "lz4": "#1f77b4", "snappy": "#ff7f0e", "deflate": "#2ca02c", "gdeflate": "#d62728",
    "zstd": "#9467bd", "ans": "#8c564b", "cascaded": "#e377c2", "bitcomp": "#7f7f7f",
}
SHUF_STYLES = {0: {"label": "No Shuffle", "hatch": None, "alpha": 0.75},
               4: {"label": "Shuffle(4)", "hatch": "//", "alpha": 0.55}}


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["success"] == True].copy()
    df["psnr_val"] = pd.to_numeric(df["psnr_db"], errors="coerce")
    return df


def get_algos(df):
    return [a for a in ALGO_ORDER if a in df["algorithm"].unique()]


def paired_boxplot(ax, df_sub, algos, metric, ylabel):
    """Side-by-side box plots for shuffle=0 vs shuffle=4 per algorithm."""
    positions_ns, positions_s = [], []
    data_ns, data_s = [], []
    colors_ns, colors_s = [], []
    tick_positions, tick_labels = [], []

    gap = 0.45
    for i, a in enumerate(algos):
        center = i * 2.5
        positions_ns.append(center - gap / 2)
        positions_s.append(center + gap / 2)
        tick_positions.append(center)
        tick_labels.append(a)

        sub_ns = df_sub[(df_sub["algorithm"] == a) & (df_sub["shuffle"] == 0)]
        sub_s = df_sub[(df_sub["algorithm"] == a) & (df_sub["shuffle"] == 4)]
        data_ns.append(sub_ns[metric].dropna().values)
        data_s.append(sub_s[metric].dropna().values)
        colors_ns.append(PALETTE[a])
        colors_s.append(PALETTE[a])

    bp_ns = ax.boxplot(data_ns, positions=positions_ns, widths=0.4, patch_artist=True,
                       showfliers=False, medianprops=dict(color="black", linewidth=1.5))
    bp_s = ax.boxplot(data_s, positions=positions_s, widths=0.4, patch_artist=True,
                      showfliers=False, medianprops=dict(color="black", linewidth=1.5))

    for patch, c in zip(bp_ns["boxes"], colors_ns):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    for patch, c in zip(bp_s["boxes"], colors_s):
        patch.set_facecolor(c)
        patch.set_alpha(0.45)
        patch.set_hatch("//")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(bottom=0)
    ax.legend(handles=[
        Patch(facecolor="gray", alpha=0.75, label="No Shuffle"),
        Patch(facecolor="gray", alpha=0.45, hatch="//", label="Shuffle(4)"),
    ], fontsize=10)


def save_fig(fig, output_dir, name):
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_compression_ratio(df, algos, output_dir):
    configs = [
        ("none", None, "Lossless"),
        ("linear", 0.001, "Lossy (eb=0.001)"),
        ("linear", 0.01, "Lossy (eb=0.01)"),
        ("linear", 0.1, "Lossy (eb=0.1)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes = axes.flatten()
    fig.suptitle("Compression Ratio — Shuffle Tradeoff: Helps Lossless, Hurts Lossy",
                 fontsize=16, fontweight="bold")

    for i, (qtype, eb, subtitle) in enumerate(configs):
        ax = axes[i]
        if qtype == "none":
            df_sub = df[df["quantization"] == "none"]
        else:
            df_sub = df[(df["quantization"] == "linear") & (np.isclose(df["error_bound"], eb))]
        paired_boxplot(ax, df_sub, algos, "compression_ratio", "Compression Ratio (higher = better)")
        ax.set_title(subtitle, fontsize=13)
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.4)

        # Annotate median values and shuffle speedup
        gap = 0.45
        for j, a in enumerate(algos):
            center = j * 2.5
            pos_ns = center - gap / 2
            pos_s = center + gap / 2
            sub_ns = df_sub[(df_sub["algorithm"] == a) & (df_sub["shuffle"] == 0)]
            sub_s = df_sub[(df_sub["algorithm"] == a) & (df_sub["shuffle"] == 4)]
            med_ns = sub_ns["compression_ratio"].median()
            med_s = sub_s["compression_ratio"].median()

            # Median value labels on each bar
            ax.annotate(f"{med_ns:.1f}", (pos_ns, med_ns), textcoords="offset points",
                        xytext=(0, -12), ha="center", fontsize=7, color="black",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
            ax.annotate(f"{med_s:.1f}", (pos_s, med_s), textcoords="offset points",
                        xytext=(0, -12), ha="center", fontsize=7, color="black",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

            # Speedup label above the pair
            if med_ns > 0 and med_s > 0:
                speedup = med_s / med_ns
                sp_color = "#2e7d32" if speedup >= 1.0 else "#c62828"
                sp_label = f"{speedup:.1f}x" if speedup >= 1.0 else f"{speedup:.2f}x"
                ymax = max(sub_ns["compression_ratio"].quantile(0.75),
                           sub_s["compression_ratio"].quantile(0.75))
                ax.annotate(sp_label, (center, ymax), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=9, fontweight="bold",
                            color=sp_color)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return save_fig(fig, output_dir, "1_compression_ratio.png")


def _throughput_2x2(df, algos, metric, ylabel, title, output_dir, filename):
    configs = [
        ("none", None, "Lossless"),
        ("linear", 0.001, "Lossy (eb=0.001)"),
        ("linear", 0.01, "Lossy (eb=0.01)"),
        ("linear", 0.1, "Lossy (eb=0.1)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes = axes.flatten()
    fig.suptitle(title, fontsize=16, fontweight="bold")

    for i, (qtype, eb, subtitle) in enumerate(configs):
        ax = axes[i]
        if qtype == "none":
            df_sub = df[df["quantization"] == "none"]
        else:
            df_sub = df[(df["quantization"] == "linear") & (np.isclose(df["error_bound"], eb))]
        paired_boxplot(ax, df_sub, algos, metric, ylabel)
        ax.set_title(subtitle, fontsize=13)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return save_fig(fig, output_dir, filename)


def fig_compression_throughput(df, algos, output_dir):
    return _throughput_2x2(df, algos,
                           "compression_throughput_mbps", "Throughput (MB/s)",
                           "Compression Throughput — All Files",
                           output_dir, "2_compression_throughput.png")


def fig_decompression_throughput(df, algos, output_dir):
    return _throughput_2x2(df, algos,
                           "decompression_throughput_mbps", "Throughput (MB/s)",
                           "Decompression Throughput — All Files",
                           output_dir, "3_decompression_throughput.png")


def fig_ratio_by_quantization(df, algos, output_dir):
    fig, ax = plt.subplots(figsize=(15, 7))
    fig.suptitle("Compression Ratio by Quantization Level — All Files", fontsize=15, fontweight="bold")

    quant_levels = [("none", "Lossless"), ("0.001", "eb=0.001"), ("0.01", "eb=0.01"), ("0.1", "eb=0.1")]
    quant_colors = {"Lossless": "#4A90D9", "eb=0.001": "#F5A623", "eb=0.01": "#7ED321", "eb=0.1": "#D0021B"}

    n_algos = len(algos)
    # 8 bars per algo: 4 quant × 2 shuffle, with a small gap between quant pairs
    bar_w = 0.09
    pair_gap = 0.02  # gap between no-shuffle and shuffle within same quant
    group_gap = 0.06  # gap between different quant levels

    for qi, (eb_str, qlabel) in enumerate(quant_levels):
        color = quant_colors[qlabel]
        for si, (shuf, sinfo) in enumerate(SHUF_STYLES.items()):
            if eb_str == "none":
                subset = df[(df["quantization"] == "none") & (df["shuffle"] == shuf)]
            else:
                subset = df[(df["quantization"] == "linear")
                            & (df["error_bound"].astype(str) == eb_str)
                            & (df["shuffle"] == shuf)]
            medians = [subset[subset["algorithm"] == a]["compression_ratio"].median() for a in algos]
            # Position: each quant pair takes (2*bar_w + pair_gap), groups separated by group_gap
            pair_width = 2 * bar_w + pair_gap
            total_width = 4 * pair_width + 3 * group_gap
            start = -total_width / 2
            offset = start + qi * (pair_width + group_gap) + si * (bar_w + pair_gap)

            x = np.arange(n_algos) * 1.2  # spread algorithms further apart
            hatch = "//" if shuf == 4 else None
            edgecolor = "black" if shuf == 4 else None
            ax.bar(x + offset, medians, bar_w, color=color,
                   alpha=0.85 if shuf == 0 else 0.55,
                   hatch=hatch, edgecolor=edgecolor, linewidth=0.5)

    # Legend: one entry per quant level (solid + hatched pair)
    legend_handles = []
    for _, qlabel in quant_levels:
        c = quant_colors[qlabel]
        legend_handles.append(Patch(facecolor=c, alpha=0.85, edgecolor="none", label=f"{qlabel} (no shuffle)"))
        legend_handles.append(Patch(facecolor=c, alpha=0.55, hatch="//", edgecolor="black",
                                    linewidth=0.5, label=f"{qlabel} + Shuffle(4)"))
    ax.legend(handles=legend_handles, fontsize=9, ncol=2, loc="upper left")
    ax.set_ylabel("Median Compression Ratio (log)", fontsize=12)
    ax.set_xticks(np.arange(n_algos) * 1.2)
    ax.set_xticklabels(algos, rotation=30, fontsize=11)
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.5)
    return save_fig(fig, output_dir, "4_ratio_by_quantization.png")


def fig_psnr(df, algos, output_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Reconstruction Quality (PSNR) — Lossy, All Files", fontsize=15, fontweight="bold")

    df_lossy = df[df["quantization"] == "linear"]
    x = np.arange(len(algos))
    eb_levels = [0.001, 0.01, 0.1]
    eb_colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    bar_w = 0.12
    for ei, (eb, ecolor) in enumerate(zip(eb_levels, eb_colors)):
        for si, (shuf, sinfo) in enumerate(SHUF_STYLES.items()):
            subset = df_lossy[(np.isclose(df_lossy["error_bound"], eb)) & (df_lossy["shuffle"] == shuf)]
            medians = []
            for a in algos:
                vals = subset[subset["algorithm"] == a]["psnr_val"].replace([np.inf], np.nan).dropna()
                medians.append(vals.median() if len(vals) > 0 else 0)
            offset = (ei * 2 + si - 2.5) * bar_w
            ax.bar(x + offset, medians, bar_w * 0.9, color=ecolor,
                   alpha=sinfo["alpha"], hatch=sinfo["hatch"])

    legend_handles = [Patch(facecolor=c, alpha=0.85, label=f"eb={eb}") for eb, c in zip(eb_levels, eb_colors)]
    legend_handles.append(Patch(facecolor="gray", alpha=0.45, hatch="//", label="= Shuffle(4)"))
    ax.legend(handles=legend_handles, fontsize=10)
    ax.set_ylabel("Median PSNR (dB)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=30, fontsize=11)
    ax.set_ylim(bottom=0)
    return save_fig(fig, output_dir, "5_psnr.png")


def fig_pareto(df, algos, output_dir):
    configs = [
        ("none", None, "Lossless"),
        ("linear", 0.001, "Lossy (eb=0.001)"),
        ("linear", 0.01, "Lossy (eb=0.01)"),
        ("linear", 0.1, "Lossy (eb=0.1)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes = axes.flatten()
    fig.suptitle("Ratio vs Throughput Tradeoff — All Files", fontsize=16, fontweight="bold")

    for i, (qtype, eb, subtitle) in enumerate(configs):
        ax = axes[i]
        if qtype == "none":
            df_sub = df[df["quantization"] == "none"]
        else:
            df_sub = df[(df["quantization"] == "linear") & (np.isclose(df["error_bound"], eb))]

        for shuf, sinfo in SHUF_STYLES.items():
            df_s = df_sub[df_sub["shuffle"] == shuf]
            marker = "o" if shuf == 0 else "D"
            for a in algos:
                sub = df_s[df_s["algorithm"] == a]
                if sub.empty:
                    continue
                med_ratio = sub["compression_ratio"].median()
                med_tp = sub["compression_throughput_mbps"].median()
                q25_r, q75_r = sub["compression_ratio"].quantile(0.25), sub["compression_ratio"].quantile(0.75)
                q25_t, q75_t = sub["compression_throughput_mbps"].quantile(0.25), sub["compression_throughput_mbps"].quantile(0.75)
                ax.errorbar(med_ratio, med_tp,
                            xerr=[[med_ratio - q25_r], [q75_r - med_ratio]],
                            yerr=[[med_tp - q25_t], [q75_t - med_tp]],
                            fmt=marker, color=PALETTE[a], markersize=10, capsize=3,
                            linewidth=1.2, markeredgecolor="black", markeredgewidth=0.7,
                            alpha=0.9 if shuf == 0 else 0.6)
                xyoff = (8, 6) if shuf == 0 else (8, -10)
                ax.annotate(a if shuf == 0 else "", (med_ratio, med_tp),
                            textcoords="offset points", xytext=xyoff,
                            fontsize=9, fontweight="bold", color=PALETTE[a])

        ax.legend(handles=[
            Line2D([0], [0], marker="o", color="gray", markersize=8, linestyle="", label="No Shuffle"),
            Line2D([0], [0], marker="D", color="gray", markersize=8, linestyle="", alpha=0.6, label="Shuffle(4)"),
        ], fontsize=9)
        ax.set_xlabel("Median Compression Ratio", fontsize=11)
        ax.set_ylabel("Median Comp. Throughput (MB/s)", fontsize=11)
        ax.set_title(subtitle, fontsize=13)
        ax.axvline(1.0, color="red", linestyle="--", alpha=0.3)
        ax.set_xlim(left=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return save_fig(fig, output_dir, "6_pareto_ratio_vs_throughput.png")


def print_summary_table(df):
    df_ll = df[df["quantization"] == "none"]
    header = f"{'Algorithm':<12} {'Shuffle':>7} {'Med Ratio':>10} {'Med Comp MB/s':>14} {'Med Decomp MB/s':>16} {'N':>6}"
    print(f"\n{header}")
    print("-" * len(header))
    for a in ALGO_ORDER:
        for shuf in [0, 4]:
            sub = df_ll[(df_ll["algorithm"] == a) & (df_ll["shuffle"] == shuf)]
            if sub.empty:
                continue
            print(f"{a:<12} {shuf:>7} {sub['compression_ratio'].median():>10.2f} "
                  f"{sub['compression_throughput_mbps'].median():>14.1f} "
                  f"{sub['decompression_throughput_mbps'].median():>16.1f} "
                  f"{len(sub):>6}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate algorithm comparison")
    parser.add_argument("csv", help="Path to benchmark_results.csv")
    parser.add_argument("-o", "--output-dir", default="plots", help="Output directory")
    args = parser.parse_args()

    csv_path = args.csv
    df = load_data(csv_path)
    os.makedirs(args.output_dir, exist_ok=True)

    algos = get_algos(df)
    print(f"Loaded {len(df)} successful rows from {csv_path}")
    print_summary_table(df)

    paths = [
        fig_compression_ratio(df, algos, args.output_dir),
        fig_compression_throughput(df, algos, args.output_dir),
        fig_decompression_throughput(df, algos, args.output_dir),
        fig_ratio_by_quantization(df, algos, args.output_dir),
        fig_psnr(df, algos, args.output_dir),
        fig_pareto(df, algos, args.output_dir),
    ]
    print()
    for p in paths:
        print(f"  Saved: {p}")
