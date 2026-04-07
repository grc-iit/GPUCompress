#!/usr/bin/env python3
"""
sc26_combined_plots.py

Reads each workload's per-chunk + ranking CSVs from an SC26 sweep directory
and produces THREE combined figures per policy, with all workloads on the
same plot:

  Figure 1: cost model MAPE convergence — line graph, all workloads
  Figure 2: top-1 regret convergence    — line graph, all workloads
  Figure 3: per-metric MAPE breakdown   — grouped bar chart with comp time,
                                          decomp time, comp ratio, and PSNR
                                          MAPE for each workload

Outputs:
  $SC26_DIR/figures/sc26_<policy>_cost_mape.png
  $SC26_DIR/figures/sc26_<policy>_regret.png
  $SC26_DIR/figures/sc26_<policy>_metric_breakdown.png
  $SC26_DIR/csv/sc26_<policy>_cost_mape.csv
  $SC26_DIR/csv/sc26_<policy>_regret.csv
  $SC26_DIR/csv/sc26_<policy>_metric_breakdown.csv

Unit handling:
  - cost_model_error_pct is stored as a FRACTION in [0,1] by
    src/api/gpucompress_compress.cpp:545 — multiplied by 100 here.
  - mape_comp / mape_decomp / mape_ratio / mape_psnr are written as
    PERCENT (0-200+) by the live-sim emitters (e.g. vpic_benchmark_deck.cxx,
    fix_gpucompress_kokkos.cpp, FlushFormatGPUCompress.cpp). No scaling.
  - top1_regret is written as a RATIO (predicted_best_cost / actual_best_cost,
    where 1.0 = oracle pick). Converted to a percent here as (ratio - 1) * 100,
    matching the in-runner plotter at 7.1_run_equalized_cross_workload_regret.sh:646.
"""
import argparse
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORKLOADS = [
    ("vpic",   "VPIC",     "#e41a1c"),
    ("nyx",    "NYX",      "#377eb8"),
    ("warpx",  "WarpX",    "#4daf4a"),
    ("lammps", "LAMMPS",   "#ff7f00"),
    ("ai",     "ViT-B/16", "#984ea3"),
]

METRIC_DEFS = [
    ("comp",   "mape_comp",   "Comp Time MAPE",  "#1f77b4"),
    ("decomp", "mape_decomp", "Decomp Time MAPE", "#ff7f0e"),
    ("ratio",  "mape_ratio",  "Comp Ratio MAPE",  "#2ca02c"),
    ("psnr",   "mape_psnr",   "PSNR MAPE",        "#d62728"),
]


def chunks_csv(rdir, key):
    p = os.path.join(rdir, key, f"benchmark_{key}_timestep_chunks.csv")
    if os.path.isfile(p):
        return p
    if key == "vpic":
        p = os.path.join(rdir, "vpic", "benchmark_vpic_deck_timestep_chunks.csv")
        if os.path.isfile(p):
            return p
    return None


def ranking_csv(rdir, key):
    p = os.path.join(rdir, key, f"benchmark_{key}_ranking.csv")
    if os.path.isfile(p):
        return p
    if key == "vpic":
        p = os.path.join(rdir, "vpic", "benchmark_vpic_deck_ranking.csv")
        if os.path.isfile(p):
            return p
    return None


def load_column(path, col, scale=1.0, drop_nan=True):
    """Return a list of float(col)*scale for each row where the cell is
    parseable. NaNs are dropped by default."""
    out = []
    if not path or not os.path.isfile(path):
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            v = row.get(col, "")
            if v in ("", None):
                continue
            try:
                fv = float(v)
            except ValueError:
                continue
            if drop_nan and math.isnan(fv):
                continue
            out.append(fv * scale)
    return out


def smooth(ys, window=None):
    if not ys:
        return [], []
    if window is None:
        window = min(10, max(1, len(ys) // 6))
    if len(ys) < window or window <= 1:
        return list(range(len(ys))), list(ys)
    sm = np.convolve(ys, np.ones(window) / window, mode="valid").tolist()
    xs = list(range(window - 1, len(ys)))
    return xs, sm


def final_mean(vals, frac=0.2):
    if not vals:
        return None
    n = max(1, int(round(len(vals) * frac)))
    return float(np.mean(vals[-n:]))


def plot_line(series, title, ylabel, png_path, csv_path, cap=None):
    """series = list of (label, color, ys)"""
    if not series:
        print(f"  no data → skipping {os.path.basename(png_path)}")
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    rows = []
    for label, color, ys in series:
        ys_capped = [min(y, cap) for y in ys] if cap is not None else list(ys)
        xs = list(range(len(ys_capped)))
        ax.plot(xs, ys_capped, color=color, alpha=0.18, linewidth=0.6)
        smx, smy = smooth(ys_capped)
        if smx and len(smx) > 1:
            ax.plot(smx, smy, color=color, linewidth=2.0, label=label)
        else:
            ax.plot(xs, ys_capped, color=color, linewidth=2.0, label=label)
        for i, y in enumerate(ys):
            rows.append((label, i, y))
    ax.set_xlabel("Chunk Index (sequential across profiled fields)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=9)
    if cap is not None:
        ax.set_ylim(bottom=0, top=cap * 1.05)
    else:
        ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workload", "chunk_index", "value_pct"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.6f}"])
    print(f"  wrote {png_path}")
    print(f"  wrote {csv_path}")


def plot_bar(workload_metrics, title, png_path, csv_path, cap=None):
    """workload_metrics = list of (label, color, {metric_key: value or None})"""
    if not workload_metrics:
        print(f"  no data → skipping {os.path.basename(png_path)}")
        return
    n_w = len(workload_metrics)
    n_m = len(METRIC_DEFS)
    bar_w = 0.8 / n_m
    x = np.arange(n_w)

    fig, ax = plt.subplots(figsize=(max(6.5, 1.6 * n_w + 2.0), 4.2))
    rows = []
    for mi, (mkey, _src, mlabel, mcolor) in enumerate(METRIC_DEFS):
        vals = []
        for label, _color, metrics in workload_metrics:
            v = metrics.get(mkey)
            vals.append(v if (v is not None and not math.isnan(v)) else 0.0)
            rows.append((label, mlabel,
                         "" if v is None or (isinstance(v, float) and math.isnan(v))
                         else f"{v:.6f}"))
        offset = mi * bar_w - 0.4 + bar_w / 2
        bars = ax.bar(x + offset, vals, width=bar_w, label=mlabel, color=mcolor)
        # annotate non-zero bars (or "n/a" for missing)
        for bi, (b, raw) in enumerate(zip(bars,
                [workload_metrics[wi][2].get(mkey) for wi in range(n_w)])):
            if raw is None or (isinstance(raw, float) and math.isnan(raw)):
                ax.text(b.get_x() + b.get_width() / 2, 0.5,
                        "n/a", ha="center", va="bottom",
                        fontsize=7, color="gray", rotation=90)
            else:
                ax.text(b.get_x() + b.get_width() / 2,
                        min(b.get_height(), cap if cap else b.get_height()) + 0.5,
                        f"{raw:.0f}", ha="center", va="bottom",
                        fontsize=7, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels([wm[0] for wm in workload_metrics])
    ax.set_ylabel("MAPE (%) — mean over final 20% of chunks")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(frameon=True, fontsize=9, ncol=n_m, loc="upper right")
    if cap is not None:
        ax.set_ylim(bottom=0, top=cap * 1.05)
    else:
        ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workload", "metric", "value_pct"])
        for r in rows:
            w.writerow(r)
    print(f"  wrote {png_path}")
    print(f"  wrote {csv_path}")


def build_for_policy(sc26_dir, policy):
    fig_dir = os.path.join(sc26_dir, "figures")
    csv_dir = os.path.join(sc26_dir, "csv")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    cost_series = []
    regret_series = []
    bar_data = []
    any_workload = False

    for key, label, color in WORKLOADS:
        rdir = os.path.join(sc26_dir, f"{key}_{policy}")
        if not os.path.isdir(rdir):
            continue
        any_workload = True
        cc = chunks_csv(rdir, key)
        rc = ranking_csv(rdir, key)

        cost_vals = load_column(cc, "cost_model_error_pct", scale=100.0)
        if cost_vals:
            cost_series.append((label, color, cost_vals))

        # top1_regret is a RATIO; convert to percent overhead vs oracle.
        reg_ratios = load_column(rc, "top1_regret")
        reg_vals = [(r - 1.0) * 100.0 for r in reg_ratios]
        if reg_vals:
            regret_series.append((label, color, reg_vals))

        metrics = {}
        for mkey, src, _l, _c in METRIC_DEFS:
            vals = load_column(cc, src)
            metrics[mkey] = final_mean(vals)
        bar_data.append((label, color, metrics))

    if not any_workload:
        print(f"=== Policy {policy}: no workload subdirs found, skipping")
        return

    print(f"\n=== Policy: {policy} ===")
    plot_line(
        cost_series,
        f"Figure 1 — Cost Model MAPE Convergence ({policy})",
        "Cost Model MAPE (%)",
        os.path.join(fig_dir, f"sc26_{policy}_cost_mape.png"),
        os.path.join(csv_dir, f"sc26_{policy}_cost_mape.csv"),
        cap=100.0,
    )
    plot_line(
        regret_series,
        f"Figure 2 — Top-1 Regret Convergence ({policy})",
        "Top-1 Regret (%)",
        os.path.join(fig_dir, f"sc26_{policy}_regret.png"),
        os.path.join(csv_dir, f"sc26_{policy}_regret.csv"),
        cap=100.0,
    )
    plot_bar(
        bar_data,
        f"Figure 3 — Per-Metric MAPE Breakdown ({policy})",
        os.path.join(fig_dir, f"sc26_{policy}_metric_breakdown.png"),
        os.path.join(csv_dir, f"sc26_{policy}_metric_breakdown.csv"),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sc26-dir", required=True,
                    help="SC26 sweep root containing <workload>_<policy>/ dirs")
    ap.add_argument("--policies", nargs="+", default=["balanced", "ratio"])
    args = ap.parse_args()

    if not os.path.isdir(args.sc26_dir):
        print(f"ERROR: {args.sc26_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    for pol in args.policies:
        build_for_policy(args.sc26_dir, pol)


if __name__ == "__main__":
    main()
