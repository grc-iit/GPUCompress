#!/usr/bin/env python3
"""
4.2.1 RL Adaptiveness on Unseen Workloads — Plotting

Produces 3 charts:
  1. Regret vs Timestep (line graph, one line per dataset)
  2. Compress Time MAPE vs Timestep (line graph, one line per dataset)
  3. Average MAPE vs Dataset (clustered bar chart with error bars)

Usage:
    python3 benchmarks/Paper_Evaluations/4/4.2.1_plot_rl_adaptiveness.py [RESULTS_DIR]

    RESULTS_DIR defaults to benchmarks/Paper_Evaluations/4/results/rl_adaptiveness/
"""

import sys
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASETS = ["nyx", "hurricane_isabel", "cesm_atm", "cesm_atm_26ts"]
DATASET_LABELS = {"nyx": "NYX", "hurricane_isabel": "Hurricane", "cesm_atm": "CESM-ATM", "cesm_atm_26ts": "CESM-ATM-26ts"}
DATASET_COLORS = {"nyx": "#1f77b4", "hurricane_isabel": "#ff7f0e", "cesm_atm": "#2ca02c", "cesm_atm_26ts": "#d62728"}
DATASET_MARKERS = {"nyx": "o", "hurricane_isabel": "s", "cesm_atm": "^", "cesm_atm_26ts": "D"}


def read_timestep_csv(path):
    """Read per-timestep CSV. Returns list of dicts."""
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def read_ranking_csv(path):
    """Read ranking CSV. Returns list of dicts."""
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def collect_regret_by_timestep(ranking_rows):
    """Aggregate top1_regret per timestep (mean across chunks)."""
    ts_regrets = {}
    for row in ranking_rows:
        try:
            ts = int(row["timestep"])
            r = float(row["top1_regret"])
            if np.isfinite(r) and r > 0:
                ts_regrets.setdefault(ts, []).append(r)
        except (KeyError, ValueError):
            continue
    timesteps = sorted(ts_regrets.keys())
    means = [np.mean(ts_regrets[t]) for t in timesteps]
    return timesteps, means


def collect_mape_by_timestep(timestep_rows):
    """Extract per-timestep MAPE values (all 4 statistics)."""
    ts_list = []
    mape_ratio = []
    mape_comp = []
    mape_decomp = []
    mape_psnr = []
    for row in timestep_rows:
        try:
            t = int(row.get("field_idx", -1))
            mr = float(row.get("mape_ratio", 0))
            mc = float(row.get("mape_comp", 0))
            md = float(row.get("mape_decomp", 0))
            mp = float(row.get("mape_psnr", 0))
            ts_list.append(t)
            mape_ratio.append(mr)
            mape_comp.append(mc)
            mape_decomp.append(md)
            mape_psnr.append(mp)
        except (ValueError, TypeError):
            continue
    return ts_list, mape_ratio, mape_comp, mape_decomp, mape_psnr


def plot_regret_vs_timestep(data, out_path):
    """Chart 1: Regret vs Timestep, one line per dataset."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for ds in DATASETS:
        if ds not in data:
            continue
        ts, regret = data[ds]
        if not ts:
            continue
        ax.plot(ts, regret,
                color=DATASET_COLORS[ds],
                marker=DATASET_MARKERS[ds],
                markersize=5, linewidth=1.5,
                label=DATASET_LABELS[ds])

    ax.axhline(y=1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7,
               label="Optimal (1.0x)")
    ax.set_xlabel("Timestep (field index)", fontsize=11)
    ax.set_ylabel("Selection Regret (1.0x = optimal)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_comp_mape_vs_timestep(data, out_path):
    """Chart 2: Compression Time MAPE vs Timestep, one line per dataset."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for ds in DATASETS:
        if ds not in data:
            continue
        ts, mape_comp = data[ds]
        if not ts:
            continue
        ax.plot(ts, mape_comp,
                color=DATASET_COLORS[ds],
                marker=DATASET_MARKERS[ds],
                markersize=5, linewidth=1.5,
                label=DATASET_LABELS[ds])

    ax.set_xlabel("Timestep (field index)", fontsize=11)
    ax.set_ylabel("Compression Time MAPE (%)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_avg_mape_bar(data, out_path):
    """Chart 3: Average MAPE vs Dataset (clustered bar chart with error bars).
    Shows all 4 prediction statistics: Ratio, Comp Time, Decomp Time, PSNR."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # data[ds] = (ratio_list, comp_list, decomp_list, psnr_list)
    stat_names = ["Ratio", "Comp Time", "Decomp Time", "PSNR"]
    stat_indices = [0, 1, 2, 3]
    stat_colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    n_stats = len(stat_names)
    bar_width = 0.18

    ds_present = [ds for ds in DATASETS if ds in data]
    x = np.arange(len(ds_present))

    for si, (stat, idx, color) in enumerate(zip(stat_names, stat_indices, stat_colors)):
        means = []
        stds = []
        for ds in ds_present:
            vals = data[ds][idx] if idx < len(data[ds]) else []
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)

        # Clamp lower error bar at zero (MAPE can't be negative)
        lower_err = [min(s, m) for s, m in zip(stds, means)]
        offset = (si - (n_stats - 1) / 2) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=[lower_err, stds],
               color=color, label=stat, capsize=3, alpha=0.85,
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in ds_present], fontsize=10)
    ax.set_ylabel("MAPE (%)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_avg_regret_bar(data, out_path):
    """Chart 4: Average Regret per Dataset (bar chart with error bars).
    data: ds -> list of per-chunk regret values."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ds_present = [ds for ds in DATASETS if ds in data]
    x = np.arange(len(ds_present))
    bar_width = 0.5

    means = []
    stds = []
    for ds in ds_present:
        vals = data[ds]
        means.append(np.mean(vals) if vals else 1.0)
        stds.append(np.std(vals) if vals else 0)

    bars = ax.bar(x, means, bar_width, yerr=stds,
                  color="#4c72b0", capsize=4, alpha=0.85,
                  edgecolor="white", linewidth=0.5)

    # Annotate values
    for i, (bar, m) in enumerate(zip(bars, means)):
        offset = stds[i] if stds[i] > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2, m + offset + 0.01,
                f"{m:.2f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(y=1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7,
               label="Optimal (1.0x)")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in ds_present], fontsize=10)
    ax.set_ylabel("Selection Regret (1.0x = optimal)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, "results", "rl_adaptiveness")

    if not os.path.isdir(results_dir):
        print(f"ERROR: results directory not found: {results_dir}")
        sys.exit(1)

    print(f"Reading results from: {results_dir}")

    # ── Collect data ──
    regret_data = {}    # ds -> (timesteps, mean_regrets)
    comp_mape_data = {} # ds -> (timesteps, mape_comp)
    bar_data = {}       # ds -> (ratio_list, comp_list, decomp_list)
    regret_bar_data = {} # ds -> list of all per-chunk regret values

    for ds in DATASETS:
        ds_dir = os.path.join(results_dir, ds)
        ts_path = os.path.join(ds_dir, f"benchmark_{ds}_timesteps.csv")
        rank_path = os.path.join(ds_dir, f"benchmark_{ds}_ranking.csv")

        if not os.path.isdir(ds_dir):
            print(f"  [{ds}] Not found, skipping.")
            continue

        # Regret from ranking CSV
        ranking_rows = read_ranking_csv(rank_path)
        if ranking_rows:
            ts, regrets = collect_regret_by_timestep(ranking_rows)
            regret_data[ds] = (ts, regrets)
            # Collect all per-chunk regret values for bar chart
            all_regrets = []
            for row in ranking_rows:
                try:
                    r = float(row["top1_regret"])
                    if np.isfinite(r) and r > 0:
                        all_regrets.append(r)
                except (KeyError, ValueError):
                    continue
            regret_bar_data[ds] = all_regrets
            print(f"  [{ds}] Ranking: {len(ranking_rows)} rows, {len(ts)} milestones")
        else:
            print(f"  [{ds}] No ranking data.")

        # MAPE from timestep CSV
        ts_rows = read_timestep_csv(ts_path)
        if ts_rows:
            ts_list, mr, mc, md, mp = collect_mape_by_timestep(ts_rows)
            comp_mape_data[ds] = (ts_list, mc)
            bar_data[ds] = (mr, mc, md, mp)
            print(f"  [{ds}] Timesteps: {len(ts_rows)} rows")
        else:
            print(f"  [{ds}] No timestep data.")

    if not regret_data and not comp_mape_data:
        print("ERROR: no data found for any dataset.")
        sys.exit(1)

    # ── Generate plots ──
    print(f"\nGenerating plots...")

    plot_regret_vs_timestep(
        regret_data,
        os.path.join(results_dir, "regret_vs_timestep.png"))

    plot_comp_mape_vs_timestep(
        comp_mape_data,
        os.path.join(results_dir, "comp_mape_vs_timestep.png"))

    plot_avg_mape_bar(
        bar_data,
        os.path.join(results_dir, "avg_mape_per_dataset.png"))

    if regret_bar_data:
        plot_avg_regret_bar(
            regret_bar_data,
            os.path.join(results_dir, "avg_regret_per_dataset.png"))

    print(f"\nAll plots saved to: {results_dir}")


if __name__ == "__main__":
    main()
