#!/usr/bin/env python3
"""
Generate all figures for a single dataset + policy combination.

Produces up to 5 figures:
  1_summary.png            - 4-panel: ratio, write, read, Pareto (all policies)
  2_latency_breakdown.png  - stacked bar: comp, NN, stats, SGD (balanced only)
  3_algorithm_evolution.png - heatmap: algo per chunk over time (all policies)
  4_predicted_vs_actual.png - per-chunk accuracy at milestones (balanced only)
  5_learning_dynamics.png  - MAPE convergence + firing rates (balanced only)

Usage:
  python3 generate_dataset_figures.py --dataset hurricane_isabel
  python3 generate_dataset_figures.py --dataset gray_scott --policy balanced_w1-1-1

  # All datasets, all policies:
  python3 generate_dataset_figures.py --all
"""
import argparse
import os
import sys
import glob

from config import *
import visualize as viz

def generate_figures(dataset, policy, out_dir):
    """Generate all applicable figures for one dataset+policy."""
    ensure_dir(out_dir)
    data_dir = get_data_dir(dataset, policy)

    if not os.path.isdir(data_dir):
        print(f"  SKIP {dataset}/{policy}: {data_dir} not found")
        return 0

    count = 0
    is_balanced = (policy == BALANCED)

    # Find aggregate CSV (try exact name first, then glob for partial match)
    agg_csv = ""
    if dataset in DATASETS:
        agg_csv = os.path.join(data_dir, DATASETS[dataset]["agg_csv"])
    if not os.path.exists(agg_csv):
        # Search for any matching CSV (handles long SDRBench names like SDRBENCH-EXASKY-NYX-...)
        all_csvs = glob.glob(os.path.join(data_dir, "benchmark_*.csv"))
        all_csvs = [c for c in all_csvs if "_chunks" not in c and "_timesteps" not in c]
        ds_lower = dataset.lower().replace("_", "")
        for c in all_csvs:
            bn = os.path.basename(c).lower().replace("-", "").replace("_", "")
            if ds_lower in bn:
                agg_csv = c
                break
        if not agg_csv and all_csvs:
            agg_csv = all_csvs[0]  # fallback: use first available

    # Derive base name from the aggregate CSV for finding related CSVs
    csv_base = ""
    if agg_csv and os.path.exists(agg_csv):
        csv_base = os.path.basename(agg_csv).replace(".csv", "")

    # ── 1. Summary (all policies) ──
    if os.path.exists(agg_csv):
        rows = viz.parse_csv(agg_csv)
        if rows:
            out = os.path.join(out_dir, "1_summary.png")
            display = dataset.replace("_", " ").title()

            # Build descriptive metadata subtitle
            r0 = rows[0]
            meta_parts = []
            # Dataset size
            orig = viz.g(r0, "orig_mib", "orig_bytes")
            if orig > 1024 * 1024:  # bytes
                meta_parts.append(f"Dataset: {orig / (1024*1024):.0f} MiB")
            elif orig > 0:
                meta_parts.append(f"Dataset: {orig:.0f} MiB")
            # Chunks
            n_ch = int(viz.g(r0, "n_chunks"))
            if n_ch > 0 and orig > 0:
                chunk_mib = (orig if orig < 1024 else orig / (1024*1024)) / max(n_ch, 1)
                meta_parts.append(f"Chunks: {n_ch} x {chunk_mib:.1f} MiB")
            # Runs
            n_runs = int(viz.g(r0, "n_runs"))
            if n_runs > 1:
                meta_parts.append(f"Runs: {n_runs}")
            # Policy
            policy_label = policy.replace("_w", " (w").replace("-", "=") + ")"
            if "balanced" in policy:
                policy_label = "Balanced (w0=1, w1=1, w2=1)"
            elif "ratio" in policy:
                policy_label = "Ratio-only (w0=0, w1=0, w2=1)"
            elif "speed" in policy:
                policy_label = "Speed-only (w0=1, w1=1, w2=0)"
            meta_parts.append(f"Policy: {policy_label}")
            # GS-specific
            L = viz.g(r0, "L")
            if L > 0:
                meta_parts.append(f"Grid: {int(L)}^3")
            steps = viz.g(r0, "steps")
            if steps > 0:
                meta_parts.append(f"Steps: {int(steps)}")

            meta_text = " | ".join(meta_parts)
            viz.make_summary_figure(display, rows, out, meta_text)
            count += 1

    # ── 2. Latency breakdown (balanced only) ──
    if os.path.exists(agg_csv):
        rows = viz.parse_csv(agg_csv)
        if rows:
            out = os.path.join(out_dir, "2_latency_breakdown.png")
            viz.make_latency_breakdown_figure(rows, out, dataset)
            count += 1

    # ── 3. Algorithm evolution heatmap (all policies) ──
    if csv_base:
        tc_csv = os.path.join(data_dir, f"{csv_base}_timestep_chunks.csv")
        ch_csv = os.path.join(data_dir, f"{csv_base}_chunks.csv")
    elif dataset in DATASETS:
        tc_csv = os.path.join(data_dir, DATASETS[dataset]["timestep_chunks_csv"])
        ch_csv = os.path.join(data_dir, DATASETS[dataset]["chunks_csv"])
    else:
        tc_csv = os.path.join(data_dir, f"benchmark_{dataset}_timestep_chunks.csv")
        ch_csv = os.path.join(data_dir, f"benchmark_{dataset}_chunks.csv")

    if os.path.exists(tc_csv):
        out = os.path.join(out_dir, "3_algorithm_evolution.png")
        viz.make_milestone_actions_figure(tc_csv, out,
                                          ch_csv if os.path.exists(ch_csv) else None)
        count += 1

    # ── 4. Predicted vs actual (balanced only) ──
    if os.path.exists(tc_csv):
        out = os.path.join(out_dir, "4_predicted_vs_actual.png")
        viz.make_timestep_chunks_figure(tc_csv, out)
        count += 1

    # ── 5. Learning dynamics: MAPE + firing rates (balanced only) ──
    if csv_base:
        ts_csv = os.path.join(data_dir, f"{csv_base}_timesteps.csv")
    elif dataset in DATASETS:
        ts_csv = os.path.join(data_dir, DATASETS[dataset]["timesteps_csv"])
    else:
        ts_csv = os.path.join(data_dir, f"benchmark_{dataset}_timesteps.csv")

    if os.path.exists(ts_csv):
        out_mape = os.path.join(out_dir, "5a_sgd_convergence.png")
        viz.make_timestep_figure(ts_csv, out_mape)
        count += 1

        out_firing = os.path.join(out_dir, "5b_sgd_exploration_firing.png")
        viz.make_sgd_exploration_figure(ts_csv, out_firing)
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Generate per-dataset benchmark figures")
    parser.add_argument("--dataset", help="Dataset name (e.g., hurricane_isabel, gray_scott)")
    parser.add_argument("--policy", default=None,
                        help="Cost model policy (default: all applicable)")
    parser.add_argument("--all", action="store_true", help="Generate for all datasets")
    args = parser.parse_args()

    if not args.all and not args.dataset:
        parser.error("Specify --dataset NAME or --all")

    # Determine datasets to process
    if args.all:
        datasets = list(DATASETS.keys()) + list(SDR_DATASETS.keys())
    else:
        datasets = [args.dataset]

    total = 0
    for ds in datasets:
        # Determine policies to run
        if args.policy:
            policies = [args.policy]
        else:
            policies_to_run = []
            for pol in ALL_POLICIES:
                d = get_data_dir(ds, pol)
                if os.path.isdir(d):
                    policies_to_run.append(pol)
            if not policies_to_run:
                policies_to_run = [BALANCED]
            policies = policies_to_run

        for pol in policies:
            out_dir = os.path.join(PER_DATASET, ds, pol)
            print(f"\n{'='*60}")
            print(f"  {ds} / {pol}")
            print(f"{'='*60}")
            n = generate_figures(ds, pol, out_dir)
            total += n
            print(f"  Generated {n} figures in {out_dir}")

    print(f"\nTotal: {total} figures in {PER_DATASET}/")


if __name__ == "__main__":
    main()
