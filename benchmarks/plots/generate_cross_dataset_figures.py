#!/usr/bin/env python3
"""
Generate cross-dataset comparison figures.

Produces figures that compare results across all datasets:
  multi_dataset_comparison.png    - grouped bars: ratio + throughput across datasets
  cross_dataset_convergence.png   - MAPE convergence overlay for all datasets
  preprocessing_impact.png        - shuffle vs no-shuffle comparison
  algorithm_selection.png         - algorithm frequency histogram
  policy_comparison.png           - 3 policies side-by-side

Usage:
  python3 generate_cross_dataset_figures.py
"""
import os
import sys
import glob
import csv

from config import *
import visualize as viz

def main():
    out_dir = ensure_dir(CROSS_DATASET)
    print(f"Generating cross-dataset figures in {out_dir}/\n")
    count = 0

    # ── Collect all aggregate CSVs ──
    all_datasets = []  # [(name, rows)]
    all_chunk_csvs = []
    all_ts_csvs = []   # [(name, path)]

    for ds in DATASETS:
        d = get_data_dir(ds, BALANCED)
        agg = os.path.join(d, DATASETS[ds]["agg_csv"])
        if os.path.exists(agg):
            rows = viz.parse_csv(agg)
            if rows:
                all_datasets.append((ds.replace("_", " ").title(), rows))
                print(f"  Loaded: {ds} ({len(rows)} phases)")

        chunks = os.path.join(d, DATASETS[ds]["chunks_csv"])
        if os.path.exists(chunks):
            all_chunk_csvs.append((ds.replace("_", " ").title(), chunks))

        ts = os.path.join(d, DATASETS[ds]["timesteps_csv"])
        if os.path.exists(ts):
            all_ts_csvs.append((ds.replace("_", " ").title(), ts))

    # SDRBench datasets
    sdr = get_data_dir("hurricane_isabel", BALANCED)
    if os.path.isdir(sdr):
        for f in sorted(glob.glob(os.path.join(sdr, "benchmark_*.csv"))):
            bn = os.path.basename(f)
            if "_chunks" in bn or "_timesteps" in bn:
                if "_chunks" in bn and "_timestep_" not in bn:
                    chunk_name = bn.replace("benchmark_", "").replace("_chunks.csv", "")
                    all_chunk_csvs.append((chunk_name.replace("_", " ").title(), f))
                if "_timesteps" in bn:
                    name = bn.replace("benchmark_", "").replace("_timesteps.csv", "")
                    all_ts_csvs.append((name.replace("_", " ").title(), f))
                continue
            name = bn.replace("benchmark_", "").replace(".csv", "")
            rows = viz.parse_csv(f)
            if rows:
                all_datasets.append((name.replace("_", " ").title(), rows))
                print(f"  Loaded: {name} ({len(rows)} phases)")

    if not all_datasets:
        print("ERROR: No benchmark data found. Run eval scripts first.")
        return

    # ── 1. Multi-dataset comparison ──
    if len(all_datasets) >= 2:
        out = os.path.join(out_dir, "multi_dataset_comparison.png")
        viz.make_multi_dataset_figure(all_datasets, out)
        count += 1

        out = os.path.join(out_dir, "per_dataset_phase_comparison.png")
        viz.make_per_dataset_phase_comparison(all_datasets, out)
        count += 1

    # ── 2. Cross-dataset convergence ──
    if len(all_ts_csvs) >= 2:
        paths = [p for _, p in all_ts_csvs]
        names = [n for n, _ in all_ts_csvs]
        out = os.path.join(out_dir, "cross_dataset_convergence.png")
        viz.make_cross_dataset_convergence_figure(paths, names, out)
        count += 1

    # ── 3. Preprocessing impact (shuffle vs no-shuffle) ──
    # Collect data for all datasets, compare fixed-algo vs fixed-algo+shuf
    preproc_data = []
    for name, rows in all_datasets:
        preproc_data.append((name, rows))
    if preproc_data:
        algo_pairs = [
            ("fixed-lz4", "fixed-lz4+shuf", "LZ4"),
            ("fixed-gdeflate", "fixed-gdeflate+shuf", "GDeflate"),
            ("fixed-zstd", "fixed-zstd+shuf", "Zstd"),
        ]
        has_shuf = False
        for _, rows in preproc_data:
            phases = {r.get("phase", "") for r in rows}
            if any("+shuf" in p for p in phases):
                has_shuf = True
                break
        if has_shuf:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle("Preprocessing Impact: Byte-Shuffle", fontsize=14, fontweight="bold")

            labels = [lbl for _, _, lbl in algo_pairs]
            no_shuf_r, shuf_r, no_shuf_w, shuf_w = [], [], [], []
            for _, _, lbl in algo_pairs:
                ns_ratios, s_ratios, ns_writes, s_writes = [], [], [], []
                for _, rows in preproc_data:
                    pm = {r.get("phase", ""): r for r in rows}
                    ns_phase = f"fixed-{lbl.lower()}"
                    s_phase = f"fixed-{lbl.lower()}+shuf"
                    if ns_phase in pm and s_phase in pm:
                        ns_ratios.append(viz.g(pm[ns_phase], "ratio"))
                        s_ratios.append(viz.g(pm[s_phase], "ratio"))
                        ns_writes.append(viz.g(pm[ns_phase], "write_mibps", "write_mbps"))
                        s_writes.append(viz.g(pm[s_phase], "write_mibps", "write_mbps"))
                no_shuf_r.append(np.mean(ns_ratios) if ns_ratios else 0)
                shuf_r.append(np.mean(s_ratios) if s_ratios else 0)
                no_shuf_w.append(np.mean(ns_writes) if ns_writes else 0)
                shuf_w.append(np.mean(s_writes) if s_writes else 0)

            x = np.arange(len(labels))
            w = 0.35
            axes[0].bar(x - w/2, no_shuf_r, w, label="No Shuffle", color="#2e86c1")
            axes[0].bar(x + w/2, shuf_r, w, label="+Shuffle", color="#85c1e9")
            axes[0].set_ylabel("Compression Ratio")
            axes[0].set_title("Ratio")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(labels)
            axes[0].legend()

            axes[1].bar(x - w/2, no_shuf_w, w, label="No Shuffle", color="#2e86c1")
            axes[1].bar(x + w/2, shuf_w, w, label="+Shuffle", color="#85c1e9")
            axes[1].set_ylabel("Write Throughput (MiB/s)")
            axes[1].set_title("Throughput")
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(labels)
            axes[1].legend()

            fig.tight_layout(rect=[0, 0, 1, 0.94])
            out = os.path.join(out_dir, "preprocessing_impact.png")
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {out}")
            count += 1

    # ── 4. Algorithm selection histogram ──
    if all_chunk_csvs:
        out = os.path.join(out_dir, "algorithm_selection.png")
        viz.make_algorithm_histogram(all_chunk_csvs, out)
        count += 1

    # ── 5. Policy comparison ──
    policy_data = {}
    for pol_label, pol_dir in [("balanced", BALANCED), ("ratio-only", RATIO_ONLY), ("speed-only", SPEED_ONLY)]:
        # Use Hurricane Isabel as representative
        d = get_data_dir("hurricane_isabel", pol_dir)
        for candidate in ["benchmark_hurricane_isabel.csv", "benchmark_100x500x500.csv"]:
            p = os.path.join(d, candidate)
            if os.path.exists(p):
                rows = viz.parse_csv(p)
                if rows:
                    policy_data[pol_label] = rows
                break
    if len(policy_data) >= 2:
        out = os.path.join(out_dir, "policy_comparison.png")
        viz.make_policy_comparison_figure(policy_data, out)
        count += 1

    print(f"\nTotal: {count} cross-dataset figures in {out_dir}/")


if __name__ == "__main__":
    main()
