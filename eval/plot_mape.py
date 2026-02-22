#!/usr/bin/env python3
"""
Generate MAPE + Reinforcement Rate plot from eval_simulation CSV output.

Usage:
    python3 eval/plot_mape.py results.csv

Produces: results_mape.png
"""

import sys
import os
import csv
import math

def rolling_median(values, window):
    """Compute rolling median with given window size."""
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = sorted(values[start:i + 1])
        mid = len(chunk) // 2
        if len(chunk) % 2 == 0:
            result.append((chunk[mid - 1] + chunk[mid]) / 2.0)
        else:
            result.append(chunk[mid])
    return result

def rolling_mean(values, window):
    """Compute rolling mean with given window size."""
    result = []
    cumsum = 0.0
    for i in range(len(values)):
        cumsum += values[i]
        if i >= window:
            cumsum -= values[i - window]
        n = min(i + 1, window)
        result.append(cumsum / n)
    return result

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_mape.py <results.csv> [output.png]")
        sys.exit(1)

    csv_path = sys.argv[1]
    if len(sys.argv) >= 3:
        out_png = sys.argv[2]
    else:
        stem = os.path.splitext(csv_path)[0]
        out_png = stem + "_mape.png"

    # Read CSV
    ratio_mapes = []
    ct_mapes = []
    reinforced = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ratio_mapes.append(float(row['mape']) * 100.0)  # to percent
            ct_mapes.append(float(row['comp_time_mape']) * 100.0)
            reinforced.append(int(row['reinforced']))

    if len(ratio_mapes) == 0:
        print("No data rows found in CSV.")
        sys.exit(1)

    n = len(ratio_mapes)
    steps = list(range(1, n + 1))

    # Compute rolling medians (10-step window)
    mape_window = 10
    ratio_rolling = rolling_median(ratio_mapes, mape_window)
    ct_rolling = rolling_median(ct_mapes, mape_window)

    # Compute reinforcement rate (50-step rolling mean)
    reinf_window = 50
    reinf_rate = rolling_mean([float(r) * 100.0 for r in reinforced], reinf_window)

    # Cap at 100%
    ratio_rolling = [min(v, 100.0) for v in ratio_rolling]
    ct_rolling = [min(v, 100.0) for v in ct_rolling]
    reinf_rate = [min(v, 100.0) for v in reinf_rate]

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping plot generation.")
        sys.exit(0)

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Left Y-axis: MAPE
    ax1.set_xlabel('File Index', fontsize=12)
    ax1.set_ylabel('Rolling Median MAPE (%)', fontsize=12)
    ax1.set_ylim(0, 100)

    line1, = ax1.plot(steps, ratio_rolling, color='#E87D0D', linewidth=1.8,
                      label='Compression Ratio MAPE', alpha=0.9)
    line2, = ax1.plot(steps, ct_rolling, color='#2CA02C', linewidth=1.8,
                      label='Compress Time MAPE', alpha=0.9)
    ax1.tick_params(axis='y')

    # Right Y-axis: Reinforcement rate
    ax2 = ax1.twinx()
    ax2.set_ylabel('Reinforcement Rate (%)', fontsize=12, color='darkred')
    ax2.set_ylim(0, 100)
    line3, = ax2.plot(steps, reinf_rate, color='darkred', linewidth=1.4,
                      linestyle='--', label='Reinforcement Rate', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor='darkred')

    # Mark SGD events on x-axis
    sgd_steps = [s for s, r in zip(steps, reinforced) if r == 1]
    if sgd_steps:
        for s in sgd_steps:
            ax1.axvline(x=s, color='darkred', alpha=0.08, linewidth=0.5)

    # Combined legend
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=10)

    ax1.set_title('NN Prediction MAPE with Online Reinforcement', fontsize=14)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Plot saved: {out_png}")

if __name__ == '__main__':
    main()
