#!/usr/bin/env python3
"""Plot NN CV MAPE and R² bar charts.

Parses the full-pipeline output file automatically, or accepts hardcoded values.
Produces two PNGs:
    neural_net/weights/cv_comparison.png   - MAPE per output
    neural_net/weights/cv_r2.png           - R² per output

Usage:
    python scripts/plot_cv_comparison.py                          # uses latest full-pipeline-*.out
    python scripts/plot_cv_comparison.py full-pipeline-12345.out  # specific file
"""

import re
import glob
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def parse_cv_summary(filepath):
    """Parse NN CV summary from pipeline output file.

    Returns dict: results['nn'][output_name] = {
        'mape': (mean, std), 'r2': (mean, std), 'mae': (mean, std)
    }
    """
    with open(filepath) as f:
        text = f.read()

    results = {}
    idx = text.find('NN 5-FOLD')
    if idx < 0:
        return results
    block = text[idx:idx+2000]
    results['nn'] = {}
    for line in block.split('\n'):
        # Match lines like: compression_time_ms     4.3761±0.0570  0.3986±0.1146   71.4±1.5%
        # Trailing tag like " [lossy-only]" is ignored.
        m = re.match(
            r'\s+(\S+)\s+'
            r'([\d.]+)±([\d.]+)\s+'   # MAE
            r'([\d.]+)±([\d.]+)\s+'   # R²
            r'([\d.]+)±([\d.]+)%',    # MAPE
            line,
        )
        if m:
            name = m.group(1)
            results['nn'][name] = {
                'mae': (float(m.group(2)), float(m.group(3))),
                'r2': (float(m.group(4)), float(m.group(5))),
                'mape': (float(m.group(6)), float(m.group(7))),
            }
    return results


def parse_cv_folds(filepath):
    """Fallback parser for live/incomplete CV logs.

    Parses per-fold metric lines like:
      compression_time_ms  MAE= 15.0636  R²=0.2854  MAPE= 86.6%
    and aggregates mean/std across all observed folds.
    """
    with open(filepath) as f:
        text = f.read()

    # key -> dict(metric -> list)
    acc = defaultdict(lambda: {'mae': [], 'r2': [], 'mape': []})

    # Optional [lossy-only] suffix is ignored.
    pat = re.compile(
        r'^\s+(\S+)\s+MAE=\s*([0-9.eE+-]+)\s+R²=\s*([0-9.eE+-]+)\s+MAPE=\s*([0-9.eE+-]+)%',
        re.MULTILINE
    )
    for m in pat.finditer(text):
        name = m.group(1)
        acc[name]['mae'].append(float(m.group(2)))
        acc[name]['r2'].append(float(m.group(3)))
        acc[name]['mape'].append(float(m.group(4)))

    if not acc:
        return {}

    out = {'nn': {}}
    for name, vals in acc.items():
        out['nn'][name] = {
            'mae': (float(np.mean(vals['mae'])), float(np.std(vals['mae']))),
            'r2': (float(np.mean(vals['r2'])), float(np.std(vals['r2']))),
            'mape': (float(np.mean(vals['mape'])), float(np.std(vals['mape']))),
        }
    return out


# Output display order and labels
OUTPUT_ORDER = [
    ('compression_time_ms', 'Compression\nTime'),
    ('decompression_time_ms', 'Decompression\nTime'),
    ('compression_ratio', 'Compression\nRatio'),
    ('psnr_db', 'PSNR'),
    ('mean_abs_err', 'Pointwise\nError (MAE)'),
    ('ssim', 'SSIM'),
]

# Find pipeline output file
if len(sys.argv) > 1:
    logfile = sys.argv[1]
else:
    files = sorted(glob.glob('full-pipeline-*.out'), key=lambda f: f)
    if not files:
        print("No full-pipeline-*.out found. Run the pipeline first.")
        sys.exit(1)
    logfile = files[-1]

print(f"Parsing: {logfile}")
results = parse_cv_summary(logfile)

if 'nn' not in results:
    print("Summary block not found; falling back to fold-level aggregation.")
    results = parse_cv_folds(logfile)
    if 'nn' not in results:
        print("Could not find usable NN CV metrics in the file.")
        sys.exit(1)

# Build arrays
labels = []
nn_mape, mape_std, nn_r2, r2_std = [], [], [], []
for key, label in OUTPUT_ORDER:
    if key in results['nn']:
        row = results['nn'][key]
        labels.append(label)
        nn_mape.append(row['mape'][0])
        mape_std.append(row['mape'][1])
        nn_r2.append(row['r2'][0])
        r2_std.append(row['r2'][1])

x = np.arange(len(labels))
width = 0.55


# Floor for the log-MAPE chart: any value parsed as <FLOOR is rendered at
# FLOOR so it's visible (and not log(0)=-inf). The original cv_report.log
# rounds quality MAPE to 1 decimal, so e.g. mean_abs_err = "0.0%" gets
# clamped to FLOOR for display only.
MAPE_FLOOR = 0.05  # percent


def _mape_log_chart(values, errs, out_path):
    capped = [max(v, MAPE_FLOOR) for v in values]
    # Clamp lower error bars so they don't go below FLOOR on log scale
    lower_err = [min(e, max(c - MAPE_FLOOR, 0)) for c, e in zip(capped, errs)]
    upper_err = list(errs)

    fig, ax = plt.subplots(figsize=(16, 6))
    bars = ax.bar(x, capped, width, yerr=[lower_err, upper_err], capsize=4,
                  label='Neural Network',
                  color='#AEC7E8', edgecolor='#4C72B0', linewidth=1.2, hatch='///')
    ax.set_yscale('log')
    ax.set_ylim(bottom=MAPE_FLOOR * 0.5, top=max(capped) * 3.0)
    ax.set_ylabel('MAPE (%, log scale)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=11, loc='upper right')

    for bar, raw in zip(bars, values):
        h = bar.get_height()
        # Show actual value; mark capped bars with a "<" prefix
        if raw < MAPE_FLOOR:
            label = f'<{MAPE_FLOOR:.2f}%'
        else:
            label = f'{raw:.2f}%' if raw < 10 else f'{raw:.1f}%'
        ax.text(bar.get_x() + bar.get_width()/2, h * 1.15,
                label, ha='center', va='bottom', fontsize=9, color='#4C72B0')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def _r2_linear_chart(values, errs, out_path):
    fig, ax = plt.subplots(figsize=(16, 6))
    bars = ax.bar(x, values, width, yerr=errs, capsize=4,
                  label='Neural Network',
                  color='#C7E9C0', edgecolor='#2CA02C', linewidth=1.2, hatch='\\\\\\')
    ax.set_ylabel('R² (coefficient of determination)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(bottom=0, top=1.10)
    ax.legend(fontsize=11, loc='upper right')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                f'{h:.3f}', ha='center', va='bottom', fontsize=9, color='#2CA02C')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


# MAPE chart on log y-axis with floor cap so quality bars (PSNR/MAE/SSIM)
# are visible alongside the much larger timing bars.
_mape_log_chart(nn_mape, mape_std, 'neural_net/weights/cv_comparison.png')

# Log tiny-quality MAPE values explicitly for easier reading in terminal logs.
for k in ['mean_abs_err', 'ssim']:
    if k in results['nn']:
        m = results['nn'][k]['mape'][0]
        safe_m = max(m, 1e-12)
        print(f"{k} MAPE: {m:.6f}%  log10(MAPE)={np.log10(safe_m):.4f}")

# R² chart on linear scale (values are already in 0..1 range — readable).
_r2_linear_chart(nn_r2, r2_std, 'neural_net/weights/cv_r2.png')
