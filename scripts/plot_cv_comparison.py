#!/usr/bin/env python3
"""Plot NN vs XGBoost 5-fold CV comparison bar chart.

Parses the full-pipeline output file automatically, or accepts hardcoded values.

Usage:
    python scripts/plot_cv_comparison.py                          # uses latest full-pipeline-*.out
    python scripts/plot_cv_comparison.py full-pipeline-12345.out  # specific file
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import glob
import re


def parse_cv_summary(filepath):
    """Parse NN and XGBoost CV summaries from pipeline output file."""
    with open(filepath) as f:
        text = f.read()

    results = {}
    for model, header in [('nn', 'NN 5-FOLD'), ('xgb', 'XGBOOST 5-FOLD')]:
        idx = text.find(header)
        if idx < 0:
            continue
        block = text[idx:idx+2000]
        results[model] = {}
        for line in block.split('\n'):
            # Match lines like: compression_time_ms     4.3761±0.0570  0.3986±0.1146   71.4±1.5%
            m = re.match(r'\s+(\S+)\s+[\d.]+±[\d.]+\s+[\d.]+±[\d.]+\s+([\d.]+)±([\d.]+)%', line)
            if m:
                name = m.group(1)
                mape = float(m.group(2))
                std = float(m.group(3))
                results[model][name] = (mape, std)
    return results


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

if 'nn' not in results or 'xgb' not in results:
    print("Could not find both NN and XGBoost CV summaries in the file.")
    sys.exit(1)

# Build arrays
labels = []
nn_mape, nn_std, xgb_mape, xgb_std = [], [], [], []
for key, label in OUTPUT_ORDER:
    if key in results['nn'] and key in results['xgb']:
        labels.append(label)
        nn_mape.append(results['nn'][key][0])
        nn_std.append(results['nn'][key][1])
        xgb_mape.append(results['xgb'][key][0])
        xgb_std.append(results['xgb'][key][1])

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(16, 6))
bars1 = ax.bar(x - width/2, nn_mape, width, label='Neural Network',
               color='#AEC7E8', edgecolor='#4C72B0', linewidth=1.2, hatch='///')
bars2 = ax.bar(x + width/2, xgb_mape, width, label='XGBoost',
               color='#FFBE7A', edgecolor='#DD8452', linewidth=1.2, hatch='...')

ax.set_ylabel('MAPE (%)', fontsize=12)
ax.set_title('')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.legend(fontsize=11, loc='upper right')

# Add MAPE values on bars
for bar in bars1:
    h = bar.get_height()
    if h > 0.05:
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.0, f'{h:.1f}%',
                ha='center', va='bottom', fontsize=7, color='#4C72B0')
for bar in bars2:
    h = bar.get_height()
    if h > 0.05:
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.0, f'{h:.1f}%',
                ha='center', va='bottom', fontsize=7, color='#DD8452')

plt.tight_layout()
out_path = 'neural_net/weights/cv_comparison.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out_path}")
