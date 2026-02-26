"""
Visualize LR x MAPE sweep benchmark results with per-chunk convergence.

Usage:
  python3 tests/bench_lr_sweep_viz.py [agg.csv] [chunks.csv]

Per pattern (saved to benchmark_hdf5_results/<pattern>/):
  1: Heatmap — LR (y) vs MAPE (x), color = final ratio
  2: Convergence heatmap — LR (y) vs MAPE (x), color = chunk# where converged
  3: Line plot — one line per MAPE, x = LR, y = ratio
  4: Convergence curves — rolling-16 ratio vs chunk, one subplot per MAPE

Summary (saved to benchmark_hdf5_results/):
  S1: NN/BestStatic % — best (lr,mape) per pattern x MAPE
"""
import sys
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

AGG_CSV   = sys.argv[1] if len(sys.argv) > 1 else 'tests/bench_lr_sweep_results/bench_lr_sweep.csv'
CHUNK_CSV = sys.argv[2] if len(sys.argv) > 2 else 'tests/bench_lr_sweep_results/bench_lr_sweep_chunks.csv'
BASE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bench_lr_sweep_results')

df = pd.read_csv(AGG_CSV)
print(f"Loaded {len(df)} rows from {AGG_CSV}")

chunk_df = None
if os.path.exists(CHUNK_CSV):
    chunk_df = pd.read_csv(CHUNK_CSV)
    print(f"Loaded {len(chunk_df)} chunk rows from {CHUNK_CSV}")
else:
    print(f"No chunk CSV at {CHUNK_CSV}")

static_df = df[df['mode'] == 'static']
nn_df     = df[df['mode'] == 'nn']

patterns  = [p for p in df['pattern'].unique() if p in static_df['pattern'].values]
lr_vals   = sorted(nn_df['lr'].unique()) if len(nn_df) else []
mape_vals = sorted(nn_df['mape_thr'].unique()) if len(nn_df) else []

plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.facecolor'] = 'white'

fig_num = 0

# ── Best static per pattern ──
best_static = {}
best_cfg    = {}
for pat in patterns:
    ps = static_df[static_df['pattern'] == pat]
    if len(ps):
        idx = ps['ratio'].idxmax()
        best_static[pat] = ps.loc[idx, 'ratio']
        shuf = '+shuf' if ps.loc[idx, 'shuffle'] > 0 else ''
        best_cfg[pat] = ps.loc[idx, 'algorithm'] + shuf
    else:
        best_static[pat] = 1.0
        best_cfg[pat] = '?'

# ── Distinct colors for 9 LR values ──
LR_COLORS = [
    '#e6194b',  # 0.1 red
    '#f58231',  # 0.2 orange
    '#ffe119',  # 0.3 yellow
    '#3cb44b',  # 0.4 green
    '#42d4f4',  # 0.5 cyan
    '#4363d8',  # 0.6 blue
    '#911eb4',  # 0.7 purple
    '#f032e6',  # 0.8 magenta
    '#000000',  # 0.9 black
]

# ═══════════════════════════════════════════════════════════════
# Per-pattern figures
# ═══════════════════════════════════════════════════════════════
for pat in patterns:
    baseline = best_static[pat]
    cfg_lbl  = best_cfg[pat]
    pat_nn   = nn_df[nn_df['pattern'] == pat]
    if len(pat_nn) == 0 or len(lr_vals) == 0:
        continue

    # Create per-pattern output dir
    pat_dir = os.path.join(BASE_DIR, pat)
    os.makedirs(pat_dir, exist_ok=True)

    # Build matrices
    hm_ratio = np.full((len(lr_vals), len(mape_vals)), np.nan)
    hm_conv  = np.full((len(lr_vals), len(mape_vals)), np.nan)
    hm_sgd   = np.full((len(lr_vals), len(mape_vals)), np.nan)

    for _, row in pat_nn.iterrows():
        li = lr_vals.index(row['lr'])
        mi = mape_vals.index(row['mape_thr'])
        hm_ratio[li, mi] = row['ratio']
        hm_conv[li, mi]  = row['converged_chunk']
        hm_sgd[li, mi]   = row['sgd_fired_count']

    # ── 1: Ratio Heatmap ──
    fig_num += 1
    fig, ax = plt.subplots(figsize=(10, 8))
    vmin = min(np.nanmin(hm_ratio), baseline) * 0.95
    vmax = max(np.nanmax(hm_ratio), baseline) * 1.05
    im = ax.imshow(hm_ratio, cmap='RdYlGn', aspect='auto',
                   vmin=vmin, vmax=vmax, origin='lower')
    ax.set_xticks(range(len(mape_vals)))
    ax.set_xticklabels([f'{m:.2f}' for m in mape_vals])
    ax.set_xlabel('MAPE Threshold')
    ax.set_yticks(range(len(lr_vals)))
    ax.set_yticklabels([f'{l:.1f}' for l in lr_vals])
    ax.set_ylabel('Learning Rate')
    ax.set_title(f'Final Ratio — {pat}\n(best static: {baseline:.2f}x {cfg_lbl})',
                 fontsize=12, fontweight='bold')
    for i in range(len(lr_vals)):
        for j in range(len(mape_vals)):
            v = hm_ratio[i, j]
            if np.isnan(v):
                ax.text(j, i, 'FAIL', ha='center', va='center', fontsize=8, color='red')
            else:
                color = 'white' if (v - vmin) / max(vmax - vmin, 1e-6) > 0.6 else 'black'
                mark = '*' if v >= baseline * 0.99 else ''
                ax.text(j, i, f'{v:.2f}x{mark}', ha='center', va='center',
                        fontsize=9, fontweight='bold', color=color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Compression Ratio')
    cbar.ax.axhline(y=baseline, color='blue', ls='--', lw=2)
    plt.tight_layout()
    path = f'{pat_dir}/lr_sweep_ratio.png'
    plt.savefig(path, bbox_inches='tight'); plt.close()
    print(f"  [{fig_num}] {path}")

    # ── 2: Convergence Heatmap ──
    fig_num += 1
    fig, ax = plt.subplots(figsize=(10, 8))
    conv_display = hm_conv.copy()
    never_mask = conv_display < 0
    conv_display[never_mask] = 256
    im = ax.imshow(conv_display, cmap='RdYlGn_r', aspect='auto',
                   vmin=0, vmax=256, origin='lower')
    ax.set_xticks(range(len(mape_vals)))
    ax.set_xticklabels([f'{m:.2f}' for m in mape_vals])
    ax.set_xlabel('MAPE Threshold')
    ax.set_yticks(range(len(lr_vals)))
    ax.set_yticklabels([f'{l:.1f}' for l in lr_vals])
    ax.set_ylabel('Learning Rate')
    ax.set_title(f'Convergence Speed (chunk #) — {pat}\n'
                 f'(target: 99% of {baseline:.2f}x)',
                 fontsize=12, fontweight='bold')
    for i in range(len(lr_vals)):
        for j in range(len(mape_vals)):
            c = hm_conv[i, j]
            if np.isnan(c) or np.isnan(hm_ratio[i, j]):
                ax.text(j, i, 'FAIL', ha='center', va='center', fontsize=8, color='red')
            elif c < 0:
                ax.text(j, i, 'never', ha='center', va='center',
                        fontsize=9, fontweight='bold', color='white')
            else:
                color = 'white' if c > 128 else 'black'
                ax.text(j, i, f'{int(c)}', ha='center', va='center',
                        fontsize=9, fontweight='bold', color=color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Chunk # at convergence (lower = faster)')
    plt.tight_layout()
    path = f'{pat_dir}/lr_sweep_convergence.png'
    plt.savefig(path, bbox_inches='tight'); plt.close()
    print(f"  [{fig_num}] {path}")

    # ── 3: Line Plot ──
    fig_num += 1
    fig, ax = plt.subplots(figsize=(12, 6))
    colors_mape = ['#e6194b', '#3cb44b', '#4363d8', '#f58231']
    markers = ['o', 's', '^', 'D']
    for mi, mape in enumerate(mape_vals):
        r = [hm_ratio[lr_vals.index(lr), mi] for lr in lr_vals]
        ax.plot(lr_vals, r, marker=markers[mi % len(markers)],
                color=colors_mape[mi % len(colors_mape)], lw=2, ms=7,
                label=f'MAPE={mape:.2f}')
    ax.axhline(y=baseline, color='gray', ls='--', lw=2, alpha=0.7,
               label=f'Best Static ({cfg_lbl}) = {baseline:.2f}x')
    ax.set_xlabel('Learning Rate'); ax.set_ylabel('Compression Ratio')
    ax.set_title(f'NN Ratio vs LR by MAPE — {pat} (no exploration)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xticks(lr_vals)
    plt.tight_layout()
    path = f'{pat_dir}/lr_sweep_lines.png'
    plt.savefig(path, bbox_inches='tight'); plt.close()
    print(f"  [{fig_num}] {path}")

    # ── 4: Per-chunk convergence curves ──
    if chunk_df is not None and len(chunk_df) > 0:
        pat_chunks = chunk_df[chunk_df['pattern'] == pat]
        if len(pat_chunks) > 0:
            fig_num += 1
            fig, axes = plt.subplots(1, len(mape_vals),
                                     figsize=(6 * len(mape_vals), 5),
                                     sharey=True, squeeze=False)
            for mi, mape in enumerate(mape_vals):
                ax = axes[0][mi]
                mape_chunks = pat_chunks[
                    (pat_chunks['mape_thr'] - mape).abs() < 0.001]

                for li, lr in enumerate(lr_vals):
                    lr_data = mape_chunks[
                        (mape_chunks['lr'] - lr).abs() < 0.001].sort_values('chunk_id')
                    if len(lr_data) == 0:
                        continue
                    ax.plot(lr_data['chunk_id'], lr_data['rolling_avg'],
                            lw=1.5, color=LR_COLORS[li],
                            label=f'lr={lr:.1f}' if mi == 0 else None)

                ax.axhline(y=baseline, color='gray', ls='--', lw=2, alpha=0.7)
                ax.axhline(y=baseline * 0.99, color='gray', ls=':', lw=1, alpha=0.4)
                ax.set_xlabel('Chunk')
                ax.set_title(f'MAPE={mape:.2f}', fontweight='bold')
                ax.grid(alpha=0.3)
                if mi == 0:
                    ax.set_ylabel('Rolling-16 Ratio')

            axes[0][0].legend(fontsize=7, loc='best', ncol=2)
            plt.suptitle(f'Convergence Curves — {pat} (target: {baseline:.2f}x {cfg_lbl})',
                         fontsize=13, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            path = f'{pat_dir}/lr_sweep_curves.png'
            plt.savefig(path, bbox_inches='tight'); plt.close()
            print(f"  [{fig_num}] {path}")

# ═══════════════════════════════════════════════════════════════
# Summary: NN/BestStatic % across all patterns
# ═══════════════════════════════════════════════════════════════
if len(patterns) > 1 and len(lr_vals) > 0 and len(mape_vals) > 0:
    fig_num += 1
    fig, ax = plt.subplots(figsize=(14, max(5, len(patterns) * 0.9)))

    summary = np.full((len(patterns), len(mape_vals)), np.nan)
    summary_lr = np.full((len(patterns), len(mape_vals)), np.nan)
    summary_conv = np.full((len(patterns), len(mape_vals)), np.nan)

    for pi, pat in enumerate(patterns):
        bl = best_static[pat]
        pnn = nn_df[nn_df['pattern'] == pat]
        for mi, mape in enumerate(mape_vals):
            sub = pnn[(pnn['mape_thr'] - mape).abs() < 0.001]
            if len(sub):
                best_row = sub.loc[sub['ratio'].idxmax()]
                summary[pi, mi] = best_row['ratio'] / bl * 100 if bl > 0 else 0
                summary_lr[pi, mi] = best_row['lr']
                summary_conv[pi, mi] = best_row['converged_chunk']

    im = ax.imshow(summary, cmap='RdYlGn', aspect='auto', vmin=50, vmax=105)
    ax.set_xticks(range(len(mape_vals)))
    ax.set_xticklabels([f'MAPE={m:.2f}' for m in mape_vals])
    ax.set_yticks(range(len(patterns)))
    ax.set_yticklabels([f'{p} ({best_cfg[p]}, {best_static[p]:.1f}x)' for p in patterns])

    for i in range(len(patterns)):
        for j in range(len(mape_vals)):
            v = summary[i, j]
            lr = summary_lr[i, j]
            conv = summary_conv[i, j]
            if np.isnan(v):
                ax.text(j, i, '-', ha='center', va='center', fontsize=8)
            else:
                color = 'white' if v < 70 else 'black'
                conv_str = f'@{int(conv)}' if conv >= 0 else 'never'
                ax.text(j, i, f'{v:.0f}%\nlr={lr:.1f}\n{conv_str}',
                        ha='center', va='center', fontsize=7,
                        fontweight='bold', color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Best NN / Best Static (%)')
    ax.set_title('Best NN (across LRs) as % of Best Static, per MAPE\n'
                 '(shows best LR and convergence chunk)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = f'{BASE_DIR}/lr_sweep_summary.png'
    plt.savefig(path, bbox_inches='tight'); plt.close()
    print(f"  [{fig_num}] {path}")

print(f"\nDone — {fig_num} figures saved")
