"""
Visualize HDF5 lossless benchmark: compression ratio, throughput, and NN adaptation.
Usage: python3 tests/benchmark_hdf5_viz.py [path/to/results.csv] [path/to/chunks.csv]
"""
import sys
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Resolve paths relative to the script's own directory (works regardless of cwd)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV   = os.path.join(SCRIPT_DIR, 'benchmark_hdf5_results', 'benchmark_hdf5_results.csv')
DEFAULT_CHUNK = os.path.join(SCRIPT_DIR, 'benchmark_hdf5_results', 'benchmark_hdf5_chunks.csv')

CSV_PATH   = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
CHUNK_CSV  = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CHUNK
OUT_DIR    = os.path.join(SCRIPT_DIR, 'benchmark_hdf5_results')
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    print(f"ERROR: CSV not found: {CSV_PATH}")
    print(f"  Run the benchmark first: ./build/benchmark_hdf5 <weights.nnwt>")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows from {CSV_PATH}")

# Load per-chunk data for adaptation plots (optional)
chunk_df = None
if os.path.exists(CHUNK_CSV):
    chunk_df = pd.read_csv(CHUNK_CSV)
    print(f"Loaded {len(chunk_df)} chunk rows from {CHUNK_CSV}")
else:
    print(f"No chunk CSV at {CHUNK_CSV} — skipping adaptation plots")

# Auto-detect patterns from the CSV data
PAT2LBL = {
    'constant': 'Constant', 'smooth_sine': 'Smooth Sine', 'ramp': 'Ramp',
    'gaussian': 'Gaussian', 'sparse': 'Sparse', 'step': 'Step',
    'hf_sine_noise': 'HF Sine+Noise', 'exp_decay': 'Exp Decay',
    'sawtooth': 'Sawtooth', 'mixed_freq': 'Mixed Freq',
    'lognormal': 'Log-Normal', 'impulse_train': 'Impulse Train',
    'mixed': 'Mixed (12 patterns)', 'uniform_ramp': 'Uniform Ramp',
    'contiguous': 'Contiguous (12 blocks)', 'cycling': 'Cycling (12 patterns)',
    'sine': 'Sine', 'random': 'Random',
}

PATTERNS = list(df['pattern'].unique())
LABELS   = [PAT2LBL.get(p, p.replace('_', ' ').title()) for p in PATTERNS]
print(f"Patterns in data: {PATTERNS}")

ALGOS    = ['lz4', 'snappy', 'deflate', 'gdeflate', 'zstd', 'ans', 'cascaded', 'bitcomp']
ALGO_LBL = ['LZ4', 'Snappy', 'Deflate', 'Gdeflate', 'Zstd', 'ANS', 'Cascaded', 'Bitcomp']

plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.facecolor'] = 'white'

# ── Subsets ──

none_df   = df[df['mode'] == 'none']
static_df = df[df['mode'] == 'static']
nn_df     = df[df['mode'] == 'nn']

# ── Per-pattern aggregation ──

best_static, best_cfg = [], []
nn_ratio, nn_lbl = [], []
nn_orig_lbl, nn_explored, nn_sgd_fired = [], [], []

for pat in PATTERNS:
    s = static_df[static_df['pattern'] == pat]
    best_static.append(s['compression_ratio'].max() if len(s) else 1.0)
    if len(s):
        row = s.loc[s['compression_ratio'].idxmax()]
        best_cfg.append(row['algorithm'] + ('+shuf' if row.get('shuffle', 0) > 0 else ''))
    else:
        best_cfg.append('')

    n = nn_df[nn_df['pattern'] == pat]
    if len(n):
        r = n.iloc[0]
        nn_ratio.append(r['compression_ratio'])
        nn_lbl.append(r['algorithm'])
        nn_explored.append(int(r.get('explored', 0)))
        nn_sgd_fired.append(int(r.get('sgd_fired', 0)))
        orig = int(r.get('nn_original_action', -1))
        if orig >= 0:
            ai, sh = orig % 8, (orig // 16) % 2
            nn_orig_lbl.append(ALGOS[ai] + ('+shuf' if sh else ''))
        else:
            nn_orig_lbl.append('')
    else:
        nn_ratio.append(1.0); nn_lbl.append('')
        nn_orig_lbl.append(''); nn_explored.append(0); nn_sgd_fired.append(0)

x = np.arange(len(PATTERNS))

# ═══════════════════════════════════════════════════════════════
# Fig 1 — Compression Ratio: Best Static vs NN
# ═══════════════════════════════════════════════════════════════
w = 0.3
fig, ax = plt.subplots(figsize=(16, 7))
b1 = ax.bar(x - w/2, best_static, w, label='Best Static', color='#4A90D9')
b2 = ax.bar(x + w/2, nn_ratio,    w, label='NN Auto',     color='#5CB85C')
ax.set_yscale('log')
ax.set_ylabel('Compression Ratio (log)')
ax.set_xticks(x); ax.set_xticklabels(LABELS, rotation=30, ha='right')
ax.set_title('Lossless Compression Ratio: Best Static vs NN')
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
ax.axhline(y=1, color='gray', ls='--', lw=0.8, alpha=0.5)
ax.set_ylim(bottom=0.8, top=ax.get_ylim()[1] * 5)

for bars, vals, c in [(b1, best_static, '#2C5F9E'), (b2, nn_ratio, '#2D7A2D')]:
    for bar, v in zip(bars, vals):
        if v > 1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
                    f'{v:.0f}x' if v >= 10 else f'{v:.2f}x',
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    rotation=90, color=c)

plt.tight_layout()
path = f'{OUT_DIR}/benchmark_ratio_comparison.png'
plt.savefig(path, bbox_inches='tight'); plt.close()
print(f"  [1/6] {path}")

# ═══════════════════════════════════════════════════════════════
# Fig 2 — NN / Best Static (%)
# ═══════════════════════════════════════════════════════════════
eff = [n / b * 100 if b > 0 else 0 for n, b in zip(nn_ratio, best_static)]

fig, ax = plt.subplots(figsize=(12, 5))
colors = ['#5CB85C' if e >= 95 else '#F0AD4E' if e >= 50 else '#D9534F' for e in eff]
bars = ax.bar(x, eff, 0.5, color=colors)
ax.axhline(y=100, color='red', ls='--', lw=1.5, label='100% = Best Static')
ax.set_ylabel('NN / Best Static (%)'); ax.set_xticks(x)
ax.set_xticklabels(LABELS, rotation=30, ha='right')
ax.set_title('NN Efficiency: Ratio as % of Best Static')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(eff) * 1.15 if eff and max(eff) > 0 else 110)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{bar.get_height():.0f}%', ha='center', va='bottom',
            fontsize=10, fontweight='bold')
avg = sum(eff) / len(eff)
ax.axhline(y=avg, color='blue', ls=':', lw=1, alpha=0.6)
ax.text(len(x) - 0.5, avg + 2, f'avg {avg:.0f}%', color='blue', fontsize=9)

plt.tight_layout()
path = f'{OUT_DIR}/benchmark_nn_efficiency.png'
plt.savefig(path, bbox_inches='tight'); plt.close()
print(f"  [2/6] {path}")

# ═══════════════════════════════════════════════════════════════
# Fig 3 — Heatmap: No-Shuffle vs Shuffle
# ═══════════════════════════════════════════════════════════════
n_pat = len(PATTERNS)
hm_ns = np.ones((n_pat, len(ALGOS)))
hm_s4 = np.ones((n_pat, len(ALGOS)))

for i, pat in enumerate(PATTERNS):
    for j, algo in enumerate(ALGOS):
        for hm, sv in [(hm_ns, 0), (hm_s4, 4)]:
            sub = static_df[(static_df['pattern'] == pat) &
                            (static_df['algorithm'] == algo) &
                            (static_df['shuffle'] == sv)]
            if len(sub):
                hm[i, j] = sub['compression_ratio'].max()

vmax = max(np.log10(np.clip(hm_ns, 1, None)).max(),
           np.log10(np.clip(hm_s4, 1, None)).max())

fig, (a1, a2) = plt.subplots(1, 2, figsize=(20, max(4, n_pat * 0.8)), sharey=True)
for ax, hm, t in [(a1, hm_ns, 'No Shuffle'), (a2, hm_s4, 'Shuffle (4-byte)')]:
    im = ax.imshow(np.log10(np.clip(hm, 1, None)), cmap='YlOrRd',
                   aspect='auto', vmin=0, vmax=vmax)
    ax.set_xticks(range(len(ALGOS))); ax.set_xticklabels(ALGO_LBL, rotation=45, ha='right')
    ax.set_yticks(range(n_pat)); ax.set_yticklabels(LABELS)
    ax.set_title(f'Lossless Ratio — {t}', fontsize=11, fontweight='bold')
    for i in range(n_pat):
        for j in range(len(ALGOS)):
            v = hm[i, j]
            txt = f'{v:.1f}x' if v < 100 else f'{v:.0f}x'
            c = 'white' if np.log10(max(v, 1)) > vmax * 0.55 else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=7, color=c)
cbar = fig.colorbar(im, ax=[a1, a2], shrink=0.8, pad=0.02)
cbar.set_label('log10(Compression Ratio)')
plt.suptitle('Lossless Ratio by Algorithm x Pattern', fontsize=13, fontweight='bold', y=0.99)
fig.subplots_adjust(top=0.9, right=0.88, wspace=0.15)
path = f'{OUT_DIR}/benchmark_algo_heatmap.png'
plt.savefig(path, bbox_inches='tight'); plt.close()
print(f"  [3/6] {path}")

# ═══════════════════════════════════════════════════════════════
# Fig 4 — Per-Pattern: Algos vs NN (subplots)
# ═══════════════════════════════════════════════════════════════
ncols = min(4, len(PATTERNS))
nrows = (len(PATTERNS) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
for idx, (pat, lbl) in enumerate(zip(PATTERNS, LABELS)):
    ax = axes.flat[idx]
    sub = static_df[static_df['pattern'] == pat]
    ns, s4 = [], []
    for algo in ALGOS:
        for lst, sv in [(ns, 0), (s4, 4)]:
            s = sub[(sub['algorithm'] == algo) & (sub['shuffle'] == sv)]
            lst.append(s['compression_ratio'].max() if len(s) else 1.0)

    nn_v = nn_ratio[idx]
    n, w = len(ALGOS), 0.3
    ax.bar(np.arange(n) - w/2, ns, w, label='No Shuffle', color='#4A90D9', alpha=0.85)
    ax.bar(np.arange(n) + w/2, s4, w, label='Shuffle 4B', color='#F0AD4E', alpha=0.85)
    ax.bar([n], [nn_v], w * 2, label='NN', color='#5CB85C')
    ax.text(n, nn_v * 1.05, f'{nn_v:.0f}x' if nn_v >= 10 else f'{nn_v:.2f}x',
            ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.set_xticks(range(n + 1))
    ax.set_xticklabels(ALGO_LBL + ['NN'], rotation=45, ha='right', fontsize=6)
    ax.set_title(lbl, fontsize=10, fontweight='bold')
    ax.set_ylabel('Ratio', fontsize=8)
    all_v = ns + s4 + [nn_v]
    if max(all_v) / max(min(v for v in all_v if v > 0), 0.1) > 20:
        ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=6, loc='upper left')

for idx in range(len(PATTERNS), nrows * ncols):
    axes.flat[idx].set_visible(False)
plt.suptitle('Per-Pattern Lossless Ratio: All Algorithms vs NN', fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
path = f'{OUT_DIR}/benchmark_per_algo_vs_nn.png'
plt.savefig(path, bbox_inches='tight'); plt.close()
print(f"  [4/6] {path}")

# ═══════════════════════════════════════════════════════════════
# Fig 5 — NN Analysis Table
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(20, 8))
ax.axis('off')

cols = ['Pattern', 'Best Static', 'Config',
        'NN Initial', 'NN Final', 'Ratio', 'NN/Best',
        'Explored', 'SGD']
rows = []
for i in range(len(PATTERNS)):
    pct = nn_ratio[i] / best_static[i] * 100 if best_static[i] > 0 else 0
    changed = nn_explored[i] and nn_orig_lbl[i] != nn_lbl[i]
    rows.append([
        LABELS[i],
        f'{best_static[i]:.2f}x', best_cfg[i],
        nn_orig_lbl[i], nn_lbl[i] if changed else '-',
        f'{nn_ratio[i]:.2f}x', f'{pct:.0f}%',
        'Yes' if nn_explored[i] else 'No',
        'Yes' if nn_sgd_fired[i] else 'No',
    ])

avg_pct = sum(nn_ratio[i] / best_static[i] * 100
              for i in range(len(PATTERNS)) if best_static[i] > 0) / len(PATTERNS)
n_expl = sum(nn_explored)
n_sgd  = sum(nn_sgd_fired)
n_chg  = sum(1 for i in range(len(PATTERNS))
             if nn_explored[i] and nn_orig_lbl[i] != nn_lbl[i])
rows.append(['AVERAGE', '', '', '', '',
             '', f'{avg_pct:.0f}%',
             f'{n_expl}/{len(PATTERNS)}', f'{n_sgd}/{len(PATTERNS)}'])

tbl = ax.table(cellText=rows, colLabels=cols, cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.6)

for i, row in enumerate(rows):
    pct = float(row[6].replace('%', ''))
    tbl[i + 1, 6].set_facecolor('#d4edda' if pct >= 95 else '#fff3cd' if pct >= 80 else '#f8d7da')
    if i < len(PATTERNS):
        tbl[i + 1, 7].set_facecolor('#fff3cd' if row[7] == 'Yes' else '#d4edda')
        tbl[i + 1, 8].set_facecolor('#f8d7da' if row[8] == 'Yes' else '#d4edda')
        # Highlight NN Final when changed
        if row[4] != '-':
            tbl[i + 1, 4].set_facecolor('#fff3cd')

for j in range(len(cols)):
    tbl[len(rows), j].set_text_props(fontweight='bold')
    tbl[len(rows), j].set_facecolor('#e9ecef')
    tbl[0, j].set_facecolor('#343a40')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

ax.set_title('NN Prediction vs Best Static (NN Final shown only when exploration changed the choice)',
             fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
path = f'{OUT_DIR}/benchmark_nn_analysis.png'
plt.savefig(path, bbox_inches='tight'); plt.close()
print(f"  [5/6] {path}")

# ═══════════════════════════════════════════════════════════════
# Fig 6 — Throughput: No Compression vs Best Static vs NN (Write & Read MB/s)
# ═══════════════════════════════════════════════════════════════
none_write, none_read = [], []
best_write, best_read = [], []
nn_write, nn_read = [], []

for pat in PATTERNS:
    nz = none_df[none_df['pattern'] == pat]
    if len(nz):
        r0 = nz.iloc[0]
        none_write.append(r0['write_mbps'])
        none_read.append(r0['read_mbps'])
    else:
        none_write.append(0); none_read.append(0)

    s = static_df[static_df['pattern'] == pat]
    if len(s):
        row = s.loc[s['compression_ratio'].idxmax()]
        best_write.append(row['write_mbps'])
        best_read.append(row['read_mbps'])
    else:
        best_write.append(0); best_read.append(0)

    n = nn_df[nn_df['pattern'] == pat]
    if len(n):
        r = n.iloc[0]
        nn_write.append(r['write_mbps'])
        nn_read.append(r['read_mbps'])
    else:
        nn_write.append(0); nn_read.append(0)

w = 0.25
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), sharey=False)

# Write throughput
b0 = ax1.bar(x - w, none_write, w, label='No Compression', color='#999999')
b1 = ax1.bar(x,     best_write, w, label='Best Static',    color='#4A90D9')
b2 = ax1.bar(x + w, nn_write,   w, label='NN Auto',        color='#5CB85C')
ax1.set_ylabel('Write Throughput (MB/s)')
ax1.set_xticks(x); ax1.set_xticklabels(LABELS, rotation=30, ha='right')
ax1.set_title('Write Throughput (compress + HDF5 write)', fontsize=11, fontweight='bold')
ax1.legend(fontsize=9); ax1.grid(axis='y', alpha=0.3)
for bars in [b0, b1, b2]:
    for bar in bars:
        if bar.get_height() > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=7)

# Read throughput
b3 = ax2.bar(x - w, none_read, w, label='No Compression', color='#999999')
b4 = ax2.bar(x,     best_read, w, label='Best Static',    color='#4A90D9')
b5 = ax2.bar(x + w, nn_read,   w, label='NN Auto',        color='#5CB85C')
ax2.set_ylabel('Read Throughput (MB/s)')
ax2.set_xticks(x); ax2.set_xticklabels(LABELS, rotation=30, ha='right')
ax2.set_title('Read Throughput (HDF5 read + decompress)', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.3)
for bars in [b3, b4, b5]:
    for bar in bars:
        if bar.get_height() > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=7)

plt.suptitle('Throughput: No Compression vs Best Static vs NN Auto', fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
path = f'{OUT_DIR}/benchmark_throughput.png'
plt.savefig(path, bbox_inches='tight'); plt.close()
print(f"  [6/6] {path}")

fig_num = 6

# ═══════════════════════════════════════════════════════════════
# Fig 7 — NN Adaptation: Exploration Per Chunk + Windowed Rate
# ═══════════════════════════════════════════════════════════════

ADAPT_WINDOW = 16  # rolling / window size

def rolling_avg(arr, w):
    """Causal rolling average (expanding at the start)."""
    cum = np.cumsum(arr)
    cum = np.insert(cum, 0, 0)
    out = np.zeros(len(arr))
    for i in range(len(arr)):
        lo = max(0, i - w + 1)
        out[i] = (cum[i + 1] - cum[lo]) / (i - lo + 1)
    return out

if chunk_df is not None and len(chunk_df) > 0:
    chunk_patterns = chunk_df['pattern'].unique()
    has_chunk_pattern = 'chunk_pattern' in chunk_df.columns

    # Color palette for chunk patterns
    CPAT_COLORS = {
        'constant': '#e6194b', 'smooth_sine': '#3cb44b', 'ramp': '#4363d8',
        'gaussian': '#f58231', 'sparse': '#911eb4', 'step': '#42d4f4',
        'hf_sine_noise': '#f032e6', 'exp_decay': '#bfef45', 'sawtooth': '#fabed4',
        'mixed_freq': '#469990', 'lognormal': '#dcbeff', 'impulse_train': '#9A6324',
    }

    for pat in chunk_patterns:
        pdf = chunk_df[chunk_df['pattern'] == pat].copy().reset_index(drop=True)
        n_chunks = len(pdf)
        if n_chunks == 0:
            continue

        chunks   = pdf['chunk_id'].values
        explored = pdf['explored'].values.astype(float)
        ratio    = pdf['ratio'].values

        expl_rolling  = rolling_avg(explored, ADAPT_WINDOW)
        ratio_rolling = rolling_avg(ratio, ADAPT_WINDOW)

        # Non-overlapping windows
        n_windows = n_chunks // ADAPT_WINDOW
        win_expl_pct = np.array([
            explored[i*ADAPT_WINDOW:(i+1)*ADAPT_WINDOW].mean() * 100
            for i in range(n_windows)])

        # ── Compute per-chunk APE against best ratio per chunk_pattern ──
        # Target = best (max) ratio achieved for each chunk_pattern across
        # the full run.  APE = |ratio - target| / target * 100.
        # For uniform mode (no chunk_pattern), use global best ratio.
        if has_chunk_pattern:
            cpats = pdf['chunk_pattern'].values
            best_per_cpat = {}
            for cp in set(cpats):
                best_per_cpat[cp] = ratio[cpats == cp].max()
            target = np.array([best_per_cpat[cp] for cp in cpats])
        else:
            target = np.full(n_chunks, ratio.max())

        ape = np.where(target > 0, np.abs(ratio - target) / target * 100.0, 0.0)
        mape_rolling = rolling_avg(ape, ADAPT_WINDOW)

        # Non-overlapping window MAPE
        win_mape = np.array([
            ape[i*ADAPT_WINDOW:(i+1)*ADAPT_WINDOW].mean()
            for i in range(n_windows)])

        # ── Detect pattern boundaries (contiguous blocks) ──
        pat_boundaries = []  # list of (chunk_idx, pattern_name)
        if has_chunk_pattern:
            prev_cp = None
            for i, cp in enumerate(cpats):
                if cp != prev_cp:
                    pat_boundaries.append((i, cp))
                    prev_cp = cp

        def draw_pattern_lines(ax, boundaries, ymin=None, ymax=None):
            """Draw vertical lines at pattern transitions with labels."""
            for idx, (ci, cp) in enumerate(boundaries):
                if ci == 0:
                    continue  # skip first boundary (start of dataset)
                ax.axvline(x=ci, color='#555555', ls='--', lw=0.8, alpha=0.6, zorder=4)
                if ymax is not None:
                    ax.text(ci + 2, ymax * 0.95,
                            PAT2LBL.get(cp, cp), fontsize=5, rotation=90,
                            va='top', ha='left', color='#555555', alpha=0.8)

        fig_num += 1
        fig, (ax_top, ax_mape, ax_mid, ax_bot) = plt.subplots(
            4, 1, figsize=(16, 18),
            gridspec_kw={'height_ratios': [2, 3, 2, 2]})
        fig.suptitle(
            f'NN Adaptation — Pattern: {pat}  '
            f'({n_chunks} chunks x 4 MB = {n_chunks * 4} MB)',
            fontsize=13, fontweight='bold', y=0.98)

        # ── Panel 1: Exploration triggered per chunk ──
        expl_idx   = np.where(explored > 0)[0]
        noexpl_idx = np.where(explored == 0)[0]
        ax_top.bar(noexpl_idx, np.ones(len(noexpl_idx)), width=1.0,
                   color='#d4edda', alpha=0.5, label='No exploration')
        ax_top.bar(expl_idx, np.ones(len(expl_idx)), width=1.0,
                   color='#f5c6cb', alpha=0.8, label='Exploration triggered')
        ax_top.plot(chunks, expl_rolling * 100, color='#c0392b', lw=2.0,
                    label=f'Rolling avg ({ADAPT_WINDOW}-chunk)')
        ax_top.set_xlabel('Chunk')
        ax_top.set_ylabel('Exploration Rate (%)')
        ax_top.set_title('Exploration Triggered Per Chunk', fontweight='bold')
        ax_top.set_ylim(-5, 110)
        ax_top.legend(fontsize=9, loc='upper right')
        ax_top.grid(axis='y', alpha=0.3)

        total_expl = int(explored.sum())
        ax_top.text(0.01, 0.92,
                    f'Total: {total_expl}/{n_chunks} chunks explored '
                    f'({100*total_expl/n_chunks:.0f}%)',
                    transform=ax_top.transAxes, fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

        if len(expl_idx) > 0:
            ax_top.annotate(f'First: chunk {expl_idx[0]}',
                            xy=(expl_idx[0], 100), fontsize=7,
                            color='#c0392b', ha='left', va='bottom')
            if expl_idx[-1] != expl_idx[0]:
                ax_top.annotate(f'Last: chunk {expl_idx[-1]}',
                                xy=(expl_idx[-1], 100), fontsize=7,
                                color='#c0392b', ha='right', va='bottom')

        draw_pattern_lines(ax_top, pat_boundaries, ymax=110)

        # ── Panel 2: MAPE — how well the model adapts over time ──
        # Scatter per-chunk APE colored by chunk_pattern
        if has_chunk_pattern:
            for cp in sorted(set(cpats)):
                mask = cpats == cp
                c = CPAT_COLORS.get(cp, '#e67e22')
                ax_mape.scatter(chunks[mask], ape[mask], s=4, alpha=0.3,
                                color=c, zorder=2)
        else:
            ax_mape.scatter(chunks, ape, s=4, alpha=0.25, color='#e67e22', zorder=2,
                            label='Per-chunk APE')

        ax_mape.plot(chunks, mape_rolling, color='#c0392b', lw=2.5, zorder=3,
                     label=f'Rolling MAPE ({ADAPT_WINDOW}-chunk)')

        # Shade first-quarter vs last-quarter to show improvement
        q1 = min(n_chunks // 4, n_chunks)
        q4_start = max(n_chunks - n_chunks // 4, 0)
        mape_first_q = ape[:q1].mean() if q1 > 0 else 0
        mape_last_q  = ape[q4_start:].mean() if q4_start < n_chunks else 0
        ax_mape.text(0.01, 0.92,
                     f'MAPE first 25%: {mape_first_q:.1f}%  |  '
                     f'MAPE last 25%: {mape_last_q:.1f}%  |  '
                     f'Overall MAPE: {ape.mean():.1f}%',
                     transform=ax_mape.transAxes, fontsize=9,
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

        # Windowed MAPE bars (background)
        if n_windows > 0:
            win_x = np.array([(i + 0.5) * ADAPT_WINDOW for i in range(n_windows)])
            ax_mape_bar = ax_mape.twinx()
            ax_mape_bar.bar(win_x, win_mape, width=ADAPT_WINDOW * 0.9,
                            alpha=0.15, color='#8e44ad', zorder=1, label='Window MAPE')
            ax_mape_bar.set_ylabel('Window MAPE (%)', fontsize=8, color='#8e44ad')
            ax_mape_bar.tick_params(axis='y', labelcolor='#8e44ad', labelsize=7)
            ymax = max(ape.max() * 1.1, 10)
            ax_mape.set_ylim(0, ymax)
            ax_mape_bar.set_ylim(0, ymax)

        ax_mape.set_xlabel('Chunk')
        ax_mape.set_ylabel('Absolute Percentage Error (%)')
        ax_mape.set_title(
            'MAPE: Compression Ratio Error vs Best-Per-Pattern '
            '(lower = better adaptation)', fontweight='bold')
        ax_mape.legend(fontsize=8, loc='upper right')
        ax_mape.grid(axis='y', alpha=0.3)
        mape_ymax = ax_mape.get_ylim()[1]
        draw_pattern_lines(ax_mape, pat_boundaries, ymax=mape_ymax)

        # ── Panel 3: Compression ratio convergence (colored by chunk_pattern) ──
        if has_chunk_pattern:
            unique_cpats = sorted(set(cpats))
            for cp in unique_cpats:
                mask = cpats == cp
                c = CPAT_COLORS.get(cp, '#2980b9')
                ax_mid.scatter(chunks[mask], ratio[mask], s=6, alpha=0.4,
                               color=c, zorder=2)
            if len(unique_cpats) > 1 and len(unique_cpats) <= 12:
                handles = [plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=CPAT_COLORS.get(cp, '#2980b9'),
                           markersize=5, label=PAT2LBL.get(cp, cp)) for cp in unique_cpats]
                ax_mid.legend(handles=handles, fontsize=6, loc='upper right',
                              ncol=2, title='Chunk pattern', title_fontsize=7)
        else:
            ax_mid.scatter(chunks, ratio, s=6, alpha=0.4, color='#2980b9', zorder=2,
                           label='Per-chunk ratio')

        ax_mid.plot(chunks, ratio_rolling, color='#e74c3c', lw=2.0, zorder=3,
                    label=f'Rolling avg ({ADAPT_WINDOW}-chunk)')
        if n_chunks > ADAPT_WINDOW:
            final_avg = ratio[-ADAPT_WINDOW:].mean()
            ax_mid.axhline(y=final_avg, color='#27ae60', ls='--', lw=1.5,
                           alpha=0.7, label=f'Final avg: {final_avg:.2f}x')
        ax_mid.set_xlabel('Chunk')
        ax_mid.set_ylabel('Compression Ratio')
        ax_mid.set_title('Compression Ratio Convergence', fontweight='bold')
        if not has_chunk_pattern or len(set(cpats)) <= 1:
            ax_mid.legend(fontsize=8, loc='lower right')
        ax_mid.grid(alpha=0.3)
        ratio_ymax = ax_mid.get_ylim()[1]
        draw_pattern_lines(ax_mid, pat_boundaries, ymax=ratio_ymax)

        # ── Panel 4: Exploration rate per window ──
        win_labels = [f'{i*ADAPT_WINDOW+1}-{(i+1)*ADAPT_WINDOW}'
                      for i in range(n_windows)]
        colors_bar = ['#e74c3c' if p > 50 else '#f39c12' if p > 0 else '#27ae60'
                      for p in win_expl_pct]
        bars_w = ax_bot.bar(range(n_windows), win_expl_pct, color=colors_bar,
                            edgecolor='white', lw=0.5)
        ax_bot.set_xticks(range(n_windows))
        step = max(1, n_windows // 16)
        ax_bot.set_xticklabels(
            [win_labels[i] if i % step == 0 else '' for i in range(n_windows)],
            rotation=45, ha='right', fontsize=7)
        ax_bot.set_xlabel('Chunk Window')
        ax_bot.set_ylabel('Exploration Rate (%)')
        ax_bot.set_title(
            f'Exploration Rate per {ADAPT_WINDOW}-Chunk Window',
            fontweight='bold')
        ax_bot.set_ylim(0, 105)
        ax_bot.grid(axis='y', alpha=0.3)

        for bar, pct in zip(bars_w, win_expl_pct):
            if pct > 0:
                ax_bot.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 1.5,
                            f'{pct:.0f}%', ha='center', va='bottom',
                            fontsize=7, fontweight='bold')

        # Shade "adapted" region (contiguous 0% windows to the end)
        zero_start = None
        for i in range(n_windows):
            if win_expl_pct[i] == 0 and zero_start is None:
                zero_start = i
            elif win_expl_pct[i] > 0:
                zero_start = None
        if zero_start is not None and zero_start < n_windows - 1:
            ax_bot.axvspan(zero_start - 0.5, n_windows - 0.5, alpha=0.1,
                           color='green', zorder=0)
            ax_bot.text((zero_start + n_windows) / 2, 50, 'Model adapted',
                        ha='center', va='center', fontsize=11, color='#27ae60',
                        fontweight='bold', alpha=0.7)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = f'{OUT_DIR}/adaptation_{pat}.png'
        plt.savefig(path, bbox_inches='tight'); plt.close()
        print(f"  [{fig_num}/{fig_num}] {path}")

    # ═══════════════════════════════════════════════════════════════
    # Fig 8 — Per-chunk-pattern ratio boxplot (mixed mode)
    # ═══════════════════════════════════════════════════════════════
    if has_chunk_pattern and len(chunk_df['chunk_pattern'].unique()) > 1:
        fig_num += 1
        cpat_order = sorted(chunk_df['chunk_pattern'].unique(),
                            key=lambda p: chunk_df[chunk_df['chunk_pattern'] == p]['ratio'].median(),
                            reverse=True)
        fig, ax = plt.subplots(figsize=(14, 6))
        box_data = [chunk_df[chunk_df['chunk_pattern'] == cp]['ratio'].values for cp in cpat_order]
        bp = ax.boxplot(box_data, vert=True, patch_artist=True, widths=0.6,
                        medianprops=dict(color='black', lw=2))
        for patch, cp in zip(bp['boxes'], cpat_order):
            patch.set_facecolor(CPAT_COLORS.get(cp, '#4A90D9'))
            patch.set_alpha(0.7)
        ax.set_xticklabels([PAT2LBL.get(cp, cp) for cp in cpat_order],
                           rotation=35, ha='right', fontsize=9)
        ax.set_ylabel('Compression Ratio')
        ax.set_title('Compression Ratio by Chunk Pattern (Mixed Mode)', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        # Annotate medians
        for i, cp in enumerate(cpat_order):
            med = chunk_df[chunk_df['chunk_pattern'] == cp]['ratio'].median()
            ax.text(i + 1, med, f' {med:.1f}x', va='center', fontsize=7, fontweight='bold')
        plt.tight_layout()
        path = f'{OUT_DIR}/ratio_by_chunk_pattern.png'
        plt.savefig(path, bbox_inches='tight'); plt.close()
        print(f"  [{fig_num}/{fig_num}] {path}")

    # Adaptation summary
    print("\n=== Adaptation Summary ===")
    for pat in chunk_patterns:
        pdf = chunk_df[chunk_df['pattern'] == pat]
        n = len(pdf)
        n_expl = int(pdf['explored'].sum())
        expl_chunks = pdf[pdf['explored'] > 0]['chunk_id']
        last_expl = int(expl_chunks.max()) if len(expl_chunks) > 0 else -1
        print(f"  {pat}: {n_expl}/{n} explored ({100*n_expl/n:.0f}%), "
              f"last exploration at chunk {last_expl}")

    if has_chunk_pattern and len(chunk_df['chunk_pattern'].unique()) > 1:
        print("\n=== Per-Pattern Compression Summary ===")
        for cp in sorted(chunk_df['chunk_pattern'].unique()):
            sub = chunk_df[chunk_df['chunk_pattern'] == cp]
            med = sub['ratio'].median()
            mean = sub['ratio'].mean()
            n_e = int(sub['explored'].sum())
            print(f"  {PAT2LBL.get(cp, cp):20s}: median {med:7.2f}x, mean {mean:7.2f}x, "
                  f"explored {n_e}/{len(sub)}")

print(f"\nDone — {fig_num} figures saved to {OUT_DIR}/")
