#!/bin/bash
# Plot partial results from the stuck 4n4g run (timesteps 0-6)
# Usage: bash scripts/plot_partial_4n4g.sh

cd /u/$USER/GPUCompress

RESULTS=benchmarks/vpic-kokkos/results/eval_NX320_chunk4mb_ts50
NRANKS=16

echo ">>> Step 1: Merging per-rank CSVs and splitting by policy..."

for pol in balanced ratio; do
    case "$pol" in
        balanced) L="balanced_w1-1-1" ;;
        ratio)    L="ratio_only_w0-0-1" ;;
    esac

    mkdir -p "$RESULTS/$L"

    for b in benchmark_vpic_deck_timesteps benchmark_vpic_deck_timestep_chunks benchmark_vpic_deck_ranking benchmark_vpic_deck_ranking_costs; do
        cp "$RESULTS/${b}_rank0.csv" "$RESULTS/${b}.csv"
        for r in $(seq 1 $((NRANKS - 1))); do
            [ -f "$RESULTS/${b}_rank${r}.csv" ] && tail -n+2 "$RESULTS/${b}_rank${r}.csv" >> "$RESULTS/${b}.csv"
        done

        # Phase column: $2 for timesteps/chunks, $1 for ranking
        pc=2
        case "$b" in *ranking*) pc=1 ;; esac
        head -1 "$RESULTS/${b}.csv" > "$RESULTS/$L/${b}.csv"
        tail -n+2 "$RESULTS/${b}.csv" | awk -F',' -v p="/$pol" -v pc="$pc" '{
            if (index($pc, "/") == 0) print;
            else if (index($pc, p) > 0) { gsub(p, "", $pc); print }
        }' OFS=',' >> "$RESULTS/$L/${b}.csv"
    done

    echo "  Split CSVs -> $RESULTS/$L/"

    # ── Generate per-policy aggregate CSV ──
    TS_CSV="$RESULTS/$L/benchmark_vpic_deck_timesteps.csv"
    AGG_CSV="$RESULTS/$L/benchmark_vpic_deck.csv"
    if [ -f "$TS_CSV" ]; then
        echo "  Generating aggregate CSV -> $AGG_CSV"
        python3 -c "
import csv, math

rows = list(csv.DictReader(open('$TS_CSV')))
phases = {}
for r in rows:
    p = r['phase']
    if p not in phases:
        phases[p] = {'sum_wr':0,'sum_rd':0,'sum_wr_sq':0,'sum_rd_sq':0,
                     'sum_file_bytes':0,'n':0,'n_chunks':0,
                     'sum_sgd':0,'sum_expl':0,'sum_mape_r':0,'sum_mape_c':0,'sum_mape_d':0,
                     'sum_mape_p':0,'sum_mae_r':0,'sum_mae_c':0,'sum_mae_d':0,'sum_mae_p':0,
                     'sum_stats':0,'sum_nn':0,'sum_pre':0,'sum_comp':0,'sum_dec':0,
                     'sum_expl_ms':0,'sum_sgd_ms':0}
    d = phases[p]
    wr_val = float(r.get('write_ms',0))
    rd_val = float(r.get('read_ms',0))
    d['sum_wr'] += wr_val
    d['sum_rd'] += rd_val
    d['sum_wr_sq'] += wr_val * wr_val
    d['sum_rd_sq'] += rd_val * rd_val
    d['sum_file_bytes'] += int(r.get('file_bytes',0))
    d['n_chunks'] = int(r.get('n_chunks',0))
    d['sum_sgd'] += int(r.get('sgd_fires',0))
    d['sum_expl'] += int(r.get('explorations',0))
    d['sum_mape_r'] += float(r.get('mape_ratio',0))
    d['sum_mape_c'] += float(r.get('mape_comp',0))
    d['sum_mape_d'] += float(r.get('mape_decomp',0))
    d['sum_mape_p'] += float(r.get('mape_psnr',0))
    d['sum_mae_r'] += float(r.get('mae_ratio',0))
    d['sum_mae_c'] += float(r.get('mae_comp_ms',0))
    d['sum_mae_d'] += float(r.get('mae_decomp_ms',0))
    d['sum_mae_p'] += float(r.get('mae_psnr_db',0))
    d['sum_stats'] += float(r.get('stats_ms',0))
    d['sum_nn'] += float(r.get('nn_ms',0))
    d['sum_pre'] += float(r.get('preproc_ms',0))
    d['sum_comp'] += float(r.get('comp_ms',0))
    d['sum_dec'] += float(r.get('decomp_ms',0))
    d['sum_expl_ms'] += float(r.get('explore_ms',0))
    d['sum_sgd_ms'] += float(r.get('sgd_ms',0))
    d['n'] += 1
first_orig = 0
if 'orig_mib' in rows[0]:
    first_orig = float(rows[0].get('orig_mib',0))
if first_orig <= 0:
    for r in rows:
        if r['phase'] == 'no-comp':
            first_orig = int(r.get('file_bytes',0)) / (1024*1024)
            break
with open('$AGG_CSV','w') as f:
    hdr = ('rank,source,phase,n_runs,write_ms,write_ms_std,read_ms,read_ms_std,'
           'file_mib,orig_mib,ratio,write_mibps,read_mibps,mismatches,'
           'sgd_fires,explorations,n_chunks,'
           'nn_ms,stats_ms,preproc_ms,comp_ms,decomp_ms,explore_ms,sgd_ms,'
           'comp_gbps,decomp_gbps,'
           'mape_ratio_pct,mape_comp_pct,mape_decomp_pct,mape_psnr_pct,'
           'mae_ratio,mae_comp_ms,mae_decomp_ms,mae_psnr_db')
    f.write(hdr + '\n')
    for p,d in phases.items():
        n = d['n']
        if n == 0: continue
        total_wr = d['sum_wr']
        total_rd = d['sum_rd']
        avg_wr = total_wr / n
        avg_rd = total_rd / n
        wr_var = (d['sum_wr_sq']/n - avg_wr**2) * n/(n-1) if n > 1 else 0
        rd_var = (d['sum_rd_sq']/n - avg_rd**2) * n/(n-1) if n > 1 else 0
        wr_std = math.sqrt(wr_var) if wr_var > 0 else 0
        rd_std = math.sqrt(rd_var) if rd_var > 0 else 0
        total_file_mib = d['sum_file_bytes'] / (1024*1024)
        avg_file_mib = total_file_mib / n
        orig_mib = first_orig if first_orig > 0 else avg_file_mib
        rat = (n * orig_mib) / total_file_mib if total_file_mib > 0 else 0
        wr_mibps = (n * orig_mib) / (total_wr / 1000.0) if total_wr > 0 else 0
        rd_mibps = (n * orig_mib) / (total_rd / 1000.0) if total_rd > 0 else 0
        avg_comp = d['sum_comp'] / n
        avg_dec = d['sum_dec'] / n
        orig_bytes = orig_mib * 1024 * 1024
        cgbps = orig_bytes / 1e9 / (avg_comp / 1000.0) if avg_comp > 0 else 0
        dgbps = orig_bytes / 1e9 / (avg_dec / 1000.0) if avg_dec > 0 else 0
        sgd=d['sum_sgd']/n; expl=d['sum_expl']/n; nch=d['n_chunks']
        nn=d['sum_nn']/n; st=d['sum_stats']/n; pre=d['sum_pre']/n
        exms=d['sum_expl_ms']/n; sgms=d['sum_sgd_ms']/n
        mr=min(200,d['sum_mape_r']/n); mc=min(200,d['sum_mape_c']/n)
        md=min(200,d['sum_mape_d']/n); mp=min(200,d['sum_mape_p']/n)
        ar=d['sum_mae_r']/n; ac=d['sum_mae_c']/n
        ad=d['sum_mae_d']/n; ap=d['sum_mae_p']/n
        f.write(f'-1,vpic,{p},{n},{avg_wr:.2f},{wr_std:.2f},{avg_rd:.2f},{rd_std:.2f},'
                f'{avg_file_mib:.2f},{orig_mib:.2f},{rat:.4f},{wr_mibps:.1f},{rd_mibps:.1f},0,'
                f'{sgd:.0f},{expl:.0f},{nch},'
                f'{nn:.2f},{st:.2f},{pre:.2f},'
                f'{avg_comp:.2f},{avg_dec:.2f},{exms:.2f},{sgms:.2f},'
                f'{cgbps:.4f},{dgbps:.4f},'
                f'{mr:.2f},{mc:.2f},{md:.2f},{mp:.2f},'
                f'{ar:.4f},{ac:.4f},{ad:.4f},{ap:.4f}\n')
print(f'  Generated {len(phases)} phase summaries')
"
    fi
done

echo ""
echo ">>> Step 2: Generating multi-rank aggregate throughput..."

MERGED_TS="$RESULTS/benchmark_vpic_deck_timesteps.csv"
AGG_MULTI="$RESULTS/benchmark_vpic_deck_aggregate_multi_rank.csv"
if [ -f "$MERGED_TS" ]; then
    python3 -c "
import csv, math

rows = list(csv.DictReader(open('$MERGED_TS')))
if not rows:
    exit()

n_ranks = len(set(r['rank'] for r in rows))

groups = {}
for r in rows:
    key = (r['phase'], r['timestep'])
    if key not in groups:
        groups[key] = []
    groups[key].append(r)

phases = {}
for (phase, ts), ranks in groups.items():
    if phase not in phases:
        phases[phase] = {'n_ts': 0,
                         'sum_agg_wr_mibps': 0, 'sum_agg_rd_mibps': 0,
                         'sum_agg_wr_sq': 0, 'sum_agg_rd_sq': 0,
                         'sum_orig_mib': 0, 'sum_file_mib': 0,
                         'sum_max_wr_ms': 0, 'sum_max_rd_ms': 0,
                         'avg_per_rank_wr': 0, 'avg_per_rank_rd': 0,
                         'n_per_rank': 0}
    d = phases[phase]

    total_orig = sum(float(r.get('orig_mib', 0)) for r in ranks)
    total_file = sum(int(r.get('file_bytes', 0)) for r in ranks) / (1024*1024)
    max_wr = max(float(r.get('write_ms', 0)) for r in ranks)
    max_rd = max(float(r.get('read_ms', 0)) for r in ranks)

    agg_wr = total_orig / (max_wr / 1000.0) if max_wr > 0 else 0
    agg_rd = total_orig / (max_rd / 1000.0) if max_rd > 0 else 0

    d['n_ts'] += 1
    d['sum_agg_wr_mibps'] += agg_wr
    d['sum_agg_rd_mibps'] += agg_rd
    d['sum_agg_wr_sq'] += agg_wr * agg_wr
    d['sum_agg_rd_sq'] += agg_rd * agg_rd
    d['sum_orig_mib'] += total_orig
    d['sum_file_mib'] += total_file
    d['sum_max_wr_ms'] += max_wr
    d['sum_max_rd_ms'] += max_rd

    for r in ranks:
        d['avg_per_rank_wr'] += float(r.get('write_mibps', 0))
        d['avg_per_rank_rd'] += float(r.get('read_mibps', 0))
        d['n_per_rank'] += 1

with open('$AGG_MULTI', 'w') as f:
    f.write('n_ranks,phase,n_timesteps,'
            'avg_agg_write_mibps,avg_agg_read_mibps,'
            'agg_write_mibps_std,agg_read_mibps_std,'
            'avg_per_rank_write_mibps,avg_per_rank_read_mibps,'
            'avg_ratio,total_orig_mib,total_file_mib,'
            'avg_max_write_ms,avg_max_read_ms\n')
    for phase in phases:
        d = phases[phase]
        n = d['n_ts']
        if n == 0:
            continue
        avg_wr = d['sum_agg_wr_mibps'] / n
        avg_rd = d['sum_agg_rd_mibps'] / n
        wr_var = (d['sum_agg_wr_sq']/n - avg_wr**2) * n/(n-1) if n > 1 else 0
        rd_var = (d['sum_agg_rd_sq']/n - avg_rd**2) * n/(n-1) if n > 1 else 0
        wr_std = math.sqrt(wr_var) if wr_var > 0 else 0
        rd_std = math.sqrt(rd_var) if rd_var > 0 else 0
        orig = d['sum_orig_mib'] / n
        fmib = d['sum_file_mib'] / n
        ratio = orig / fmib if fmib > 0 else 0
        pr_wr = d['avg_per_rank_wr'] / d['n_per_rank'] if d['n_per_rank'] > 0 else 0
        pr_rd = d['avg_per_rank_rd'] / d['n_per_rank'] if d['n_per_rank'] > 0 else 0
        avg_mwr = d['sum_max_wr_ms'] / n
        avg_mrd = d['sum_max_rd_ms'] / n
        f.write(f'{n_ranks},{phase},{n},'
                f'{avg_wr:.1f},{avg_rd:.1f},'
                f'{wr_std:.1f},{rd_std:.1f},'
                f'{pr_wr:.1f},{pr_rd:.1f},'
                f'{ratio:.4f},{orig:.2f},{fmib:.2f},'
                f'{avg_mwr:.2f},{avg_mrd:.2f}\n')

print(f'  Aggregate: {n_ranks} ranks, {len(phases)} phases -> $AGG_MULTI')
"
fi

echo ""
echo ">>> Step 3: Generating plots..."

for pol in balanced ratio; do
    case "$pol" in
        balanced) L="balanced_w1-1-1" ;;
        ratio)    L="ratio_only_w0-0-1" ;;
    esac

    echo "  Plotting $L ..."
    VPIC_DIR="$RESULTS/$L" python3 benchmarks/plots/generate_dataset_figures.py \
        --dataset vpic --policy "$L"
done

echo ""
echo ">>> Done. Figures in: benchmarks/results/per_dataset/vpic/"
