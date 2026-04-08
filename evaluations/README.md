# Paper evaluation scripts

Clean entry points for the paper's GPUCompress evaluations. Each
subdirectory is a self-contained experiment with its own orchestrator,
per-workload runners, plotter, and Delta slurm submission wrapper.

## Layout

```
evaluations/
├── figure6_threshold_sweep/   — Figure 6: SGD MAPE × exploration threshold sensitivity
│   ├── eval_vpic.sh           — per-workload inner loop (live VPIC per cell)
│   ├── eval_warpx.sh          — per-workload inner loop (live WarpX per cell)
│   ├── eval_lammps.sh         — per-workload inner loop (live LAMMPS per cell)
│   ├── eval_nyx.sh            — per-workload inner loop (Nyx dump-once + replay)
│   ├── run_all.sh             — orchestrator (calls eval_{vpic,warpx,lammps,nyx}.sh)
│   ├── plot.py                — renders 9 heatmaps per workload
│   └── submit.sbatch          — Delta slurm wrapper (A100, 4h wall)
│
└── sc26/                      — Section 7.1 cross-workload regret convergence
    ├── run_one_workload.sh    — per-(workload, policy) cell worker
    ├── run_all.sh             — orchestrator (4 workloads × 2 policies = 8 cells)
    ├── plot.py                — combined plotter (Figures 1/2/3, per policy)
    └── submit.sbatch          — Delta slurm wrapper (A100, 3h wall)
```

## figure6_threshold_sweep — what it measures

2D sweep over the online-learning thresholds:
- **X1**: `sgd_mape` — MAPE threshold above which SGD fires
- **X2 = X1 + delta**: `explore_thresh` — MAPE threshold above which exploration fires

Grid: `X1 ∈ {0.05, 0.10, 0.20, 0.30, 0.50, 1.00, 10.00}` × same values for delta → **7×7 = 49 cells per workload**, 4 workloads, 196 simulations total (+ one shared Nyx dump).

Per cell, each script executes its live simulator once, runs GPUCompress
with that cell's (`sgd_mape`, `explore_thresh`), and writes per-chunk CSVs
under `benchmarks/Paper_Evaluations/4/results/<workload>_threshold_sweep_<policy>_<eb>_lr<lr>/x1_<x1>_delta_<delta>/`.

After all cells finish, each workload's results dir is rendered into 9 heatmaps
(regret, MAPE ratio/comp/decomp/psnr, write/read BW, explorations, total SGD samples).

### Submit
```bash
sbatch evaluations/figure6_threshold_sweep/submit.sbatch
```

### Env knobs (forwarded to all 4 workloads)
- `WORKLOADS="vpic warpx lammps nyx"` — subset selection
- `POLICY=balanced` — cost policy (`balanced|ratio|speed`)
- `SGD_LR=0.2` `EXPLORE_K=4`
- `DRY_RUN=1` — skip simulator execution; exercise plotter pipeline
- Per-workload sizing: `VPIC_NX`, `WARPX_MAX_STEP`, `LMP_ATOMS`, `NYX_NCELL`, `CHUNK_MB`, `ERROR_BOUND`, ...

## sc26 — what it measures

Single-point cross-workload comparison at fixed equalized hyperparameters:
`PHASE=nn-rl+exp50 ERROR_BOUND=0.01 SGD_LR=0.2 SGD_MAPE=0.10 EXPLORE_K=4 EXPLORE_THRESH=0.25`.

Runs all 4 workloads × 2 policies (`balanced`, `ratio`) = 8 sequential cells, each
producing regret + cost MAPE + PSNR MAPE convergence curves. The combined plotter
then renders three paper figures per policy:

- **Figure 1** — cost model MAPE convergence (all 4 workloads on one plot)
- **Figure 2** — top-1 regret convergence (all 4 workloads on one plot)
- **Figure 3** — per-metric MAPE breakdown bar chart (comp time, decomp time, ratio, PSNR)

Combined figures land in `deleteAfterwards/sc26_allworkloads/figures/sc26_<policy>_<fig>.png`.

### Submit
```bash
sbatch evaluations/figure8_cross_workload_regret/submit.sbatch
```
