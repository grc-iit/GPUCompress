# gpucompress_nyx_delta

Jarvis Path-B package that builds the **Nyx AMR astrophysics simulation**
(Sedov blast wave) against NeuroPress and runs it inside an Apptainer SIF
on NCSA Delta A100 nodes.

Nyx is a **two-phase** workload:

- **Phase 1** — `nyx_HydroTests` runs the Sedov blast with
  `NYX_DUMP_FIELDS=1`, writing raw `.f32` files (one per field component
  per FAB per plotfile) into `<results>/raw_fields/plt*/`.
- **Phase 2** — the pkg flattens those dumps into a single directory via
  symlinks and invokes `generic_benchmark` with the requested HDF5 mode /
  phase / policy, producing ranking + cost CSVs.

**Pipeline YAMLs:**
- `gpucompress_pkgs/pipelines/gpucompress_nyx_single_node.yaml` — adaptive (VOL + NN)
- `gpucompress_pkgs/pipelines/gpucompress_nyx_single_node_baseline.yaml` — no-comp

---

## Prerequisites

- Delta account with an A100 allocation (set `$SLURM_ACCOUNT` to your own)
- Jarvis-CD installed and on `$PATH` (checked via `which jarvis`)
- ~10 GB of free quota under `/u/$USER/` (SIF ~6-10 GB + build artefacts)
- This repo cloned to `/u/$USER/GPUCompress` (paths in the YAML assume that)

---

## End-to-end, from zero (copy-paste ready)

Four phases: **clean → build → allocate → run+archive**.
`dt-loginNN` = login node (no GPU), `gpuaNNN` = compute node (A100).

### Phase 0 — Clean state (login node)

```bash
# Scan for stray CSVs first — never delete these
find ~/.jarvis-cd/pipelines/gpucompress_nyx_delta_single_node -name '*.csv' 2>/dev/null
# If anything prints, move it aside before continuing.

rm -rf ~/.jarvis-cd/pipelines/gpucompress_nyx_delta_single_node/
rm -rf ~/.ppi-jarvis/shared/gpucompress_nyx_delta_single_node/
rm -rf ~/.ppi-jarvis/config/pipelines/gpucompress_nyx_delta_single_node/
rm -f  ~/build_nyx.log ~/run_nyx_*.log

# Jarvis validates hostfile: at load time — create a placeholder on login.
# The real compute-node hostname gets written in Phase 3.
echo "$(hostname)" > ~/hostfile_single.txt
```

Only wipe the apptainer cache (`apptainer cache clean --force`) if you
need to force a cold rebuild of `gpucompress_base`. Otherwise the Nyx
build reuses the base layer from any earlier workload and finishes in
~5-10 min.

### Phase 1 — Build the SIF (login node)

`jarvis ppl load yaml <path>` builds the SIF as part of loading when
`install_manager: container` is set in the YAML (see
`jarvis_cd/core/pipeline.py` — `_load_from_file` calls
`_build_pipeline_container()` at line 1024). There is no separate
`jarvis ppl build` step.

```bash
cd /u/$USER/GPUCompress/gpucompress_pkgs
jarvis repo add /u/$USER/GPUCompress/gpucompress_pkgs 2>/dev/null || true
jarvis ppl load yaml /u/$USER/GPUCompress/gpucompress_pkgs/pipelines/gpucompress_nyx_single_node.yaml 2>&1 | tee ~/build_nyx.log

# Verify the SIF was produced
ls -la ~/.ppi-jarvis/shared/gpucompress_nyx_delta_single_node/*.sif
```

Expected tail of log:
```
Apptainer SIF ready: .../gpucompress_nyx_delta_single_node.sif
Loaded pipeline: gpucompress_nyx_delta_single_node
```

Subsequent loads print `Deploy image '<name>' already exists, skipping
build` and return in seconds.

### Phase 2 — Get a GPU node

```bash
salloc --account="$SLURM_ACCOUNT" --partition=gpuA100x4 \
       --nodes=1 --ntasks=1 --gpus=1 --time=01:00:00 --mem=64g
```

Wait until the shell prompt changes from `dt-loginNN` to `gpuaNNN`. If it
still says `dt-login`, the allocation is pending — `squeue -u $USER` to
check. **Do not run the pipeline on a login node — it has no GPU.**

### Phase 3 — Run on the compute node

```bash
# Sanity
hostname          # must start with "gpua"
nvidia-smi        # must list an A100

# Refresh BOTH hostfiles: the global one AND the per-pipeline cached copy
# (jarvis hostfile set does NOT propagate to the per-pipeline copy)
hostname > ~/hostfile_single.txt
jarvis hostfile set ~/hostfile_single.txt
hostname > ~/.ppi-jarvis/shared/gpucompress_nyx_delta_single_node/hostfile

# Enter pipeline
jarvis cd gpucompress_nyx_delta_single_node

# Adaptive run — hdf5_mode=vol, phase=nn-rl+exp50, policy=ratio, lossless
jarvis ppl run 2>&1 | tee ~/run_nyx_adaptive.log

# Baseline run — reload the sibling baseline YAML (hdf5_mode=default).
# Editing YAML + reloading is the canonical way to switch configs.
# Avoid `jarvis pkg conf` overrides: multi-key calls are not atomic and
# leave the persisted config in inconsistent states.
jarvis ppl load yaml /u/$USER/GPUCompress/gpucompress_pkgs/pipelines/gpucompress_nyx_single_node_baseline.yaml
hostname > ~/.ppi-jarvis/shared/gpucompress_nyx_delta_single_node/hostfile
jarvis cd gpucompress_nyx_delta_single_node
jarvis ppl run 2>&1 | tee ~/run_nyx_baseline.log

# To go back to adaptive, reload the primary YAML:
jarvis ppl load yaml /u/$USER/GPUCompress/gpucompress_pkgs/pipelines/gpucompress_nyx_single_node.yaml
```

### Phase 4 — Archive results BEFORE releasing the allocation

`/tmp` is per-node and vanishes when the allocation ends.

```bash
STAMP=$(date +%Y%m%d_%H%M)
OUT=~/nyx_results_${STAMP}
mkdir -p "$OUT"
cp -r /tmp/gpucompress_nyx_*_vol      "$OUT/" 2>/dev/null
cp -r /tmp/gpucompress_nyx_*_default  "$OUT/" 2>/dev/null
cp    ~/run_nyx_adaptive.log  ~/run_nyx_baseline.log  "$OUT/"

ls -la "$OUT/"
du -sh "$OUT"

exit   # release the allocation
```

---

## Configuration reference

All knobs live in `pkg.py:_configure_menu()`. **Configure them by editing
the pipeline YAML and re-running `jarvis ppl load yaml <path>`.** This is
the canonical workflow.

`jarvis pkg conf gpuc_nyx <k>=<v>` also exists but is not recommended —
multi-key invocations are not atomic, and the persisted config can drift
out of sync with the source YAML, producing confusing validation failures
at the next `ppl run`. Edit the YAML, reload, done.

For repeatable variants (baseline vs adaptive, policy sweeps, etc.),
create sibling YAMLs with descriptive names and switch between them by
reloading.

### HDF5 mode / algorithm / policy

| Key | Default | Meaning |
|---|---|---|
| `hdf5_mode` | `default` | `default` (no-comp baseline) or `vol` (NeuroPress VOL) |
| `phase` | `lz4` | `lz4`/`snappy`/`deflate`/`gdeflate`/`zstd`/`ans`/`cascaded`/`bitcomp` (fixed) or `nn`/`nn-rl`/`nn-rl+exp50` (adaptive) |
| `policy` | `balanced` | NN cost-model weights: `balanced` (w=1,1,1), `ratio` (0,0,1), `speed` (1,1,0) |
| `error_bound` | `0.0` | Lossy tolerance; `0.0` = lossless |

### I/O volume knobs

| Key | Default | Notes |
|---|---|---|
| `ncell` | `64` | Grid cells per dim; data/plotfile ≈ `ncell³ × 6 fields × 4 B` + small metadata. `ncell=64` → ~6 MB per plotfile, `ncell=128` → ~50 MB, `ncell=256` → ~400 MB |
| `max_step` | `30` | Total AMR time steps in the Sedov sim |
| `plot_int` | `10` | Steps between plotfile dumps. Number of plotfiles ≈ `max_step / plot_int + 1` |
| `chunk_mb` | `4` | HDF5 chunk size for Phase-2 replay |
| `verify` | `1` | `1` = bitwise readback verify in Phase 2, `0` = skip |

### NN online learning

| Key | Default | Purpose |
|---|---|---|
| `sgd_lr` | `0.2` | SGD learning rate (`--lr`) |
| `sgd_mape` | `0.10` | MAPE threshold for SGD firing (`--mape`) |
| `explore_k` | `4` | Top-K exploration alternatives (`--explore-k`) |
| `explore_thresh` | `0.20` | Exploration error threshold (`--explore-thresh`) |

### Container / launch

| Key | Default | Notes |
|---|---|---|
| `cuda_arch` | `80` | A100 = 80. Must match `gpucompress_base`. |
| `deploy_base` | `nvidia/cuda:12.6.0-runtime-ubuntu24.04` | Pinned to CUDA 12.6; 12.8 hits an NVCC ICE with Kokkos. |
| `use_gpu` | `True` | Adds `--nv` to the apptainer invocation |

The Sedov-blast problem setup (prob_type, r_init, exp_energy, species
fractions, boundary conditions, etc.) is hardcoded in
`pkg.py:_write_inputs_sedov()` since it is the canonical Sedov test and
does not need per-run tuning.

---

## Expected outputs

Each run writes to `/tmp/gpucompress_nyx_<pkg_id>_n<N>_ms<M>_<mode>/`:

```
<results_dir>/
├── inputs.sedov                       # generated AMReX input deck
├── nyx_sim.log                        # Phase-1 stdout+stderr
├── raw_fields/                        # Phase-1 output
│   ├── plt00000/*.f32  (per-component raw dumps)
│   ├── plt00010/*.f32
│   └── ...
├── flat_fields/                       # symlinks consumed by Phase 2
│   └── plt<NNNNN>_fab0000_comp<NN>_<name>.f32
└── <mode>_<phase>/                    # e.g. vol_nn-rl+exp50 or default_no-comp
    ├── nyx_bench.log                  # Phase-2 stdout+stderr
    ├── benchmark_nyx_n<N>.csv         # per-field summary (the primary CSV)
    ├── benchmark_nyx_n<N>_ranking.csv         # Kendall τ + regret per field
    ├── benchmark_nyx_n<N>_ranking_costs.csv   # NN-predicted vs observed costs
    └── gpucompress_vol_summary.txt    # Lifetime I/O totals, VOL timing
```

### Success indicators

- Phase 1: `>>> Phase 1: Nyx Sedov blast (<N> steps, dump every <M>)` and
  `Dumped: <D> plt* directories` with `<D> = max_step/plot_int + 1`.
- Phase 2: `Benchmark PASSED` printed near the end of `nyx_bench.log`.
- Adaptive runs on the smoke defaults produce ratios ≈ 85-267× (the
  Sedov blast is highly sparse — most cells are ambient), τ values split
  between 0.2-0.4 on dynamic fields (density, `rho_E`, `Temp`) and 0.6-0.85
  on sparse fields, regret ≈ 1.00× with the occasional ~2× spike on
  the expanding-wave-front rho_E.
- Baseline: `phase=no-comp`, ratio=1.00 across all fields.

### Benign artefacts to ignore

- `environment: line 17: /usr/share/lmod/lmod/libexec/lmod: No such file
  or directory` — harmless lmod call inside the container.
- `Unused ParmParse Variables: amr.ref_ratio, amr.regrid_int,
  amrex.the_async_arena_init_size` — AMReX warning; those keys are still
  honored when relevant, just unused at max_level=0.
- `Nyx::hydro_tile_size unset, using fabarray.mfiter_tile_size default`
  — advisory, not an error.

---

## Scaling up

| Scale | `ncell` | `max_step` | `plot_int` | Plotfiles | Data/plot | Wall (A100, 1 GPU) |
|---|---|---|---|---|---|---|
| Smoke (default) | 64 | 30 | 10 | 4 | ~6 MB | ~10-20 s |
| Small | 128 | 50 | 10 | 6 | ~50 MB | ~1-2 min |
| Medium | 192 | 100 | 20 | 6 | ~170 MB | ~10-20 min |
| Paper | 256 | 200 | 50 | 5 | ~400 MB | ~1-2 h |

Nyx is single-rank in this pipeline (the wrapper does not split the grid
across MPI ranks). For multi-rank experiments you need to adjust
`pkg.py:start()` to pass `nprocs > 1` to `MpiExecInfo` and ensure `ncell`
is divisible by the rank decomposition.

---

## Troubleshooting

### `Error: Hostfile not found: /u/$USER/hostfile_single.txt`
Jarvis validates `hostfile:` at `ppl load` time. Create a placeholder on
the login node before loading:
```bash
echo "$(hostname)" > ~/hostfile_single.txt
```

### Benchmark aborts with `cudaErrorNoDevice: no CUDA-capable device`
You're running on the login node. Check the prompt — it must say
`gpuaNNN`, not `dt-loginNN`. Re-`salloc` if needed.

### Apptainer asks for a SSH password
The per-pipeline cached hostfile still points at the old node.
`jarvis hostfile set` updates the global hostfile but does NOT propagate
to the per-pipeline copy. Overwrite it directly:
```bash
hostname > ~/.ppi-jarvis/shared/gpucompress_nyx_delta_single_node/hostfile
```

### `Pipeline startup failed at package 'gpuc_nyx'`
1. Open `/tmp/gpucompress_nyx_*/nyx_sim.log` — if Phase 1 crashed there,
   the real cause is in the AMReX trace at the top.
2. If Phase 1 says "Dumped: 0 plt* directories", the Nyx binary died
   before writing any dump. Common causes: CUDA unavailable (wrong host),
   OOM at large `ncell`, or a bad `nyx.*` override in the generated
   `inputs.sedov`.
3. If Phase 2 died, inspect `/tmp/gpucompress_nyx_*/<mode>_<phase>/nyx_bench.log`.

### `phase 'no-comp' not in [...]`
`no-comp` is not a valid `phase` value — it's an internal pseudo-phase
selected implicitly when `hdf5_mode=default`. Either:
- Use the baseline YAML variant (recommended):
  ```bash
  jarvis ppl load yaml .../gpucompress_nyx_single_node_baseline.yaml
  ```
- Or in your own YAML, set `hdf5_mode: default` and keep `phase` at any
  real value (e.g. `phase: nn-rl+exp50`) — it is ignored in default mode.

If the persisted config is stuck with `phase: no-comp`, patch it:
```bash
sed -i 's/^  phase: no-comp$/  phase: nn-rl+exp50/' \
    ~/.ppi-jarvis/config/pipelines/gpucompress_nyx_delta_single_node/pipeline.yaml
```

### Suspiciously high compression ratios (>100×)
This is a **feature of the Sedov-blast workload**, not a bug. Most cells
are at ambient conditions (outside the blast wave), making the field
arrays highly sparse. Look at the per-field rows in the ranking CSV —
dynamic fields (`density`, `rho_E`, `Temp`) show realistic 80-220×
ratios while the "empty" components (e.g. unused species slots named
`_comp08_.f32` and up) saturate near 133.85×.

### `HDF5 library does not support parallel I/O`
Base image was built without `-DHDF5_ENABLE_PARALLEL=ON`. The current
`gpucompress_base/build.sh` sets that flag; if you see this error, your
cached base image is stale — rebuild from scratch (see below).

### NVCC internal compiler error at `lexical.c:22310`
You're on CUDA 12.8 with gcc-13. The YAML pins CUDA 12.6 specifically
to avoid this. If the error appears, double-check `deploy_base` in the
YAML and the `container_base` line.

### Rebuild from scratch
If you change source under `/u/$USER/GPUCompress/` or the base image,
wipe the cached SIF + apptainer cache and reload — the load step
triggers a fresh build:
```bash
rm -rf ~/.jarvis-cd/pipelines/gpucompress_nyx_delta_single_node/
rm -rf ~/.ppi-jarvis/shared/gpucompress_nyx_delta_single_node/
rm -rf ~/.ppi-jarvis/config/pipelines/gpucompress_nyx_delta_single_node/
apptainer cache clean --force
jarvis ppl load yaml /u/$USER/GPUCompress/gpucompress_pkgs/pipelines/gpucompress_nyx_single_node.yaml
```

---

## File map

```
gpucompress_nyx_delta/
├── pkg.py               # Jarvis Application class; two-phase start/stop/clean lifecycle
├── build.sh             # apptainer fakeroot build step (clones Nyx + AMReX, patches, links NeuroPress)
├── Dockerfile.deploy    # deploy-stage template (COPY /opt + /usr/local, LD_LIBRARY_PATH)
├── __init__.py          # empty, required for Python import
└── README.md            # this file
```

Paired pipeline YAMLs:
```
gpucompress_pkgs/pipelines/gpucompress_nyx_single_node.yaml           # adaptive
gpucompress_pkgs/pipelines/gpucompress_nyx_single_node_baseline.yaml  # no-comp
```
