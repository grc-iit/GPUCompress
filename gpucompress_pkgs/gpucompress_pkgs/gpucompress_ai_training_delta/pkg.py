"""
gpucompress_ai_training_delta — AI training checkpoint compression benchmark.

Jarvis Path B (install_manager: container, container_engine: apptainer).
Requires `gpucompress_base` to precede this package in the pipeline YAML so
that HDF5, nvcomp, the GPUCompress library/weights, and the training scripts
are present in the shared build image.

Workload mirrors bench_tests/ai_training.sh combined with
scripts/train_and_export_checkpoints.py:

  Phase 1 — Train the chosen model (resnet18 / vit_b_16 / vit_l_16 / gpt2)
            and export per-epoch checkpoints as .f32 files. Each checkpoint
            emits 4 tensor types (weights, adam_m, adam_v, gradients).
  Phase 2 — Flatten those files into a single directory, then run
            generic_benchmark with the requested HDF5 mode / phase / policy.

The smoke-YAML default is ResNet-18 + CIFAR-10, matching the AI-workload
figures referenced in commits 2f50e46 / 16fddfd (40 epochs × 4 tensors ×
short-horizon batches, producing the paper's cost_mape / regret plots).

Unlike the physics sims (VPIC/Nyx/WarpX), Phase 1 here is Python training
rather than a native simulation, and Phase 1 can be skipped entirely via
the `skip_training` config flag to reuse an existing checkpoint directory.
"""
import os
import math
import glob

from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, MpiExecInfo
from jarvis_cd.shell.process import Mkdir, Rm


POLICY_WEIGHTS = {
    'ratio':    (0.0, 0.0, 1.0),
    'speed':    (1.0, 1.0, 0.0),
    'balanced': (1.0, 1.0, 1.0),
}

VALID_PHASES = {
    'lz4', 'snappy', 'deflate', 'gdeflate', 'zstd',
    'ans', 'cascaded', 'bitcomp',
    'nn', 'nn-rl', 'nn-rl+exp50',
}

VALID_MODELS = {'resnet18', 'vit_b_16', 'vit_l_16', 'gpt2'}
VALID_DATASETS = {'cifar10', 'wikitext2'}

# Absolute paths inside the built image. Must match build.sh / Dockerfile.deploy.
TRAIN_VIT_SCRIPT  = '/opt/GPUCompress/scripts/train_and_export_checkpoints.py'
TRAIN_GPT2_SCRIPT = '/opt/GPUCompress/scripts/train_gpt2_checkpoints.py'
GENERIC_BIN       = '/opt/GPUCompress/build/generic_benchmark'
WEIGHTS           = '/opt/GPUCompress/neural_net/weights/model.nnwt'

# Runtime LD_LIBRARY_PATH — Jarvis's auto-generated %environment sets
# /opt/<pkg>/install/lib (doesn't exist in our SIF), so we prefix each Exec
# command with `env LD_LIBRARY_PATH=…` to override at exec time. Mirrors the
# gpucompress_nyx_delta / gpucompress_warpx_delta pattern.
LD_LIBRARY_PATH = (
    '/.singularity.d/libs'              # host libcuda.so.1 bound via --nv
    ':/usr/local/cuda/lib64'            # CUDA runtime libs
    ':/opt/hdf5-install/lib'            # HDF5 2.0.0
    ':/opt/nvcomp/lib'                  # nvcomp
    ':/opt/GPUCompress/build'           # libgpucompress + VOL/Filter .so
)


class GpucompressAiTrainingDelta(Application):

    def _init(self):
        pass

    def _configure_menu(self):
        return [
            # ─ Model / dataset choice ─────────────────────────────────
            {'name': 'model',
             'msg': 'Model: resnet18 | vit_b_16 | vit_l_16 | gpt2',
             'type': str, 'default': 'resnet18'},
            {'name': 'dataset',
             'msg': 'Dataset: cifar10 (ViT/ResNet) | wikitext2 (GPT-2)',
             'type': str, 'default': 'cifar10'},

            # ─ Training knobs (Phase 1) ───────────────────────────────
            # Defaults mirror the paper's AI run (commits 2f50e46 / 16fddfd):
            # ResNet-18, short-horizon training with a small max-batches-per-epoch
            # cap so each "epoch" produces a checkpoint quickly.
            {'name': 'epochs',
             'msg': 'Training epochs',
             'type': int, 'default': 40},
            {'name': 'checkpoint_epochs',
             'msg': 'Comma-separated epochs to export checkpoints',
             'type': str, 'default': '1,5,10,15,20,25,30,35,40'},
            {'name': 'batch_size',
             'msg': 'Training batch size',
             'type': int, 'default': 64},
            {'name': 'max_batches_per_epoch',
             'msg': 'Cap batches/epoch for short-horizon benchmarking '
                    '(0 = uncapped, matches production training)',
             'type': int, 'default': 20},
            {'name': 'skip_validate',
             'msg': 'Skip per-epoch validation pass (1 = skip, faster)',
             'type': int, 'default': 1},
            {'name': 'skip_training',
             'msg': 'Skip Phase 1 and reuse existing checkpoints under '
                    '{results_dir}/checkpoints/ (1 = skip)',
             'type': int, 'default': 0},

            # ─ HDF5 mode / algorithm / policy (Phase 2) ───────────────
            {'name': 'hdf5_mode',
             'msg': "HDF5 mode: 'default' (no-comp baseline) or 'vol' "
                    "(GPUCompress VOL)",
             'type': str, 'default': 'default'},
            {'name': 'phase',
             'msg': 'Compression phase (vol mode only): '
                    'lz4|snappy|deflate|gdeflate|zstd|ans|cascaded|bitcomp'
                    '|nn|nn-rl|nn-rl+exp50',
             'type': str, 'default': 'lz4'},
            {'name': 'policy',
             'msg': 'NN cost-model policy: balanced | ratio | speed',
             'type': str, 'default': 'balanced'},
            {'name': 'error_bound',
             'msg': 'Lossy error bound (0.0 = lossless; '
                    'inference deployment: 1e-4 – 1e-3)',
             'type': float, 'default': 0.0},

            # ─ Benchmark knobs ────────────────────────────────────────
            {'name': 'chunk_mb',
             'msg': 'HDF5 chunk size (MB)',
             'type': int, 'default': 4},
            {'name': 'verify',
             'msg': 'Bitwise readback verify: 1 = on, 0 = off',
             'type': int, 'default': 1},

            # ─ NN online learning (Phase 2 knobs) ─────────────────────
            {'name': 'sgd_lr',
             'msg': 'SGD learning rate',
             'type': float, 'default': 0.2},
            {'name': 'sgd_mape',
             'msg': 'MAPE threshold for SGD firing',
             'type': float, 'default': 0.10},
            {'name': 'explore_k',
             'msg': 'Top-K exploration alternatives',
             'type': int, 'default': 4},
            {'name': 'explore_thresh',
             'msg': 'Exploration error threshold',
             'type': float, 'default': 0.20},

            # ─ Container build options ────────────────────────────────
            {'name': 'cuda_arch',
             'msg': 'CUDA compute capability (must match gpucompress_base)',
             'type': int, 'default': 80},
            {'name': 'deploy_base',
             'msg': 'Base image for the deploy stage',
             'type': str,
             'default': 'nvidia/cuda:12.6.0-runtime-ubuntu24.04'},
            {'name': 'use_gpu',
             'msg': 'Pass --nv to apptainer at run time',
             'type': bool, 'default': True},
            {'name': 'num_gpus',
             'msg': 'Number of visible GPUs (script uses cuda:0 by default)',
             'type': int, 'default': 1},

            # ─ Runtime output path ────────────────────────────────────
            {'name': 'results_dir',
             'msg': 'Output root (empty = /tmp/gpucompress_ai_training_<pkg_id>_…)',
             'type': str, 'default': ''},
        ]

    def _configure(self, **kwargs):
        pass

    # ------------------------------------------------------------------
    # Container build hooks (Jarvis Path B)
    # ------------------------------------------------------------------

    def _build_phase(self):
        if self.config.get('deploy_mode') != 'container':
            return None
        content = self._read_build_script('build.sh', {
            'CUDA_ARCH': str(self.config['cuda_arch']),
        })
        suffix = f"cuda-{self.config['cuda_arch']}"
        return content, suffix

    def _build_deploy_phase(self):
        if self.config.get('deploy_mode') != 'container':
            return None
        suffix = getattr(self, '_build_suffix', '')
        content = self._read_dockerfile('Dockerfile.deploy', {
            'BUILD_IMAGE': self.build_image_name(),
            'DEPLOY_BASE': self.config['deploy_base'],
        })
        return content, suffix

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        cfg = self.config
        self._validate(cfg)

        verify = int(cfg['verify'])
        error_bound = float(cfg['error_bound'])
        lossy = error_bound not in (0, 0.0)
        verify_tag = '_noverify' if verify == 0 else ''
        eb_tag = f'_lossy{error_bound}' if lossy else ''

        # Directory naming mirrors bench_tests/ai_training.sh:
        #   vit_b_16 + cifar10  → vit_b_cifar10
        #   vit_l_16 + cifar10  → vit_l_cifar10
        #   resnet18 + cifar10  → resnet18_cifar10
        #   gpt2 + wikitext2    → gpt2_wikitext2
        model = cfg['model']
        dataset = cfg['dataset']
        if model.startswith('vit_'):
            ai_short = f"vit_{model.split('_')[1]}"
        else:
            ai_short = model
        ai_dir_name = f"{ai_short}_{dataset}"

        results_dir = cfg['results_dir'] or (
            f"/tmp/gpucompress_ai_training_{self.pkg_id}"
            f"_{ai_dir_name}_ep{cfg['epochs']}_{cfg['hdf5_mode']}"
            f"{eb_tag}{verify_tag}"
        )
        ckpt_dir    = f'{results_dir}/checkpoints'
        flat_dir    = f'{results_dir}/flat_fields'
        data_root   = f'{results_dir}/dataset_cache'
        bench_phase = 'no-comp' if cfg['hdf5_mode'] == 'default' else cfg['phase']
        bench_dir   = f'{results_dir}/{cfg["hdf5_mode"]}_{bench_phase}'
        for d in (results_dir, ckpt_dir, flat_dir, data_root, bench_dir):
            Mkdir(d).run()

        train_log = f'{results_dir}/train.log'
        bench_log = f'{bench_dir}/ai_bench.log'

        w0, w1, w2 = POLICY_WEIGHTS[cfg['policy']]

        # ── Phase 1: training → export .f32 checkpoints ─────────────────
        if not int(cfg['skip_training']):
            script = TRAIN_GPT2_SCRIPT if model == 'gpt2' else TRAIN_VIT_SCRIPT
            train_parts = [f'python3 {script}']
            # GPT-2 script has a fixed model + dataset; no --model/--dataset args.
            if model != 'gpt2':
                train_parts += [
                    f'--model {model}',
                    f'--dataset {dataset}',
                ]
            train_parts += [
                f'--epochs {cfg["epochs"]}',
                f'--checkpoint-epochs {cfg["checkpoint_epochs"]}',
                f'--batch-size {cfg["batch_size"]}',
                '--num-workers 0',
                f'--outdir {ckpt_dir}',
                f'--data-root {data_root}',
            ]
            # NOTE: deliberately NOT passing --hdf5-direct. With that flag the
            # training script writes pre-compressed .h5 files through the VOL,
            # which defeats Phase 2 (generic_benchmark) since it would then be
            # measuring re-compression of already-compressed data. We want raw
            # .f32 checkpoints so Phase 2 can fairly benchmark every algorithm.
            # The in-situ training-time VOL measurement is a separate
            # experiment (commits 2f50e46 / 16fddfd) — not what our
            # Phase 1 + Phase 2 pipeline is for.
            mbpe = int(cfg['max_batches_per_epoch'])
            if mbpe > 0 and model != 'gpt2':
                train_parts.append(f'--max-batches-per-epoch {mbpe}')
            if int(cfg['skip_validate']) and model != 'gpt2':
                train_parts.append('--no-validate')
            train_cmd = ' '.join(train_parts)

            train_env = dict(self.mod_env)
            # Point the training script at the in-container GPUCompress module
            # (gpucompress_hdf5.py etc.) via PYTHONPATH; Dockerfile.deploy already
            # exports this, but Jarvis's mod_env may override — reassert here.
            train_env['PYTHONPATH'] = '/opt/GPUCompress/scripts:/opt/GPUCompress'
            train_env['GPUCOMPRESS_WEIGHTS'] = WEIGHTS
            train_env['HDF5_PLUGIN_PATH']    = '/opt/GPUCompress/build'

            phase1 = Exec(
                f'env LD_LIBRARY_PATH={LD_LIBRARY_PATH} {train_cmd}',
                MpiExecInfo(
                nprocs=1,
                ppn=1,
                hostfile=self.hostfile,
                port=self.ssh_port,
                container=self._container_engine,
                container_image=self.deploy_image_name(),
                shared_dir=self.shared_dir,
                private_dir=self.private_dir,
                env=train_env,
                gpu=cfg.get('use_gpu', True),
                pipe_stdout=train_log,
                pipe_stderr=train_log,
            ))
            phase1.run()
            p1_rc = max(phase1.exit_code.values()) if getattr(phase1, 'exit_code', None) else 0
            if p1_rc != 0:
                raise RuntimeError(
                    f"Training failed (exit {p1_rc}) — see {train_log}"
                )

        # ── Host-side: locate exported .f32 files, compute 2-D dims ────
        # The training script writes checkpoints under ckpt_dir as a mix of
        # flat .f32 files and subdirs per-epoch; recurse so we catch both.
        ckpt_files = sorted(glob.glob(f'{ckpt_dir}/**/*.f32', recursive=True))
        if not ckpt_files:
            raise RuntimeError(
                f"No .f32 checkpoint files under {ckpt_dir} — "
                f"{'skip_training=1 but directory is empty' if int(cfg['skip_training']) else 'check ' + train_log}"
            )

        n_floats = os.path.getsize(ckpt_files[0]) // 4

        # 2-D dims via largest-square factorization — mirrors reference shell
        # (bench_tests/ai_training.sh:165-172). Produces roughly-square chunks
        # that give the NN meaningful spatial context for ranking.
        dim0 = int(math.isqrt(n_floats))
        while dim0 > 1 and n_floats % dim0 != 0:
            dim0 -= 1
        dim1 = n_floats // dim0
        dims = f'{dim0},{dim1}'

        # Flatten all .f32 checkpoints into a single directory with symlinks
        # so generic_benchmark sees one flat chunk stream (same pattern as
        # Nyx / WarpX / VPIC Phase-2 driver).
        for src in ckpt_files:
            rel = os.path.relpath(src, ckpt_dir).replace('/', '_')
            link = f'{flat_dir}/{rel}'
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(src, link)

        # ── Phase 2: generic_benchmark ──────────────────────────────────
        # Arg layout matches bench_tests/ai_training.sh lines 211-225 exactly,
        # plus the SGD / explore knobs exposed as YAML-tunable (defaults equal
        # reference hardcoded values).
        bench_parts = [
            GENERIC_BIN,
            WEIGHTS,
            f'--data-dir {flat_dir}',
            f'--dims {dims}',
            '--ext .f32',
            f'--chunk-mb {cfg["chunk_mb"]}',
            f'--name {ai_dir_name}',
        ]
        if lossy:
            bench_parts.append(f'--error-bound {error_bound}')
        if verify == 0:
            bench_parts.append('--no-verify')
        bench_parts += [
            f'--phase {bench_phase}',
            f'--w0 {w0} --w1 {w1} --w2 {w2}',
            f'--lr {cfg["sgd_lr"]} --mape {cfg["sgd_mape"]}',
            f'--explore-k {cfg["explore_k"]} --explore-thresh {cfg["explore_thresh"]}',
            f'--out-dir {bench_dir}',
        ]
        bench_cmd = ' '.join(bench_parts)

        bench_env = dict(self.mod_env)
        bench_env['GPUCOMPRESS_DETAILED_TIMING'] = '1'
        phase2 = Exec(
            f'env LD_LIBRARY_PATH={LD_LIBRARY_PATH} {bench_cmd}',
            MpiExecInfo(
            nprocs=1,
            ppn=1,
            hostfile=self.hostfile,
            port=self.ssh_port,
            container=self._container_engine,
            container_image=self.deploy_image_name(),
            shared_dir=self.shared_dir,
            private_dir=self.private_dir,
            env=bench_env,
            gpu=cfg.get('use_gpu', True),
            pipe_stdout=bench_log,
            pipe_stderr=bench_log,
        ))
        phase2.run()
        p2_rc = max(phase2.exit_code.values()) if getattr(phase2, 'exit_code', None) else 0
        if p2_rc != 0:
            raise RuntimeError(
                f"generic_benchmark failed (exit {p2_rc}) — see {bench_log}"
            )

    def stop(self):
        pass

    def clean(self):
        results_dir = self.config.get('results_dir')
        if results_dir and os.path.isdir(results_dir):
            Rm(results_dir).run()
        # Also sweep the auto-generated default-path variants created by start()
        Rm(f'/tmp/gpucompress_ai_training_{self.pkg_id}_*').run()

    # ------------------------------------------------------------------
    def _validate(self, cfg):
        if cfg['model'] not in VALID_MODELS:
            raise ValueError(
                f"model must be one of {sorted(VALID_MODELS)}, got {cfg['model']!r}"
            )
        if cfg['dataset'] not in VALID_DATASETS:
            raise ValueError(
                f"dataset must be one of {sorted(VALID_DATASETS)}, got {cfg['dataset']!r}"
            )
        # Script-level constraints: GPT-2 is fixed to wikitext2; ViT/ResNet are
        # fixed to cifar10 in the current training scripts.
        if cfg['model'] == 'gpt2' and cfg['dataset'] != 'wikitext2':
            raise ValueError("model=gpt2 requires dataset=wikitext2")
        if cfg['model'] != 'gpt2' and cfg['dataset'] != 'cifar10':
            raise ValueError(f"model={cfg['model']} requires dataset=cifar10")
        if cfg['hdf5_mode'] not in ('default', 'vol'):
            raise ValueError(
                f"hdf5_mode must be 'default' or 'vol', got {cfg['hdf5_mode']!r}"
            )
        if cfg['hdf5_mode'] == 'vol' and cfg['phase'] not in VALID_PHASES:
            raise ValueError(
                f"phase {cfg['phase']!r} not in {sorted(VALID_PHASES)}"
            )
        if cfg['policy'] not in POLICY_WEIGHTS:
            raise ValueError(
                f"policy must be one of {sorted(POLICY_WEIGHTS)}, "
                f"got {cfg['policy']!r}"
            )
