"""
AI Training benchmark package for GPUCompress.
Fine-tunes a ViT or ResNet model on CIFAR-10 and exports training
checkpoints through the HDF5 VOL connector with GPU-accelerated compression.
"""
from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, LocalExecInfo
from jarvis_cd.shell.process import Rm
import os


class GpucompressAiTraining(Application):
    """
    Deploy and run the AI Training GPUCompress benchmark.
    """

    def _init(self):
        pass

    def _configure_menu(self):
        return [
            {
                'name': 'nprocs',
                'msg': 'Number of MPI processes',
                'type': int,
                'default': 1,
            },
            {
                'name': 'ppn',
                'msg': 'Processes per node',
                'type': int,
                'default': 1,
            },
            {
                'name': 'gpucompress_dir',
                'msg': 'Path to GPUCompress installation',
                'type': str,
                'default': '/opt/GPUCompress',
            },
            {
                'name': 'nvcomp_lib',
                'msg': 'Path to nvcomp lib directory',
                'type': str,
                'default': '/tmp/lib',
            },
            {
                'name': 'hdf5_lib',
                'msg': 'Path to HDF5 lib directory',
                'type': str,
                'default': '/tmp/hdf5-install/lib',
            },
            {
                'name': 'weights',
                'msg': 'Path to NN weights file (.nnwt)',
                'type': str,
                'default': '/opt/GPUCompress/neural_net/weights/model.nnwt',
            },
            {
                'name': 'model',
                'msg': 'Model architecture (vit_b_16, vit_l_16, resnet18)',
                'type': str,
                'default': 'vit_b_16',
            },
            {
                'name': 'epochs',
                'msg': 'Training epochs',
                'type': int,
                'default': 5,
            },
            {
                'name': 'checkpoint_epochs',
                'msg': 'Comma-separated epochs to export checkpoints',
                'type': str,
                'default': '1,3,5',
            },
            {
                'name': 'batch_size',
                'msg': 'Training batch size',
                'type': int,
                'default': 64,
            },
            {
                'name': 'chunk_mb',
                'msg': 'HDF5 chunk size in MB',
                'type': int,
                'default': 4,
            },
            {
                'name': 'error_bound',
                'msg': 'Lossy error bound (0.0 = lossless)',
                'type': float,
                'default': 0.0,
            },
            {
                'name': 'policies',
                'msg': 'NN cost model policy (balanced, ratio, speed)',
                'type': str,
                'default': 'balanced',
            },
            {
                'name': 'hdf5_direct',
                'msg': 'Write checkpoints via HDF5 VOL (1=yes, 0=no)',
                'type': int,
                'default': 1,
            },
            {
                'name': 'results_dir',
                'msg': 'Output directory for results',
                'type': str,
                'default': '/tmp/gpucompress_ai_training_results',
            },
        ]

    def _configure(self, **kwargs):
        pass

    def start(self):
        gpuc = self.config['gpucompress_dir']
        os.makedirs(self.config['results_dir'], exist_ok=True)

        env = dict(os.environ)
        lib_paths = [
            f'{gpuc}/build',
            self.config['nvcomp_lib'],
            self.config['hdf5_lib'],
        ]
        ld = ':'.join(lib_paths)
        if 'LD_LIBRARY_PATH' in env:
            env['LD_LIBRARY_PATH'] = f'{ld}:{env["LD_LIBRARY_PATH"]}'
        else:
            env['LD_LIBRARY_PATH'] = ld
        env['GPUCOMPRESS_WEIGHTS'] = self.config['weights']
        env['HDF5_PLUGIN_PATH'] = f'{gpuc}/build'
        env['PYTHONPATH'] = f'{gpuc}/scripts:{env.get("PYTHONPATH", "")}'

        cmd = (f'python3 {gpuc}/scripts/train_and_export_checkpoints.py'
               f' --model {self.config["model"]}'
               f' --epochs {self.config["epochs"]}'
               f' --checkpoint-epochs {self.config["checkpoint_epochs"]}'
               f' --batch-size {self.config["batch_size"]}'
               f' --num-workers 0'
               f' --outdir {self.config["results_dir"]}')

        if self.config['hdf5_direct']:
            cmd += (f' --hdf5-direct'
                    f' --chunk-mb {self.config["chunk_mb"]}'
                    f' --error-bound {self.config["error_bound"]}'
                    f' --policy {self.config["policies"]}')

        Exec(cmd,
             LocalExecInfo(env=env,
                           cwd=self.config['results_dir'])).run()

    def stop(self):
        pass

    def clean(self):
        if self.config['results_dir'] and os.path.isdir(self.config['results_dir']):
            Rm(self.config['results_dir']).run()
