#!/bin/bash
# gpucompress_ai_training_delta build script — installs the Python ML stack
# (PyTorch + torchvision + HuggingFace + h5py + tqdm) into the base image so
# scripts/train_and_export_checkpoints.py and scripts/train_gpt2_checkpoints.py
# can run during Phase 1.
#
# Must run AFTER gpucompress_base in the pipeline so that /opt/GPUCompress
# (with the training scripts under scripts/) is already present.
#
# No C++ compilation — pip-only; typical run time is 3-5 min.
#
# Placeholders: (none — CUDA_ARCH is embedded in the PyTorch wheel index URL
#                below and must match the container_base CUDA version.)
set -e
export DEBIAN_FRONTEND=noninteractive

# Base image already has python3 + pip + CUDA 12.6 runtime from gpucompress_base.
# We only add ML stack on top.

# ── PyTorch + torchvision (CUDA 12.6 wheels) ──────────────────────────
# The index URL pin ensures PyTorch picks up the matching CUDA runtime from
# /usr/local/cuda rather than downloading its own. Must stay in sync with
# container_base (see YAML: nvidia/cuda:12.6.0-devel-ubuntu24.04).
pip3 install --break-system-packages --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126

# ── HDF5 bindings + HuggingFace stack (for GPT-2 path) ────────────────
pip3 install --break-system-packages --no-cache-dir \
    h5py \
    transformers \
    datasets \
    tqdm \
    numpy

# ── Sanity check — training scripts exist under /opt/GPUCompress ──────
# (The base image clones GPUCompress and all scripts land under /opt/GPUCompress/scripts;
# this is a belt-and-suspenders assertion so a broken base build fails here, not
# at Phase 1 run time.)
test -f /opt/GPUCompress/scripts/train_and_export_checkpoints.py
test -f /opt/GPUCompress/scripts/train_gpt2_checkpoints.py
test -x /opt/GPUCompress/build/generic_benchmark
