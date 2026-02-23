#!/usr/bin/env python3
"""
Workload Adaptation Evaluation

Streams 5 scientific fields from Post Neutron Star Merger dataset,
compresses each timestep with GPUCompress ALGO_AUTO (NN + reinforcement),
records prediction accuracy, and plots MAPE + reinforcement rate.

Usage:
    source venv/bin/activate
    export LD_LIBRARY_PATH=/tmp/lib:$LD_LIBRARY_PATH
    python eval/workload_adaptation.py
"""

import os
import sys
import csv
import ctypes
import time
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FIELDS = ["density", "internal_energy", "temperature", "pressure", "entropy"]
N_TIMESTEPS = 40          # timesteps per field (max 181)
ERROR_BOUND = 0.1       # lossy quantization bound
EXPLORATION_THRESH = 0.20  # matches library default
REINFORCE_LR = 0.1
REINFORCE_MAPE = 0.20

REPO_ID = "polymathic-ai/post_neutron_star_merger"
HDF5_PATH = "data/test/post_neutron_star_merger_scenario_2.hdf5"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_PATH = os.path.join(PROJECT_ROOT, "build", "libgpucompress.so")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "neural_net", "weights", "model.nnwt")
CSV_PATH = os.path.join(PROJECT_ROOT, "eval", "workload_adaptation.csv")

# ---------------------------------------------------------------------------
# ctypes bindings for libgpucompress
# ---------------------------------------------------------------------------
class GpuCompressConfig(ctypes.Structure):
    _fields_ = [
        ("algorithm", ctypes.c_int),
        ("preprocessing", ctypes.c_uint),
        ("error_bound", ctypes.c_double),
        ("cuda_device", ctypes.c_int),
        ("cuda_stream", ctypes.c_void_p),
    ]

class GpuCompressStats(ctypes.Structure):
    _fields_ = [
        ("original_size", ctypes.c_size_t),
        ("compressed_size", ctypes.c_size_t),
        ("compression_ratio", ctypes.c_double),
        ("entropy_bits", ctypes.c_double),
        ("mad", ctypes.c_double),
        ("second_derivative", ctypes.c_double),
        ("algorithm_used", ctypes.c_int),
        ("preprocessing_used", ctypes.c_uint),
        ("throughput_mbps", ctypes.c_double),
        ("predicted_ratio", ctypes.c_double),
        ("predicted_comp_time_ms", ctypes.c_double),
        ("actual_comp_time_ms", ctypes.c_double),
        ("sgd_fired", ctypes.c_int),
    ]

def load_library(lib_path):
    lib = ctypes.CDLL(lib_path)

    lib.gpucompress_init.restype = ctypes.c_int
    lib.gpucompress_init.argtypes = [ctypes.c_char_p]
    lib.gpucompress_cleanup.restype = None

    lib.gpucompress_compress.restype = ctypes.c_int
    lib.gpucompress_compress.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(GpuCompressConfig), ctypes.POINTER(GpuCompressStats),
    ]

    lib.gpucompress_max_compressed_size.restype = ctypes.c_size_t
    lib.gpucompress_max_compressed_size.argtypes = [ctypes.c_size_t]

    lib.gpucompress_algorithm_name.restype = ctypes.c_char_p
    lib.gpucompress_algorithm_name.argtypes = [ctypes.c_int]

    lib.gpucompress_enable_active_learning.restype = ctypes.c_int
    lib.gpucompress_enable_active_learning.argtypes = [ctypes.c_char_p]
    lib.gpucompress_disable_active_learning.restype = None

    lib.gpucompress_set_exploration_threshold.restype = None
    lib.gpucompress_set_exploration_threshold.argtypes = [ctypes.c_double]

    lib.gpucompress_set_reinforcement.restype = None
    lib.gpucompress_set_reinforcement.argtypes = [
        ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ]

    lib.gpucompress_experience_count.restype = ctypes.c_size_t

    return lib

# ---------------------------------------------------------------------------
# Phase 1: Stream workloads from HuggingFace
# ---------------------------------------------------------------------------
def open_remote_hdf5():
    """Open the HDF5 dataset via remote streaming (no full download)."""
    import fsspec
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(REPO_ID, HDF5_PATH, repo_type="dataset")
    fs = fsspec.filesystem("https")
    fobj = fs.open(url, "rb")
    import h5py
    hf = h5py.File(fobj, "r")
    return hf, fobj

def iter_workloads(hf, fields, n_timesteps):
    """Yield (field_name, timestep_idx, numpy_array) tuples."""
    for field in fields:
        dset = hf[f"t0_fields/{field}"]
        n = min(n_timesteps, dset.shape[1])
        for t in range(n):
            arr = dset[0, t, :, :, :].astype(np.float32)
            yield field, t, arr

# ---------------------------------------------------------------------------
# Phase 2: Evaluate with GPUCompress
# ---------------------------------------------------------------------------
def run_evaluation(lib, hf):
    """Compress each workload timestep, record predictions vs actuals."""
    # Initialize library
    rc = lib.gpucompress_init(WEIGHTS_PATH.encode())
    if rc != 0:
        print(f"FATAL: gpucompress_init failed: {rc}")
        sys.exit(1)

    # Enable active learning + reinforcement
    exp_path = os.path.join(PROJECT_ROOT, "eval", "experience_workload.csv")
    lib.gpucompress_enable_active_learning(exp_path.encode())
    lib.gpucompress_set_exploration_threshold(EXPLORATION_THRESH)
    lib.gpucompress_set_reinforcement(1, REINFORCE_LR, REINFORCE_MAPE, 0.0)

    # Preallocate output buffer (reuse across calls)
    max_input = 192 * 128 * 66 * 4  # ~6.2 MB per timestep
    max_out_size = lib.gpucompress_max_compressed_size(max_input)
    output_buf = (ctypes.c_uint8 * max_out_size)()

    results = []
    step = 0

    print(f"\n{'Step':>5} | {'Field':>16} {'t':>4} | "
          f"{'PredR':>6} {'ActR':>6} {'MAPE%':>6} | "
          f"{'PredT':>7} {'ActT':>7} | {'Algo':>8} {'SGD':>3}")
    print("-" * 90)

    for field, t, arr in iter_workloads(hf, FIELDS, N_TIMESTEPS):
        input_size = arr.nbytes
        output_size = ctypes.c_size_t(max_out_size)

        cfg = GpuCompressConfig()
        cfg.algorithm = 0  # ALGO_AUTO
        cfg.preprocessing = 0
        cfg.error_bound = ERROR_BOUND
        cfg.cuda_device = -1
        cfg.cuda_stream = None

        stats = GpuCompressStats()

        rc = lib.gpucompress_compress(
            arr.ctypes.data, input_size,
            output_buf, ctypes.byref(output_size),
            ctypes.byref(cfg), ctypes.byref(stats),
        )

        if rc != 0:
            print(f"  WARNING: step {step} compress failed: rc={rc}")
            step += 1
            continue

        algo_name = lib.gpucompress_algorithm_name(stats.algorithm_used).decode()

        # Compute MAPE for this step
        ratio_mape = 0.0
        if stats.compression_ratio > 0:
            ratio_mape = abs(stats.predicted_ratio - stats.compression_ratio) / stats.compression_ratio * 100

        results.append({
            "step": step,
            "field": field,
            "timestep": t,
            "predicted_ratio": stats.predicted_ratio,
            "actual_ratio": stats.compression_ratio,
            "predicted_comp_time_ms": stats.predicted_comp_time_ms,
            "actual_comp_time_ms": stats.actual_comp_time_ms,
            "algorithm": algo_name,
            "sgd_fired": stats.sgd_fired,
            "entropy": stats.entropy_bits,
            "mad": stats.mad,
            "second_derivative": stats.second_derivative,
        })

        print(f"{step:5d} | {field:>16} {t:4d} | "
              f"{stats.predicted_ratio:6.3f} {stats.compression_ratio:6.3f} {ratio_mape:5.1f}% | "
              f"{stats.predicted_comp_time_ms:7.3f} {stats.actual_comp_time_ms:7.3f} | "
              f"{algo_name:>8} {stats.sgd_fired:>3}")

        step += 1

    exp_count = lib.gpucompress_experience_count()
    lib.gpucompress_disable_active_learning()
    lib.gpucompress_cleanup()

    print(f"\nTotal steps: {step}, Experience samples: {exp_count}")
    return results

def save_csv(results, csv_path):
    """Write results to CSV."""
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {csv_path}")

# ---------------------------------------------------------------------------
# Phase 3: Plot
# ---------------------------------------------------------------------------
def plot_results(csv_path, plot_path):
    """Plot MAPE rolling median + reinforcement rate (dual Y-axis)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Read CSV
    data = {k: [] for k in ["step", "field", "predicted_ratio", "actual_ratio",
                              "predicted_comp_time_ms", "actual_comp_time_ms",
                              "sgd_fired"]}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["step"].append(int(row["step"]))
            data["field"].append(row["field"])
            data["predicted_ratio"].append(float(row["predicted_ratio"]))
            data["actual_ratio"].append(float(row["actual_ratio"]))
            data["predicted_comp_time_ms"].append(float(row["predicted_comp_time_ms"]))
            data["actual_comp_time_ms"].append(float(row["actual_comp_time_ms"]))
            data["sgd_fired"].append(int(row["sgd_fired"]))

    steps = np.array(data["step"])
    pred_ratio = np.array(data["predicted_ratio"])
    act_ratio = np.array(data["actual_ratio"])
    pred_ct = np.array(data["predicted_comp_time_ms"])
    act_ct = np.array(data["actual_comp_time_ms"])
    sgd = np.array(data["sgd_fired"])
    fields = data["field"]

    # MAPE per step
    ratio_mape = np.where(act_ratio > 0,
                          np.abs(pred_ratio - act_ratio) / act_ratio * 100, 0)
    ct_mape = np.where(act_ct > 0,
                       np.abs(pred_ct - act_ct) / act_ct * 100, 0)

    # Rolling windows
    def rolling_median(arr, w):
        out = np.zeros_like(arr)
        for i in range(len(arr)):
            start = max(0, i - w + 1)
            out[i] = np.median(arr[start:i + 1])
        return out

    def rolling_mean(arr, w):
        out = np.zeros_like(arr, dtype=float)
        for i in range(len(arr)):
            start = max(0, i - w + 1)
            out[i] = np.mean(arr[start:i + 1])
        return out

    ratio_mape_roll = rolling_median(ratio_mape, 10)
    ct_mape_roll = rolling_median(ct_mape, 10)
    reinf_cumulative = np.cumsum(sgd)

    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    ax1.plot(steps, ratio_mape_roll, "-", color="#FF8C00", linewidth=1.5,
             alpha=0.9, label="Compression Ratio")
    ax1.plot(steps, ct_mape_roll, "-", color="#8B008B", linewidth=1.5,
             alpha=0.9, label="Compress Time")

    ax2.plot(steps, reinf_cumulative, "--", color="#B22222", linewidth=1.2,
             alpha=0.7, label="Reinforcements")

    # Vertical lines between workloads
    prev_field = fields[0]
    for i, f in enumerate(fields):
        if f != prev_field:
            ax1.axvline(x=steps[i], color="gray", linestyle=":", alpha=0.5)
            ax1.text(steps[i] + 0.5, 95, f, rotation=90, va="top",
                     fontsize=8, color="gray")
            prev_field = f
    # Label first field
    ax1.text(1, 95, fields[0], rotation=90, va="top", fontsize=8, color="gray")

    ax1.set_xlabel("Step (Simulation Timestep)", fontsize=11)
    ax1.set_ylabel("NN MAPE (%, 10-step rolling median)", fontsize=11)
    ax2.set_ylabel("Cumulative Reinforcements", fontsize=11,
                    color="#B22222")

    ax1.set_ylim(0, 100)
    ax2.tick_params(axis="y", labelcolor="#B22222")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
               fontsize=9, framealpha=0.8)

    plt.title("NN Prediction Accuracy vs Workload Adaptation\n"
              "(Post Neutron Star Merger Dataset)", fontsize=12)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  GPUCompress Workload Adaptation Evaluation")
    print("=" * 60)

    # Phase 1: Open remote dataset
    print("\n--- Phase 1: Connecting to HuggingFace dataset ---")
    hf, fobj = open_remote_hdf5()
    print(f"  Fields: {list(hf['t0_fields'].keys())}")
    print(f"  Shape per field: {hf['t0_fields/density'].shape}")
    print(f"  Using {N_TIMESTEPS} timesteps x {len(FIELDS)} fields = "
          f"{N_TIMESTEPS * len(FIELDS)} total steps")

    # Phase 2: Evaluate
    print("\n--- Phase 2: Running GPUCompress evaluation ---")
    lib = load_library(LIB_PATH)
    results = run_evaluation(lib, hf)
    save_csv(results, CSV_PATH)

    hf.close()
    fobj.close()

    # Phase 3: Plot
    print("\n--- Phase 3: Plotting results ---")
    plot_path = os.path.join(PROJECT_ROOT, "eval", "workload_adaptation.png")
    plot_results(CSV_PATH, plot_path)

    print("\nDone!")
