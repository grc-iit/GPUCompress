#!/usr/bin/env python3
"""
Download Gray-Scott reaction-diffusion data from The Well (HuggingFace).

Extracts fields A and B as raw float32 .bin files for GPUCompress evaluation.
Each file is 128x128 = 16,384 floats = 64 KB.

Dataset: polymathic-ai/gray_scott_reaction_diffusion
  6 patterns: bubbles, gliders, maze, spirals, spots, worms
  Each HDF5: (n_traj, 1001, 128, 128) per field (A, B)
  Valid split: 20 trajectories per pattern, 2.5 GB each

Usage:
    python eval/download_grayscott.py --output-dir eval/data_gs/
    python eval/download_grayscott.py --output-dir eval/data_gs/ --patterns spots maze --timesteps 10 --trajectories 3
    python eval/download_grayscott.py --explore --pattern spots
"""

import argparse
import os
import sys
import numpy as np

REPO_ID = "polymathic-ai/gray_scott_reaction_diffusion"
REPO_TYPE = "dataset"

FIELDS = ["A", "B"]

PATTERNS = {
    "bubbles":  {"F": 0.098, "k": 0.057},
    "gliders":  {"F": 0.014, "k": 0.054},
    "maze":     {"F": 0.029, "k": 0.057},
    "spirals":  {"F": 0.018, "k": 0.051},
    "spots":    {"F": 0.03,  "k": 0.062},
    "worms":    {"F": 0.058, "k": 0.065},
}


def resolve_hdf5_path(pattern: str, split: str = "valid") -> str:
    """Build the repo-relative HDF5 path for a given pattern and split."""
    info = PATTERNS[pattern]
    return f"data/{split}/gray_scott_reaction_diffusion_{pattern}_F_{info['F']}_k_{info['k']}.hdf5"


def get_hdf5_file(pattern: str, split: str = "valid"):
    """Download and open HDF5 file via hf_hub_download."""
    import h5py
    from huggingface_hub import hf_hub_download

    repo_path = resolve_hdf5_path(pattern, split)
    print(f"Downloading {repo_path} ...")
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=repo_path,
        repo_type=REPO_TYPE,
    )
    print(f"  Cached at: {local_path}")
    return h5py.File(local_path, "r")


def explore_hdf5(h5f):
    """Print HDF5 tree structure."""
    import h5py

    def visit(name, obj):
        indent = "  " * name.count("/")
        if isinstance(obj, h5py.Dataset):
            print(f"  {indent}{name}: shape={obj.shape}, dtype={obj.dtype}")
        else:
            print(f"  {indent}{name}/")

    print("\nHDF5 structure:")
    h5f.visititems(visit)

    if "scalars" in h5f:
        print("\nScalars:")
        for k in h5f["scalars"]:
            print(f"  {k} = {h5f['scalars'][k][()]}")


def select_indices(total: int, count: int):
    """Select evenly spaced indices."""
    if count >= total:
        return list(range(total))
    indices = np.linspace(0, total - 1, count, dtype=int)
    return sorted(set(indices.tolist()))


def extract_fields(h5f, pattern: str, num_timesteps: int, num_trajectories: int,
                   output_dir: str):
    """Extract fields A and B as raw float32 .bin files."""
    os.makedirs(output_dir, exist_ok=True)

    # Check shape: (n_traj, n_time, 128, 128)
    ds_a = h5f["t0_fields/A"]
    print(f"  Dataset shape: {ds_a.shape}, dtype: {ds_a.dtype}")

    n_traj = ds_a.shape[0]
    n_time = ds_a.shape[1]

    traj_indices = select_indices(n_traj, num_trajectories)
    time_indices = select_indices(n_time, num_timesteps)

    print(f"  Trajectories: {len(traj_indices)} of {n_traj} -> {traj_indices}")
    print(f"  Timesteps: {len(time_indices)} of {n_time}")
    print(f"  Fields: {FIELDS}")
    print(f"  Output: {output_dir}\n")

    file_count = 0
    total_bytes = 0

    for field_name in FIELDS:
        ds = h5f[f"t0_fields/{field_name}"]

        for traj_idx in traj_indices:
            for t_idx in time_indices:
                data = np.asarray(ds[traj_idx, t_idx], dtype=np.float32).ravel()

                out_name = (f"float32_gs_{pattern}_{field_name}_"
                            f"tr{traj_idx:03d}_t{t_idx:04d}.bin")
                out_path = os.path.join(output_dir, out_name)

                data.tofile(out_path)
                file_count += 1
                total_bytes += data.nbytes

                print(f"  [{file_count:4d}] {out_name}  "
                      f"shape={data.shape}  "
                      f"range=[{data.min():.6f}, {data.max():.6f}]  "
                      f"size={data.nbytes / 1024:.1f} KB")

    print(f"\n  Subtotal for {pattern}: {file_count} files, "
          f"{total_bytes / 1024 / 1024:.1f} MB")
    return file_count, total_bytes


def main():
    parser = argparse.ArgumentParser(
        description="Download Gray-Scott reaction-diffusion data from The Well")
    parser.add_argument("--output-dir", default="eval/data_gs/",
                        help="Output directory for .bin files (default: eval/data_gs/)")
    parser.add_argument("--patterns", nargs="+", default=list(PATTERNS.keys()),
                        choices=list(PATTERNS.keys()),
                        help="Pattern types to download (default: all 6)")
    parser.add_argument("--timesteps", type=int, default=15,
                        help="Number of evenly spaced timesteps to extract (default: 15)")
    parser.add_argument("--trajectories", type=int, default=5,
                        help="Number of trajectories per pattern (default: 5)")
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"],
                        help="Dataset split (default: valid, 2.5 GB/pattern)")
    parser.add_argument("--explore", action="store_true",
                        help="Print HDF5 structure without extracting")
    parser.add_argument("--pattern", default="spots",
                        help="Pattern to explore (with --explore)")
    args = parser.parse_args()

    if args.explore:
        h5f = get_hdf5_file(args.pattern, args.split)
        explore_hdf5(h5f)
        h5f.close()
        return

    total_files = 0
    total_bytes = 0

    for pattern in args.patterns:
        print(f"\n{'='*60}")
        print(f"Pattern: {pattern} (F={PATTERNS[pattern]['F']}, k={PATTERNS[pattern]['k']})")
        print(f"{'='*60}")

        h5f = get_hdf5_file(pattern, args.split)
        nf, nb = extract_fields(h5f, pattern, args.timesteps, args.trajectories,
                                args.output_dir)
        h5f.close()
        total_files += nf
        total_bytes += nb

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_files} files, {total_bytes / 1024 / 1024:.1f} MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
