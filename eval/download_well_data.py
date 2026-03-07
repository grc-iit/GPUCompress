#!/usr/bin/env python3
"""
Download post-neutron-star-merger simulation data from The Well (HuggingFace).

Streams HDF5 from polymathic-ai/post_neutron_star_merger, extracts selected
scalar fields as raw float32 .bin files for GPUCompress evaluation.

Usage:
    python eval/download_well_data.py --output-dir eval/data/
    python eval/download_well_data.py --explore --scenario 0
    python eval/download_well_data.py --output-dir eval/data/ --timesteps 3 --scenario 0
"""

import argparse
import os
import sys
import numpy as np

# Fields to extract (3D scalar fields on the simulation grid)
FIELDS = [
    "density",
    "internal_energy",
    "electron_fraction",
    "temperature",
    "entropy",
    "pressure",
]

# HuggingFace dataset info
REPO_ID = "polymathic-ai/post_neutron_star_merger"
REPO_TYPE = "dataset"


def ensure_dependencies():
    """Check that required dependencies are installed."""
    missing = []
    try:
        import h5py  # noqa: F401
    except ImportError:
        missing.append("h5py")

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        missing.append("huggingface_hub")

    try:
        import fsspec  # noqa: F401
    except ImportError:
        missing.append("fsspec")

    if missing:
        raise ImportError(
            f"Missing required package(s): {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}"
        )


def resolve_hdf5_path(scenario: int) -> str:
    """Resolve the repo-relative path for a given scenario number.

    The Well dataset layout:
        data/train/post_neutron_star_merger_scenario_{0,3,4,5,6,7}.hdf5
        data/valid/post_neutron_star_merger_scenario_1.hdf5
        data/test/post_neutron_star_merger_scenario_2.hdf5
    """
    SPLIT_MAP = {0: "train", 1: "valid", 2: "test",
                 3: "train", 4: "train", 5: "train", 6: "train", 7: "train"}
    split = SPLIT_MAP.get(scenario, "train")
    return f"data/{split}/post_neutron_star_merger_scenario_{scenario}.hdf5"


def get_hdf5_file(scenario: int):
    """Open HDF5 file via hf_hub_download, with fsspec streaming fallback."""
    import h5py

    repo_path = resolve_hdf5_path(scenario)

    # Primary: download via huggingface_hub (caches locally)
    try:
        from huggingface_hub import hf_hub_download
        print(f"Downloading {repo_path} ...")
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=repo_path,
            repo_type=REPO_TYPE,
        )
        print(f"  Cached at: {local_path}")
        h5f = h5py.File(local_path, "r")
        return h5f
    except Exception as e:
        print(f"  hf_hub_download failed: {e}")

    # Fallback: fsspec streaming
    hf_url = f"hf://datasets/{REPO_ID}/{repo_path}"
    try:
        import fsspec
        print(f"Attempting streaming access: {hf_url}")
        fs = fsspec.filesystem("hf", repo_type="dataset")
        fp = fs.open(f"{REPO_ID}/{repo_path}", "rb")
        h5f = h5py.File(fp, "r")
        print("  Streaming access succeeded")
        return h5f
    except Exception as e2:
        print(f"  Streaming also failed: {e2}")
        sys.exit(1)


def explore_hdf5(h5f):
    """Print HDF5 tree structure for discovery."""
    import h5py

    def visit(name, obj):
        indent = "  " * name.count("/")
        if isinstance(obj, h5py.Dataset):
            print(f"  {indent}{name}: shape={obj.shape}, dtype={obj.dtype}")
        else:
            print(f"  {indent}{name}/")

    print("\nHDF5 structure:")
    h5f.visititems(visit)

    # Print top-level attributes
    if h5f.attrs:
        print("\nTop-level attributes:")
        for k, v in h5f.attrs.items():
            print(f"  {k}: {v}")


def select_timesteps(total_timesteps: int, num_timesteps: int):
    """Select evenly spaced timestep indices."""
    if num_timesteps >= total_timesteps:
        return list(range(total_timesteps))
    indices = np.linspace(0, total_timesteps - 1, num_timesteps, dtype=int)
    return sorted(set(indices.tolist()))


def extract_fields(h5f, scenario: int, num_timesteps: int, output_dir: str):
    """Extract scalar fields as raw float32 .bin files."""
    os.makedirs(output_dir, exist_ok=True)

    # Discover available fields and timestep count
    # The Well layout: t0_fields/{name} with shape (n_traj, n_time, n_r, n_theta, n_phi)
    available_fields = []
    total_timesteps = None

    PREFIXES = ["t0_fields", "t1_fields", "fields", "data", ""]

    for field in FIELDS:
        found = False
        for prefix in PREFIXES:
            key = f"{prefix}/{field}" if prefix else field
            if key in h5f:
                ds = h5f[key]
                available_fields.append(key)
                if total_timesteps is None:
                    # Shape: (n_traj, n_time, ...) — timestep is dim 1
                    total_timesteps = ds.shape[1] if len(ds.shape) >= 2 else ds.shape[0]
                    print(f"Dataset shape for '{key}': {ds.shape}, dtype: {ds.dtype}")
                found = True
                break
        if not found:
            print(f"  Warning: field '{field}' not found, skipping")

    if not available_fields:
        print("Error: no matching fields found in HDF5 file")
        print("Use --explore to see available datasets")
        sys.exit(1)

    if total_timesteps is None:
        print("Error: could not determine timestep count")
        sys.exit(1)

    timestep_indices = select_timesteps(total_timesteps, num_timesteps)
    print(f"\nTotal timesteps in file: {total_timesteps}")
    print(f"Extracting {len(timestep_indices)} timesteps: {timestep_indices}")
    print(f"Fields: {[f.split('/')[-1] for f in available_fields]}")
    print(f"Output directory: {output_dir}\n")

    file_count = 0
    total_bytes = 0

    for field_key in available_fields:
        field_name = field_key.split("/")[-1]
        ds = h5f[field_key]

        for t_idx in timestep_indices:
            # Shape is (n_traj, n_time, ...) — use trajectory 0
            if len(ds.shape) >= 2 and ds.shape[0] == 1:
                data = ds[0, t_idx]  # trajectory 0, timestep t_idx
            else:
                data = ds[t_idx]
            data = np.asarray(data, dtype=np.float32).ravel()

            # Build output filename
            out_name = f"float32_nsm_{field_name}_s{scenario:02d}_t{t_idx:03d}.bin"
            out_path = os.path.join(output_dir, out_name)

            data.tofile(out_path)
            file_count += 1
            total_bytes += data.nbytes

            print(f"  [{file_count:3d}] {out_name}  "
                  f"shape={data.shape}  "
                  f"range=[{data.min():.4e}, {data.max():.4e}]  "
                  f"size={data.nbytes / 1024 / 1024:.2f} MB")

    print(f"\nDone: {file_count} files, {total_bytes / 1024 / 1024:.1f} MB total")


def main():
    parser = argparse.ArgumentParser(
        description="Download post-neutron-star-merger data from The Well (HuggingFace)")
    parser.add_argument("--output-dir", default="eval/data/",
                        help="Output directory for .bin files (default: eval/data/)")
    parser.add_argument("--timesteps", type=int, default=15,
                        help="Number of evenly spaced timesteps to extract (default: 15)")
    parser.add_argument("--scenario", type=int, default=0,
                        help="Scenario index to use (default: 0)")
    parser.add_argument("--explore", action="store_true",
                        help="Print HDF5 tree structure without extracting")
    args = parser.parse_args()

    ensure_dependencies()

    h5f = get_hdf5_file(args.scenario)

    if args.explore:
        explore_hdf5(h5f)
        h5f.close()
        return

    extract_fields(h5f, args.scenario, args.timesteps, args.output_dir)
    h5f.close()


if __name__ == "__main__":
    main()
