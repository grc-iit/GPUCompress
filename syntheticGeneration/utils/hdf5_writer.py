"""
HDF5 file I/O operations for synthetic datasets.

Provides functions to write datasets to HDF5 files with comprehensive
metadata for traceability and analysis.
"""

import h5py
import numpy as np
from typing import Dict, Any, Union, Tuple
from datetime import datetime
import sys
import os

# Add parent directory to path to import metrics module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import measure_all_attributes


def write_dataset(filename: str,
                 data: np.ndarray,
                 generation_params: Dict[str, Any] = None,
                 dataset_name: str = 'data',
                 compression: str = None,
                 chunks: Union[Tuple[int, ...], bool, None] = None,
                 measure_metrics: bool = True) -> str:
    """
    Write dataset to HDF5 file with metadata.

    Args:
        filename: Output HDF5 filename
        data: Data array to write (will be converted to float32)
        generation_params: Dictionary of generation parameters to store as attributes
        dataset_name: Name of the dataset within HDF5 file (default: 'data')
        compression: HDF5 compression filter ('gzip', 'lzf', None)
                    Default: None (no compression during write, test externally)
        chunks: HDF5 chunk shape. None=contiguous (default), True=auto, or tuple for specific size
                Example: (4096, 4096) for 64MB chunks with 2D data
        measure_metrics: Whether to measure and store actual metrics

    Returns:
        Path to created file

    Raises:
        ValueError: If data is not a numpy array
        IOError: If file write fails
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("data must be a numpy array")

    # Ensure data is float32
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    # Measure actual attributes if requested
    actual_metrics = measure_all_attributes(data) if measure_metrics else {}

    try:
        with h5py.File(filename, 'w') as f:
            # Create dataset
            dset = f.create_dataset(
                dataset_name,
                data=data,
                compression=compression,
                chunks=chunks,
                shuffle=(compression == 'gzip')  # Shuffle filter helps compression
            )

            # Store generation parameters as attributes
            if generation_params:
                for key, value in generation_params.items():
                    if value is not None:
                        # Convert to JSON-compatible types
                        if isinstance(value, (list, tuple)):
                            dset.attrs[f'param_{key}'] = str(value)
                        elif isinstance(value, (np.ndarray,)):
                            dset.attrs[f'param_{key}'] = str(value.tolist())
                        else:
                            dset.attrs[f'param_{key}'] = value

            # Store measured metrics as attributes
            if actual_metrics:
                for key, value in actual_metrics.items():
                    if key == 'shape':
                        dset.attrs['actual_shape'] = str(value)
                    elif isinstance(value, (int, float, str)):
                        dset.attrs[f'actual_{key}'] = value

            # Store metadata
            dset.attrs['generation_timestamp'] = datetime.now().isoformat()
            dset.attrs['dataset_name'] = dataset_name
            dset.attrs['dtype'] = str(data.dtype)
            dset.attrs['ndim'] = data.ndim

        return filename

    except Exception as e:
        raise IOError(f"Failed to write HDF5 file {filename}: {e}")


def read_dataset(filename: str,
                dataset_name: str = 'data') -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read dataset and metadata from HDF5 file.

    Args:
        filename: Input HDF5 filename
        dataset_name: Name of the dataset within HDF5 file (default: 'data')

    Returns:
        Tuple of (data array, attributes dictionary)

    Raises:
        IOError: If file read fails
        KeyError: If dataset not found
    """
    try:
        with h5py.File(filename, 'r') as f:
            if dataset_name not in f:
                raise KeyError(f"Dataset '{dataset_name}' not found in {filename}")

            dset = f[dataset_name]

            # Read data
            data = dset[:]

            # Read attributes
            attrs = dict(dset.attrs)

        return data, attrs

    except Exception as e:
        raise IOError(f"Failed to read HDF5 file {filename}: {e}")


def print_file_info(filename: str, dataset_name: str = 'data'):
    """
    Print information about an HDF5 file.

    Args:
        filename: HDF5 filename
        dataset_name: Name of the dataset to inspect
    """
    try:
        data, attrs = read_dataset(filename, dataset_name)

        print(f"\n{'=' * 70}")
        print(f"HDF5 File: {filename}")
        print(f"{'=' * 70}")

        print(f"\nDataset: {dataset_name}")
        print(f"  Shape:  {data.shape}")
        print(f"  Dtype:  {data.dtype}")
        print(f"  Size:   {data.size:,} elements")

        # Memory size
        total_bytes = data.nbytes
        if total_bytes < 1024:
            size_str = f"{total_bytes} B"
        elif total_bytes < 1024**2:
            size_str = f"{total_bytes / 1024:.2f} KB"
        elif total_bytes < 1024**3:
            size_str = f"{total_bytes / (1024**2):.2f} MB"
        else:
            size_str = f"{total_bytes / (1024**3):.2f} GB"
        print(f"  Memory: {size_str}")

        # Print generation parameters
        print(f"\nGeneration Parameters:")
        param_keys = [k for k in attrs.keys() if k.startswith('param_')]
        if param_keys:
            for key in sorted(param_keys):
                clean_key = key.replace('param_', '')
                print(f"  {clean_key:20s}: {attrs[key]}")
        else:
            print("  (none)")

        # Print measured metrics
        print(f"\nMeasured Attributes:")
        actual_keys = [k for k in attrs.keys() if k.startswith('actual_')]
        if actual_keys:
            for key in sorted(actual_keys):
                clean_key = key.replace('actual_', '')
                value = attrs[key]
                if isinstance(value, float):
                    print(f"  {clean_key:20s}: {value:.6f}")
                else:
                    print(f"  {clean_key:20s}: {value}")
        else:
            print("  (none)")

        # Print other metadata
        print(f"\nMetadata:")
        meta_keys = [k for k in attrs.keys()
                    if not k.startswith('param_') and not k.startswith('actual_')]
        for key in sorted(meta_keys):
            print(f"  {key:20s}: {attrs[key]}")

        print(f"{'=' * 70}\n")

    except Exception as e:
        print(f"Error reading file: {e}")


def verify_hdf5_file(filename: str,
                    dataset_name: str = 'data',
                    verbose: bool = True) -> bool:
    """
    Verify HDF5 file validity and data integrity.

    Args:
        filename: HDF5 filename to verify
        dataset_name: Name of the dataset to check
        verbose: Print verification results

    Returns:
        True if file is valid, False otherwise
    """
    try:
        # Try to open and read file
        data, attrs = read_dataset(filename, dataset_name)

        checks = []

        # Check 1: Data type is float32
        dtype_ok = (data.dtype == np.float32)
        checks.append(('Data type is float32', dtype_ok))

        # Check 2: No NaN or Inf values
        finite_ok = np.all(np.isfinite(data))
        checks.append(('All values finite (no NaN/Inf)', finite_ok))

        # Check 3: Non-empty data
        nonempty_ok = (data.size > 0)
        checks.append(('Non-empty dataset', nonempty_ok))

        # Check 4: Has metadata
        has_metadata = len(attrs) > 0
        checks.append(('Has metadata attributes', has_metadata))

        if verbose:
            print(f"\nVerifying: {filename}")
            print("-" * 50)
            for check_name, passed in checks:
                status = "✓" if passed else "✗"
                print(f"{status} {check_name}")
            print("-" * 50)

        return all(passed for _, passed in checks)

    except Exception as e:
        if verbose:
            print(f"✗ File verification failed: {e}")
        return False
