#!/usr/bin/env python3
"""
Calculate entropy of an HDF5 dataset.

Usage:
    python calculate_entropy.py <file.h5>
    python calculate_entropy.py <file.h5> --dataset data
    python calculate_entropy.py <file.h5> --bins 256
"""

import argparse
import h5py
import sys
from utils.metrics import measure_all_attributes


def calculate_entropy(filename, dataset_name='data', n_bins=512, verbose=True):
    """
    Calculate entropy of HDF5 dataset.

    Args:
        filename: Path to HDF5 file
        dataset_name: Name of dataset within file
        n_bins: Number of bins for entropy calculation (default: 512 for accuracy)
        verbose: Print detailed information

    Returns:
        Dictionary with entropy and statistics
    """
    try:
        # Read dataset
        with h5py.File(filename, 'r') as f:
            if dataset_name not in f:
                print(f"✗ Error: Dataset '{dataset_name}' not found in file")
                print(f"  Available datasets: {list(f.keys())}")
                return None

            data = f[dataset_name][:]

            # Get stored attributes if available
            stored_entropy = f[dataset_name].attrs.get('actual_entropy')
            stored_target = f[dataset_name].attrs.get('param_entropy_target')

        # Calculate metrics
        metrics = measure_all_attributes(data, n_entropy_bins=n_bins)

        if verbose:
            print("=" * 70)
            print(f"Entropy Analysis: {filename}")
            print("=" * 70)
            print(f"Dataset:     {dataset_name}")
            print(f"Shape:       {metrics['shape']}")
            print(f"Data type:   {metrics['dtype']}")
            print(f"Elements:    {metrics['size']:,}")
            print()
            print(f"Calculated entropy:  {metrics['entropy']:.4f} bits")

            if stored_entropy is not None:
                print(f"Stored entropy:      {stored_entropy:.4f} bits")
                if abs(metrics['entropy'] - stored_entropy) < 0.01:
                    print("                     ✓ Matches calculated entropy")
                else:
                    print(f"                     ⚠ Differs by {abs(metrics['entropy'] - stored_entropy):.4f} bits")

            if stored_target is not None:
                print(f"Target entropy:      {stored_target:.4f} bits")
                deviation = abs(metrics['entropy'] - stored_target)
                if deviation < 0.1:
                    print(f"                     ✓ Within tolerance (Δ {deviation:.4f})")
                else:
                    print(f"                     ⚠ Outside tolerance (Δ {deviation:.4f})")

            print()
            print("Statistics:")
            print(f"  Min:       {metrics['min']:.6f}")
            print(f"  Max:       {metrics['max']:.6f}")
            print(f"  Mean:      {metrics['mean']:.6f}")
            print(f"  Std Dev:   {metrics['std']:.6f}")
            print("=" * 70)

        return metrics

    except FileNotFoundError:
        print(f"✗ Error: File not found: {filename}")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Calculate entropy of HDF5 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate entropy of dataset
  python calculate_entropy.py data.h5

  # Specify dataset name
  python calculate_entropy.py data.h5 --dataset mydata

  # Use more bins for even higher precision
  python calculate_entropy.py data.h5 --bins 1024

  # Use fewer bins (faster but less accurate)
  python calculate_entropy.py data.h5 --bins 256

  # Quiet mode (just print entropy value)
  python calculate_entropy.py data.h5 --quiet
        """
    )

    parser.add_argument(
        'filename',
        help='Path to HDF5 file'
    )
    parser.add_argument(
        '--dataset', '-d',
        default='data',
        help='Dataset name within HDF5 file (default: data)'
    )
    parser.add_argument(
        '--bins', '-b',
        type=int,
        default=512,
        help='Number of bins for entropy calculation (default: 512 for accuracy)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Only print entropy value'
    )

    args = parser.parse_args()

    # Calculate entropy
    metrics = calculate_entropy(
        args.filename,
        dataset_name=args.dataset,
        n_bins=args.bins,
        verbose=not args.quiet
    )

    if metrics is None:
        sys.exit(1)

    # Quiet mode: just print entropy value
    if args.quiet:
        print(f"{metrics['entropy']:.4f}")

    sys.exit(0)


if __name__ == '__main__':
    main()
