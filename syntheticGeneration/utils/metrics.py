"""
Metrics calculation for synthetic datasets.

Provides functions to measure various data attributes and validate
that generated datasets match specifications.
"""

import numpy as np
from typing import Dict, Any
import sys
import os

# Add parent directory to path to import attributes modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attributes.entropy import calculate_entropy


def measure_all_attributes(data: np.ndarray,
                          n_entropy_bins: int = None) -> Dict[str, float]:
    """
    Measure all data attributes for a dataset.

    Args:
        data: Input data array
        n_entropy_bins: Number of bins for entropy calculation (default: auto-detect)

    Returns:
        Dictionary with all measured attributes:
        - entropy: Shannon entropy in bits
        - mean: Mean value
        - std: Standard deviation
        - min: Minimum value
        - max: Maximum value
        - shape: Data shape (as tuple)
        - size: Total number of elements
        - dtype: Data type
    """
    metrics = {
        'entropy': calculate_entropy(data, n_bins=n_entropy_bins),
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'shape': tuple(data.shape),
        'size': int(data.size),
        'dtype': str(data.dtype)
    }

    return metrics


def print_metrics(metrics: Dict[str, Any], title: str = "Dataset Metrics"):
    """
    Print metrics in a formatted table.

    Args:
        metrics: Dictionary of metrics from measure_all_attributes()
        title: Title for the metrics display
    """
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print(f"{'=' * 60}")

    # Primary attributes
    print(f"\nPrimary Attributes:")
    print(f"  Entropy:              {metrics['entropy']:>14.8f} bits")

    # Statistics
    print(f"\nStatistics:")
    print(f"  Mean:                 {metrics['mean']:>14.8f}")
    print(f"  Std Dev:              {metrics['std']:>14.8f}")
    print(f"  Min:                  {metrics['min']:>14.8f}")
    print(f"  Max:                  {metrics['max']:>14.8f}")

    # Data properties
    print(f"\nData Properties:")
    print(f"  Shape:                {str(metrics['shape']):>20}")
    print(f"  Size:                 {metrics['size']:>20,} elements")
    print(f"  Dtype:                {metrics['dtype']:>20}")

    # Memory size
    element_size = np.dtype(metrics['dtype']).itemsize
    total_bytes = metrics['size'] * element_size
    if total_bytes < 1024:
        size_str = f"{total_bytes} B"
    elif total_bytes < 1024**2:
        size_str = f"{total_bytes / 1024:.2f} KB"
    elif total_bytes < 1024**3:
        size_str = f"{total_bytes / (1024**2):.2f} MB"
    else:
        size_str = f"{total_bytes / (1024**3):.2f} GB"

    print(f"  Memory Size:          {size_str:>20}")
    print(f"{'=' * 60}\n")


def validate_attributes(data: np.ndarray,
                       target_entropy: float = None,
                       tolerance: float = 0.15) -> Dict[str, bool]:
    """
    Validate that generated data matches target specifications.

    Args:
        data: Generated data array
        target_entropy: Target entropy in bits (None to skip)
        tolerance: Acceptable relative error for validation

    Returns:
        Dictionary with validation results (True = passed, False = failed)
    """
    results = {}

    # Measure actual values
    actual_entropy = calculate_entropy(data)

    # Validate entropy
    if target_entropy is not None:
        error = abs(actual_entropy - target_entropy) / max(target_entropy, 0.1)
        results['entropy_valid'] = error < tolerance
        results['entropy_error'] = error
        results['actual_entropy'] = actual_entropy
        results['target_entropy'] = target_entropy

    return results


def print_validation_results(validation: Dict[str, Any]):
    """
    Print validation results in formatted output.

    Args:
        validation: Dictionary from validate_attributes()
    """
    print(f"\n{'=' * 60}")
    print(f"{'Validation Results':^60}")
    print(f"{'=' * 60}\n")

    if 'entropy_valid' in validation:
        status = "✓ PASS" if validation['entropy_valid'] else "✗ FAIL"
        print(f"Entropy: {status}")
        print(f"  Target:  {validation['target_entropy']:.8f} bits")
        print(f"  Actual:  {validation['actual_entropy']:.8f} bits")
        print(f"  Error:   {validation['entropy_error']:.2%}\n")

    print(f"{'=' * 60}\n")
