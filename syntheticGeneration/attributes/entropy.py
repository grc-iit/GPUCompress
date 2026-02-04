"""
Shannon entropy control for synthetic datasets.

Provides functions to adjust the information content of data by quantizing
values into discrete bins to achieve target entropy levels.
"""

import numpy as np
from typing import Union


def calculate_entropy(data: np.ndarray, n_bins: int = None) -> float:
    """
    Calculate Shannon entropy of data in bits.

    Shannon entropy: H = -sum(p_i * log2(p_i))
    where p_i is the probability of each bin.

    Args:
        data: Input data array
        n_bins: Number of bins for histogram (default: auto-detect from unique values)

    Returns:
        Shannon entropy in bits
    """
    # Auto-detect number of bins based on unique values
    if n_bins is None:
        n_unique = len(np.unique(data.flatten()))
        # Use enough bins to capture distinct values, but cap at reasonable max
        n_bins = min(n_unique * 2, 65536)

    # Create histogram
    hist, _ = np.histogram(data.flatten(), bins=n_bins)

    # Calculate probabilities (ignore zero bins)
    hist = hist[hist > 0]
    probabilities = hist / np.sum(hist)

    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return float(entropy)


def adjust_entropy(data: np.ndarray,
                   target_entropy: float,
                   seed: int = None) -> np.ndarray:
    """
    Adjust data entropy to target value through quantile-based quantization.

    Uses equal-frequency binning (quantiles) instead of equal-width binning.
    This ensures each bin has approximately the same number of values,
    making it work correctly for any distribution (uniform, normal, exponential, etc.)

    Higher entropy = more distinct values
    Lower entropy = fewer distinct values

    Args:
        data: Input data array (will be modified in place)
        target_entropy: Target entropy in bits
                       0 = all same value
                       1 = 2 distinct values
                       3 = ~8 distinct values
                       8 = ~256 distinct values
        seed: Random seed (currently unused, for API consistency)

    Returns:
        Modified array with adjusted entropy (same reference as input)

    Raises:
        ValueError: If target_entropy is negative or unreasonably large
    """
    if target_entropy < 0:
        raise ValueError(f"target_entropy must be non-negative, got {target_entropy}")

    if target_entropy > 32:
        raise ValueError(f"target_entropy too large (>32 bits), got {target_entropy}")

    # Calculate number of bins needed
    # With equal-frequency bins: entropy ≈ log2(n_bins) for ANY distribution
    n_bins = max(2, round(2 ** target_entropy))

    # Special case: zero entropy (all same value)
    if target_entropy < 0.01:
        data[:] = np.mean(data)
        return data

    # Store original range
    data_min = np.min(data)
    data_max = np.max(data)

    if data_max <= data_min:
        # Data is already constant
        return data

    # QUANTILE-BASED QUANTIZATION (equal-frequency bins)
    # Create bin edges at quantiles so each bin has ~equal number of values
    flat_data = data.flatten()
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(flat_data, quantiles)

    # Handle edge case where multiple quantiles map to same value
    # (can happen with discrete or low-variance data)
    bin_edges = np.unique(bin_edges)

    # If we lost bins due to duplicates, we won't achieve target entropy
    # but this is unavoidable with the data distribution
    actual_n_bins = len(bin_edges) - 1

    if actual_n_bins < 2:
        # Data is too uniform, force to constant
        data[:] = np.mean(data)
        return data

    # Assign each value to a bin using the quantile-based edges
    quantized = np.digitize(flat_data, bin_edges, right=False) - 1
    quantized = np.clip(quantized, 0, actual_n_bins - 1)

    # Map each bin to its center value
    # Calculate the actual center of values in each bin
    bin_centers = np.zeros(actual_n_bins)
    for i in range(actual_n_bins):
        mask = quantized == i
        if np.any(mask):
            bin_centers[i] = np.mean(flat_data[mask])
        else:
            # Empty bin (shouldn't happen with quantile-based binning)
            bin_centers[i] = (bin_edges[i] + bin_edges[i + 1]) / 2

    # Assign bin center values
    quantized_values = bin_centers[quantized]
    data[:] = quantized_values.reshape(data.shape).astype(data.dtype)

    return data


def adjust_entropy_iterative(data: np.ndarray,
                             target_entropy: float,
                             max_iterations: int = 5,
                             tolerance: float = 0.1,
                             seed: int = None) -> np.ndarray:
    """
    Iteratively adjust entropy to precisely hit target.

    Uses binary search to find optimal number of bins.

    Args:
        data: Input data array (will be modified in place)
        target_entropy: Target entropy in bits
        max_iterations: Maximum iterations for binary search
        tolerance: Acceptable error in entropy (bits)
        seed: Random seed (currently unused, for API consistency)

    Returns:
        Modified array with adjusted entropy
    """
    if target_entropy < 0:
        raise ValueError(f"target_entropy must be non-negative, got {target_entropy}")

    # Initial estimate
    n_bins_min = max(2, int(2 ** (target_entropy - 1)))
    n_bins_max = max(n_bins_min + 1, int(2 ** (target_entropy + 1)))

    # Store original data for retries
    data_min = np.min(data)
    data_max = np.max(data)

    best_n_bins = int(2 ** target_entropy)
    best_error = float('inf')

    for iteration in range(max_iterations):
        n_bins = (n_bins_min + n_bins_max) // 2

        # Apply quantization with current n_bins
        if data_max > data_min:
            normalized = (data - data_min) / (data_max - data_min)
            quantized = np.floor(normalized * (n_bins - 1)).astype(np.int32)
            quantized = np.clip(quantized, 0, n_bins - 1)
            bin_centers = (quantized + 0.5) / n_bins
            test_data = (bin_centers * (data_max - data_min) + data_min).astype(data.dtype)
        else:
            test_data = data.copy()

        # Measure actual entropy
        actual_entropy = calculate_entropy(test_data, n_bins=min(n_bins * 2, 1024))
        error = abs(actual_entropy - target_entropy)

        # Track best result
        if error < best_error:
            best_error = error
            best_n_bins = n_bins
            np.copyto(data, test_data)

        # Check if close enough
        if error < tolerance:
            break

        # Adjust search range
        if actual_entropy < target_entropy:
            n_bins_min = n_bins + 1
        else:
            n_bins_max = n_bins - 1

        # Prevent infinite loop
        if n_bins_min >= n_bins_max:
            break

    return data


def reduce_entropy(data: np.ndarray,
                   reduction_factor: float = 0.5,
                   seed: int = None) -> np.ndarray:
    """
    Reduce entropy by specified factor through quantization.

    Args:
        data: Input data array (will be modified in place)
        reduction_factor: Factor to reduce entropy [0.0-1.0]
                         0.0 = maximum reduction (constant)
                         1.0 = no reduction
        seed: Random seed for reproducibility

    Returns:
        Modified array with reduced entropy
    """
    if not 0.0 <= reduction_factor <= 1.0:
        raise ValueError(f"reduction_factor must be in [0, 1], got {reduction_factor}")

    # Measure current entropy
    current_entropy = calculate_entropy(data)

    # Calculate target entropy
    target_entropy = current_entropy * reduction_factor

    # Apply adjustment
    return adjust_entropy(data, target_entropy, seed=seed)


def add_entropy(data: np.ndarray,
                noise_scale: float = 0.1,
                seed: int = None) -> np.ndarray:
    """
    Increase entropy by adding noise.

    Args:
        data: Input data array (will be modified in place)
        noise_scale: Scale of noise relative to data range [0.0-1.0]
        seed: Random seed for reproducibility

    Returns:
        Modified array with increased entropy
    """
    rng = np.random.default_rng(seed)

    data_range = np.max(data) - np.min(data)
    noise = rng.normal(0, noise_scale * data_range, data.shape)

    data += noise.astype(data.dtype)

    return data
