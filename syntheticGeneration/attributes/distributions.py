"""
Data distribution generators for synthetic datasets.

Provides functions to generate base data arrays with various statistical distributions.
All functions return float32 numpy arrays.
"""

import numpy as np
from typing import Tuple, Union


def uniform(shape: Union[int, Tuple[int, ...]],
            value_min: float = 0.0,
            value_max: float = 10000.0,
            seed: int = None) -> np.ndarray:
    """
    Generate data with uniform distribution.

    Args:
        shape: Shape of the output array (int for 1D or tuple for multi-D)
        value_min: Minimum value (default: 0.0)
        value_max: Maximum value (default: 10000.0)
        seed: Random seed for reproducibility

    Returns:
        Float32 array with uniform distribution
    """
    rng = np.random.default_rng(seed)
    if isinstance(shape, int):
        shape = (shape,)
    return rng.uniform(value_min, value_max, shape).astype(np.float32)


def normal(shape: Union[int, Tuple[int, ...]],
           mean: float = 5000.0,
           std: float = 1000.0,
           seed: int = None) -> np.ndarray:
    """
    Generate data with normal (Gaussian) distribution.

    Args:
        shape: Shape of the output array (int for 1D or tuple for multi-D)
        mean: Mean of the distribution (default: 5000.0)
        std: Standard deviation (default: 1000.0)
        seed: Random seed for reproducibility

    Returns:
        Float32 array with normal distribution
    """
    rng = np.random.default_rng(seed)
    if isinstance(shape, int):
        shape = (shape,)
    return rng.normal(mean, std, shape).astype(np.float32)


def exponential(shape: Union[int, Tuple[int, ...]],
                scale: float = 1000.0,
                seed: int = None) -> np.ndarray:
    """
    Generate data with exponential distribution.

    Args:
        shape: Shape of the output array (int for 1D or tuple for multi-D)
        scale: Scale parameter (1/lambda) of the distribution (default: 1000.0)
        seed: Random seed for reproducibility

    Returns:
        Float32 array with exponential distribution
    """
    rng = np.random.default_rng(seed)
    if isinstance(shape, int):
        shape = (shape,)
    return rng.exponential(scale, shape).astype(np.float32)


def bimodal(shape: Union[int, Tuple[int, ...]],
            mean1: float = 2000.0,
            mean2: float = 8000.0,
            std1: float = 500.0,
            std2: float = 500.0,
            ratio: float = 0.5,
            seed: int = None) -> np.ndarray:
    """
    Generate data with bimodal distribution (mixture of two Gaussians).

    Args:
        shape: Shape of the output array (int for 1D or tuple for multi-D)
        mean1: Mean of first Gaussian (default: 2000.0)
        mean2: Mean of second Gaussian (default: 8000.0)
        std1: Std dev of first Gaussian (default: 500.0)
        std2: Std dev of second Gaussian (default: 500.0)
        ratio: Mixing ratio [0-1], proportion from first Gaussian (default: 0.5)
        seed: Random seed for reproducibility

    Returns:
        Float32 array with bimodal distribution
    """
    rng = np.random.default_rng(seed)
    if isinstance(shape, int):
        shape = (shape,)

    total_size = np.prod(shape)
    n1 = int(total_size * ratio)
    n2 = total_size - n1

    # Generate two Gaussian distributions
    data1 = rng.normal(mean1, std1, n1)
    data2 = rng.normal(mean2, std2, n2)

    # Combine and shuffle
    data = np.concatenate([data1, data2])
    rng.shuffle(data)

    return data.reshape(shape).astype(np.float32)


def constant(shape: Union[int, Tuple[int, ...]],
             value: float = 5000.0,
             seed: int = None) -> np.ndarray:
    """
    Generate data with constant value (zero entropy case).

    Args:
        shape: Shape of the output array (int for 1D or tuple for multi-D)
        value: The constant value to fill (default: 5000.0)
        seed: Random seed (unused, for API consistency)

    Returns:
        Float32 array filled with constant value
    """
    if isinstance(shape, int):
        shape = (shape,)
    return np.full(shape, value, dtype=np.float32)


# Distribution registry for easy access
DISTRIBUTIONS = {
    'uniform': uniform,
    'normal': normal,
    'exponential': exponential,
    'bimodal': bimodal,
    'constant': constant
}


def generate(distribution: str,
             shape: Union[int, Tuple[int, ...]],
             seed: int = None,
             **kwargs) -> np.ndarray:
    """
    Generate data with specified distribution.

    Args:
        distribution: Distribution name ('uniform', 'normal', 'exponential', 'bimodal', 'constant')
        shape: Shape of the output array
        seed: Random seed for reproducibility
        **kwargs: Additional distribution-specific parameters

    Returns:
        Float32 array with specified distribution

    Raises:
        ValueError: If distribution name is not recognized
    """
    if distribution not in DISTRIBUTIONS:
        raise ValueError(f"Unknown distribution: {distribution}. "
                        f"Available: {list(DISTRIBUTIONS.keys())}")

    return DISTRIBUTIONS[distribution](shape, seed=seed, **kwargs)
