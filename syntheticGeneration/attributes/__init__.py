"""
Attribute generators for synthetic HDF5 datasets.

Active modules:
- distributions: Base data distribution generation (uniform, normal, exponential, bimodal, constant)
- entropy: Shannon entropy adjustment using quantile-based quantization (ONLY active attribute)

The generation pipeline is simplified to: Distribution → Entropy
"""
