# Synthetic HDF5 Dataset Generator

A command-line tool for generating synthetic HDF5 datasets with controllable entropy and distribution characteristics for compression algorithm benchmarking.

## Features

- **Entropy Control**: Precise Shannon entropy targeting (1-8 bits) using quantile-based quantization
- **Multiple Distributions**: Uniform, Normal, Exponential, Bimodal, Constant
- **Flexible Dimensions**: Support for 1D through 4D tensors
- **Byte-Size Specification**: Easy dataset and chunk sizing using human-readable formats (KB, MB, GB, TB)
- **Preset System**: Generate custom preset combinations for batch testing
- **Comprehensive Metadata**: All generation parameters and measured metrics stored in HDF5 attributes
- **Float32 Output**: All datasets generated as float32 arrays
- **Validation**: Automatic entropy validation with configurable tolerance

## 📚 Documentation

- **[README.md](README.md)** (this file) - Quick start, installation, basic usage
- **[GENERATION_PROCESS.md](GENERATION_PROCESS.md)** - Complete technical guide (how it works, entropy, quantization)
- **[EXTENDED_ENTROPY_GUIDE.md](EXTENDED_ENTROPY_GUIDE.md)** - Advanced usage for >8 bits entropy
- **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** - Complete documentation map and workflow guides

**New to this tool?** Start with this README, then read [GENERATION_PROCESS.md](GENERATION_PROCESS.md) for deeper understanding.

## Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Generate with custom parameters:
```bash
# Using explicit shape dimensions
python generator.py generate \
  --distribution normal \
  --entropy 5.0 \
  -s 500 -s 500 \
  --output custom.h5

# Using dataset size (easier for large datasets)
python generator.py generate \
  --distribution normal \
  --entropy 5.0 \
  --dataset-size 4MB \
  --output custom.h5
```

### Generate from a preset:
```bash
python generator.py generate \
  --preset combo_00001 \
  --config config/presets_small.yaml \
  --output test.h5
```

### List available presets:
```bash
python generator.py list-presets --config config/presets_small.yaml
```

### View file information:
```bash
python generator.py info test.h5
```

### Batch generate multiple datasets:
```bash
python generator.py batch \
  --config config/presets_small.yaml \
  --output-dir datasets/
```

## Core Concepts

### 1. Dataset Size Specification

You can specify dataset and chunk sizes in two ways:

**Option 1: Explicit Dimensions** (using `-s` or `--shape`)
```bash
-s 1000 -s 1000  # Creates a 1000×1000 2D array (4MB for float32)
```

**Option 2: Byte Sizes** (using `--dataset-size` or `--chunk-bytes`)
```bash
--dataset-size 4MB     # Automatically calculates square dimensions
--dataset-size 1GB     # Creates ~16384×16384 array
--chunk-bytes 64MB     # Sets chunk size for better I/O
```

**Supported units**: `B`, `KB`, `MB`, `GB`, `TB`
- Case insensitive (mb, MB, Mb all work)
- Defaults to bytes if no unit specified
- Byte sizes create square 2D arrays by default

### 2. Distribution Types

Base statistical distributions:
- **uniform** - Equal probability across range
- **normal** - Gaussian bell curve
- **exponential** - Skewed distribution with exponential decay
- **bimodal** - Two distinct peaks
- **constant** - All same value (entropy = 0)

### 3. Shannon Entropy

Information content measured in bits. This is the **primary controllable attribute**:

- **1-2 bits**: Highly predictable, excellent compressibility (~2-4 distinct values)
- **3-5 bits**: Moderate randomness (~8-32 distinct values)
- **6-8 bits**: Near-random, poor compressibility (~64-256 distinct values)

**Implementation**: Uses quantile-based (equal-frequency) quantization to achieve target entropy for any distribution type. Unlike uniform quantization, this approach creates bins with equal number of values, making it distribution-agnostic.

## Generation Pipeline

The tool applies a simple 2-step pipeline:

1. **Generate Base Distribution** - Create array with chosen distribution
2. **Adjust Entropy** - Apply quantile-based quantization to hit target entropy

**Key Feature**: Quantile-based quantization ensures consistent entropy control across all distribution types (uniform, normal, exponential, bimodal).

## Preset System

Generate preset combinations using the preset generator:

```bash
# Generate 32 presets (4 distributions × 8 entropy levels)
python preset_generator.py small

# Generate 64 presets (4 distributions × 16 entropy levels)
python preset_generator.py medium

# Generate 128 presets (4 distributions × 32 entropy levels)
python preset_generator.py large
```

This creates files like `config/presets_small.yaml` with all combinations of:
- Distributions: uniform, normal, exponential, bimodal
- Entropy levels: 0.5 to 8.0 bits

## HDF5 Output Format

Each generated file contains:
```
filename.h5:
  /data                      # Main dataset (float32)
    Attributes:
      - param_distribution     # Generation parameters
      - param_entropy_target
      - param_shape
      - param_seed
      - actual_entropy         # Measured metrics
      - generation_timestamp
      ... (and more)
```

## Compression Testing

Test compression using h5repack:

```bash
# GZIP compression
h5repack -f GZIP=9 input.h5 output_gzip.h5

# GZIP with shuffle filter
h5repack -f SHUF -f GZIP=9 input.h5 output_shuffle_gzip.h5

# Compare sizes
ls -lh input.h5 output_*.h5
```

## CLI Commands

### `generate` - Generate a single dataset
```bash
python generator.py generate [OPTIONS]

Options:
  -p, --preset TEXT          Preset name from config file
  --config TEXT              Config file path (default: config/presets.yaml)
  -d, --distribution TEXT    Distribution type (uniform/normal/exponential/bimodal/constant)
  -s, --shape INTEGER        Shape dimensions (multiple -s for multi-D)
  --dataset-size TEXT        Dataset size in bytes (e.g., 4MB, 1GB). Overrides --shape
  -e, --entropy FLOAT        Target entropy in bits
  --seed INTEGER             Random seed for reproducibility
  -o, --output TEXT          Output HDF5 filename [required]
  --compression [gzip|lzf|none]  HDF5 compression (default: none)
  -c, --chunk-size INTEGER   HDF5 chunk dimensions (multiple -c)
  --chunk-bytes TEXT         Chunk size in bytes (e.g., 64MB, 128MB). Overrides --chunk-size
  --validate/--no-validate   Validate against targets (default: on)
  -q, --quiet                Suppress output
```

### `batch` - Generate multiple datasets
```bash
python generator.py batch [OPTIONS]

Options:
  -c, --config TEXT     Path to presets YAML file (default: config/presets.yaml)
  -o, --output-dir TEXT Output directory (default: datasets)
  -p, --presets TEXT    Specific presets to generate (multiple -p)
  -q, --quiet           Suppress output
```

### `info` - Display file information
```bash
python generator.py info FILENAME [OPTIONS]

Options:
  -d, --dataset TEXT  Dataset name within HDF5 file (default: data)
```

### `verify` - Verify file validity
```bash
python generator.py verify FILENAME [OPTIONS]

Options:
  -d, --dataset TEXT  Dataset name within HDF5 file (default: data)
```

### `list-presets` - List available presets
```bash
python generator.py list-presets [OPTIONS]

Options:
  --config TEXT  Config file to list from (default: config/presets.yaml)
```

## Examples

### Generate different entropy levels
```bash
# Low entropy (high compressibility)
python generator.py generate -d uniform -s 1000 -s 1000 -e 2.0 -o low_entropy.h5

# Medium entropy
python generator.py generate -d normal -s 1000 -s 1000 -e 5.0 -o medium_entropy.h5

# High entropy (low compressibility)
python generator.py generate -d uniform -s 1000 -s 1000 -e 8.0 -o high_entropy.h5
```

### Generate different distributions
```bash
python generator.py generate -d uniform -s 1000 -s 1000 -e 4.0 -o uniform.h5
python generator.py generate -d normal -s 1000 -s 1000 -e 4.0 -o normal.h5
python generator.py generate -d exponential -s 1000 -s 1000 -e 4.0 -o exponential.h5
python generator.py generate -d bimodal -s 1000 -s 1000 -e 4.0 -o bimodal.h5
```

### Generate different dimensional datasets
```bash
# 1D time series
python generator.py generate -d normal -s 1000000 -e 4.0 -o timeseries.h5

# 2D image
python generator.py generate -d uniform -s 1000 -s 1000 -e 6.0 -o image.h5

# 3D volume
python generator.py generate -d normal -s 200 -s 200 -s 200 -e 5.0 -o volume.h5

# 4D tensor
python generator.py generate -d normal -s 100 -s 100 -s 50 -s 20 -e 3.0 -o tensor.h5
```

### Generate with byte-size specifications (NEW)
```bash
# Small dataset (4MB)
python generator.py generate -d normal --dataset-size 4MB -e 5.0 -o small.h5

# Medium dataset (64MB)
python generator.py generate -d uniform --dataset-size 64MB -e 6.0 -o medium.h5

# Large dataset (1GB)
python generator.py generate -d exponential --dataset-size 1GB -e 7.0 -o large.h5

# With chunking for better I/O performance
python generator.py generate \
  --dataset-size 1GB \
  --chunk-bytes 64MB \
  -d normal -e 5.0 \
  -o chunked.h5

# Custom chunking and compression
python generator.py generate \
  --dataset-size 512MB \
  --chunk-bytes 128MB \
  --compression gzip \
  -d bimodal -e 4.0 \
  -o compressed.h5
```

### Batch generate preset combinations
```bash
# Generate all presets
python generator.py batch -c config/presets_small.yaml -o datasets/

# Generate specific presets only
python generator.py batch \
  -c config/presets_small.yaml \
  -o datasets/ \
  -p combo_00001 \
  -p combo_00002 \
  -p combo_00003
```

## Understanding Target vs Actual Entropy

This section explains the relationship between target entropy (what you request) and actual entropy (what you get), and why both are important.

### What is Shannon Entropy?

**Shannon entropy** measures the "information content" or "randomness" of data:
- **Low entropy (1-2 bits)**: Very predictable, highly compressible (few distinct values)
- **High entropy (7-8 bits)**: Very random, poorly compressible (many distinct values)

**Formula**: `H = -Σ p(x) * log₂(p(x))` where p(x) is the probability of each value.

### Target Entropy vs Actual Entropy

#### **TARGET ENTROPY** - What You Want

**Source**: This is the **user input** - what you *specify* when generating data.

```bash
# Specify target entropy via command line
python generator.py generate -d normal -e 6.13333333 -o test.h5

# Or use a preset (which contains entropy_target)
python generator.py generate --preset combo_00044 -o test.h5
```

**Purpose**:
- Control the compressibility of your test data
- "I want to test my compressor on 6-bit data"
- Stored in HDF5 as `param_entropy_target` attribute

#### **ACTUAL ENTROPY** - What You Got

**Source**: This is **measured** after the data is generated using the Shannon entropy formula.

**Purpose**:
- Verify what was actually achieved
- "Did I really get 6-bit data, or something else?"
- Stored in HDF5 as `actual_entropy` attribute

### Complete Generation Workflow

Let's trace a real example: **Target = 6.13333333 bits, Actual = 6.01499683 bits**

#### **Step 1: Request Target Entropy**
```bash
python generator.py generate --preset combo_00044 -o test.h5
# combo_00044 contains: entropy_target: 6.13333333, distribution: exponential
```

#### **Step 2: Generate Base Distribution**
```python
# Creates array with exponential distribution
data = distributions.generate(
    distribution='exponential',
    shape=(2048, 2048),
    seed=44
)
```
**Result**: 2048×2048 array with exponential distribution, but **uncontrolled entropy** (could be 4-8+ bits)

#### **Step 3: Adjust Entropy to Target** (Key Algorithm)
```python
# From entropy.py: adjust_entropy()

# Calculate bins needed
n_bins = round(2 ** 6.13333333)  # = 70 bins

# Create quantile-based bin edges
# Divides data into 70 bins with equal frequency
quantiles = np.linspace(0, 100, 71)
bin_edges = np.percentile(data, quantiles)

# Assign each value to a bin
quantized = np.digitize(data, bin_edges)

# Replace with bin centers
data[:] = bin_centers[quantized]
```

**What this does**:
- Takes continuous exponential data (millions of unique values)
- Reduces it to exactly **70 distinct values** (2^6.13 ≈ 70)
- Uses "quantile-based binning" so each value appears roughly equally often
- **Mathematical expectation**: With 70 equally-probable values, entropy ≈ log₂(70) ≈ 6.13 bits

#### **Step 4: Measure Actual Entropy**
```python
# From entropy.py: calculate_entropy()

# Create histogram
hist, _ = np.histogram(data.flatten(), bins=256)

# Calculate probabilities
probabilities = hist / sum(hist)

# Shannon entropy: H = -Σ p(x) * log₂(p(x))
entropy = -np.sum(probabilities * np.log2(probabilities))
# Result: 6.01499683 bits
```

**Why 6.015 instead of 6.133?**
- The quantization created ~70 distinct values
- But they're not *perfectly* equally distributed
- Small rounding errors in bin assignment
- Edge effects at data boundaries
- Result: **actual entropy = 6.015 bits**

#### **Step 5: Validation**
```python
# From metrics.py: validate_attributes()

error = abs(6.015 - 6.133) / 6.133 = 1.93%

# Default tolerance is 15%
if error < 15%:
    print("✓ PASS")
else:
    print("✗ FAIL")
```

### Output Interpretation

When you generate a dataset, you'll see:

```
Primary Attributes:
  Entropy:                  6.01499683 bits    ← ACTUAL (measured)

Validation Results:
Entropy: ✓ PASS
  Target:  6.13333333 bits    ← TARGET (requested)
  Actual:  6.01499683 bits    ← ACTUAL (measured)
  Error:   1.93%               ← Relative error
```

### Why Track Both Values?

#### **Real-World Example: Compression Testing**

```python
# Test 1: Low entropy (highly compressible)
Target:  2.0 bits  →  Actual: 2.03 bits  ✓
Expected compression ratio: ~8:1
If actual was 4.0 bits, your test would be invalid!

# Test 2: Medium entropy
Target:  5.0 bits  →  Actual: 4.97 bits  ✓
Expected compression ratio: ~2:1

# Test 3: High entropy (barely compressible)
Target:  8.0 bits  →  Actual: 7.89 bits  ✓
Expected compression ratio: ~1.1:1
```

**Critical Insight**: If target=8.0 but actual=6.0, your "incompressible" test data is actually quite compressible! This would invalidate your benchmark results.

### Visual Summary

```
USER INPUT              ALGORITHM              MEASUREMENT
    ↓                       ↓                       ↓
Target: 6.13 bits  →  Quantization      →  Actual: 6.01 bits
                      (70 bins, quantile-based)
                                         ↓
                                  Validation: 1.93% error ✓
                                  (within 15% tolerance)
```

### Key Takeaways

1. **Target entropy** = Your goal, the compressibility level you want to test
2. **Actual entropy** = What was really achieved, measured from the data
3. **Both are stored** in the HDF5 file for full traceability
4. **Validation** ensures the algorithm worked correctly (typical error: <5%)
5. **Small differences** (1-3%) are normal and acceptable due to:
   - Discrete quantization effects
   - Imperfect bin probability distribution
   - Edge effects in the data range

### HDF5 Storage

Both values are stored as attributes in the generated file:

```python
import h5py
with h5py.File('test.h5', 'r') as f:
    target = f['data'].attrs['param_entropy_target']    # 6.13333333
    actual = f['data'].attrs['actual_entropy']          # 6.01499683
```

This allows you to verify data characteristics later during analysis or benchmarking.

## Project Structure

```
syntheticWorkload/
├── generator.py            # Main CLI tool
├── preset_generator.py     # Preset combination generator
├── attributes/             # Attribute control modules
│   ├── distributions.py    # Base distributions
│   └── entropy.py          # Entropy adjustment (quantile-based)
├── utils/                  # Utility modules
│   ├── hdf5_writer.py      # HDF5 I/O operations
│   └── metrics.py          # Measurement and validation
├── config/                 # Configuration files
│   ├── presets_small.yaml   # 32 presets
│   ├── presets_medium_v2.yaml  # 64 presets (if generated)
│   └── presets_large_v2.yaml   # 128 presets (if generated)
├── requirements.txt        # Dependencies
├── venv/                   # Virtual environment
└── README.md              # This file
```

## Entropy Algorithm

The entropy adjustment uses **quantile-based quantization** (equal-frequency binning):

1. **Calculate target bins**: `n_bins = 2^entropy_target`
2. **Create quantile-based bin edges**: Each bin contains approximately equal number of values
3. **Assign values to bins**: Using `np.digitize()`
4. **Replace with bin centers**: Calculate mean of actual values in each bin

**Why quantile-based?** Traditional uniform quantization fails for non-uniform distributions because bins have vastly different occupancy, leading to poor entropy control. Quantile-based binning ensures equal occupancy regardless of distribution shape.

## Known Limitations

1. **Low Entropy Targets (<1.0 bit)**: Entropy targets below 1.0 bit may not be reliable due to quantization algorithm constraints. Targets ≥ 1.0 bit work consistently.

2. **Memory Usage**: Large datasets (GB range) require sufficient RAM. For 4D tensors, keep dimensions reasonable (e.g., 100×100×50×20 = 40MB).

3. **Entropy Validation**: Default tolerance is 10% error. For very low entropy targets (<2 bits), validation may fail due to the discrete nature of quantization.

## Validation Testing

Test entropy accuracy across presets:

```bash
python test_validation.py
```

This runs validation tests on multiple preset combinations and reports pass/fail rates.

## License

This tool was generated for compression benchmarking purposes.

## Contributing

To add new distributions or modify entropy algorithms, edit the corresponding modules in the `attributes/` directory.
