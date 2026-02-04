# Data Generation Process: Complete Technical Guide

This document provides a detailed, step-by-step explanation of how synthetic datasets are generated with controlled entropy levels for compression benchmarking.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [The Complete Generation Pipeline](#the-complete-generation-pipeline)
4. [Understanding Entropy Bits](#understanding-entropy-bits)
5. [Understanding Bins](#understanding-bins)
6. [Quantization Explained](#quantization-explained)
7. [Step-by-Step Example](#step-by-step-example)
8. [Code Flow Through generator.py](#code-flow-through-generatorpy)

---

## Quick Start

### Generate a Dataset

```bash
# Basic command
python generator.py generate -d normal -e 6.0 -s 1000 -s 1000 -o test.h5

# Using dataset size
python generator.py generate -d normal -e 6.0 --dataset-size 4MB -o test.h5

# From a preset
python generator.py generate --preset combo_00001 --config config/presets_small.yaml -o test.h5
```

### Check the Generated Entropy

```bash
python calculate_entropy.py test.h5
```

---

## Core Concepts

### What is Shannon Entropy?

**Shannon entropy** measures the **information content** or **randomness** of data in bits.

**Formula:**
```
H = -Σ p(x) · log₂(p(x))
```
where `p(x)` is the probability of each value occurring.

**Intuitive Understanding:**
- **High entropy**: Data is unpredictable, has many different values
- **Low entropy**: Data is predictable, has few different values

**Examples:**

| Entropy | Unique Values | Example | Compressibility |
|---------|---------------|---------|-----------------|
| 0 bits  | 1 (constant)  | [5, 5, 5, 5, 5] | Excellent (100:1) |
| 1 bit   | 2             | [1, 2, 1, 2, 1] | Very good (8:1) |
| 3 bits  | ~8            | [1,2,3,4,5,6,7,8,1,2...] | Good (3:1) |
| 6 bits  | ~64           | More random distribution | Fair (1.5:1) |
| 8 bits  | ~256          | Nearly random | Poor (1.1:1) |

### Why Control Entropy?

**Purpose:** Test compression algorithms across different data characteristics

```
Low Entropy (1-2 bits)  → Should compress very well
Medium Entropy (4-5 bits) → Should compress moderately
High Entropy (7-8 bits)  → Should barely compress
```

---

## Understanding Entropy Bits

### What Does "6 bits of entropy" Mean?

**Mathematical Definition:**
- 6 bits means approximately `2^6 = 64` equally-probable distinct values

**Practical Meaning:**
- If you see a value, you need ~6 bits to encode it
- Data has ~64 different values that appear roughly equally often
- Information content is equivalent to a 6-bit number

### Entropy Range

| Entropy Value | Number of Values | Description |
|---------------|------------------|-------------|
| 0.0 bits      | 1                | All same value (constant) |
| 1.0 bits      | 2                | Binary data |
| 2.0 bits      | 4                | Very low variety |
| 3.0 bits      | 8                | Low variety |
| 4.0 bits      | 16               | Moderate variety |
| 5.0 bits      | 32               | Good variety |
| 6.0 bits      | 64               | High variety |
| 7.0 bits      | 128              | Very high variety |
| 8.0 bits      | 256              | Near-maximum variety |

### What About Higher Entropy? (Beyond 8 bits)

**Question:** What if we have 1 million distinct values?

**Answer:**
```
Entropy = log₂(1,000,000) = 19.93 bits ≈ 20 bits
```

**Extended Entropy Table:**

| Entropy Value | Number of Values | Description | Compression |
|---------------|------------------|-------------|-------------|
| 8.0 bits      | 256              | Tool maximum | ~1.2:1 |
| 9.0 bits      | 512              | Near-continuous | ~1.1:1 |
| 10.0 bits     | 1,024            | Near-continuous | ~1.05:1 |
| 12.0 bits     | 4,096            | Very high | ~1.02:1 |
| 16.0 bits     | 65,536           | 16-bit color | ~1.00:1 |
| **19.93 bits** | **1,000,000**   | **1 million values** | **1.00:1** |
| 24.0 bits     | 16,777,216       | 24-bit RGB | 1.00:1 |
| 32.0 bits     | 4,294,967,296    | 32-bit integer | 1.00:1 |

**Why This Tool Stops at 8 Bits:**

1. **Compression Testing Purpose:**
   - At 8 bits: Compression ratio ~1.1-1.2:1 (poor)
   - At 20 bits: Compression ratio ~1.0:1 (none)
   - **No useful difference** between 8 bits and 20 bits for compression testing

2. **Diminishing Returns:**
   ```
   1 bit  → 8 bits:  Huge compression difference (8:1 → 1.2:1)
   8 bits → 20 bits: Negligible difference (1.2:1 → 1.0:1)
   ```

3. **Memory Efficiency:**
   - Data stored as float32 (~7 significant digits)
   - Beyond ~10 bits, precision becomes limiting factor
   - 1M unique values would need 4MB just for value lookup

4. **Processing Speed:**
   - More unique values = slower quantization
   - 256 values: milliseconds
   - 1M values: several seconds

**Practical Compression Results:**

| Entropy | GZIP Ratio | Use Case |
|---------|------------|----------|
| 1-3 bits | 4:1 to 8:1 | Testing "good" compression |
| 4-6 bits | 1.5:1 to 3:1 | Testing "moderate" compression |
| 7-8 bits | 1.1:1 to 1.3:1 | Testing "poor" compression |
| 9+ bits | 1.0:1 to 1.05:1 | Essentially random (no point) |

**Conclusion:** For compression benchmarking, testing beyond 8-10 bits provides no additional insight.

### Calculating Entropy

```python
# Pseudo-code for entropy calculation
def calculate_entropy(data):
    # 1. Create histogram (count occurrences)
    histogram = count_values_in_bins(data, n_bins=512)

    # 2. Calculate probabilities
    probabilities = histogram / total_count

    # 3. Apply Shannon formula
    entropy = -sum(p * log2(p) for p in probabilities if p > 0)

    return entropy
```

---

## Understanding Bins

### Two Types of "Bins"

**IMPORTANT:** There are TWO different concepts both called "bins" in this system:

#### 1. Generation Bins (Quantization Bins)

**Purpose:** Control how many unique values to create

**Location:** `attributes/entropy.py`, line 75
```python
n_bins = round(2 ** target_entropy)
```

**Example:**
```
Target entropy: 6.0 bits
Generation bins: 2^6.0 = 64 bins
Result: Creates 64 unique values in the data
```

**Key Point:** This determines the **actual data content**

#### 2. Measurement Bins (Histogram Bins)

**Purpose:** Measure the distribution of values

**Location:** `attributes/entropy.py`, line 12
```python
def calculate_entropy(data, n_bins=512)
```

**Example:**
```
Measurement bins: 512 (default)
Purpose: Create histogram with 512 bins to measure distribution
Result: Measures how values are distributed
```

**Key Point:** This is just for **measurement**, doesn't change data

### Why the Confusion?

Both use the word "bins" but serve completely different purposes:

```
GENERATION:
  Input: target_entropy = 6.0 bits
  Calculate: generation_bins = 2^6 = 64
  Action: Quantize data to 64 unique values

MEASUREMENT:
  Input: data with unknown entropy
  Use: measurement_bins = 512 (fixed)
  Action: Create 512-bin histogram to measure distribution
  Output: calculated_entropy = 6.01 bits
```

**Rule:** `measurement_bins >= generation_bins` for accurate measurement

---

## Quantization Explained

### What is Quantization?

**Quantization** is the process of reducing continuous data to discrete values.

**Analogy:**
- **Before:** Temperature = 72.3456°F (infinite precision)
- **After:** Temperature = 72°F (rounded to nearest integer)

### Why Quantize?

**Goal:** Control entropy by controlling the number of distinct values

```
More unique values → Higher entropy → Less compressible
Fewer unique values → Lower entropy → More compressible
```

### Types of Quantization

#### 1. Uniform Quantization (NOT used in this tool)

**How it works:** Divide the data range into equal-width bins

```
Data range: [0, 100]
Bins: 10
Bin width: 10

Bins: [0-10], [10-20], [20-30], ..., [90-100]
```

**Problem:** Doesn't work well for skewed distributions!

**Example Problem:**
```
Data: Exponential distribution (most values near 0)
Bin 1 [0-10]:   90% of data  → High probability
Bin 2 [10-20]:   5% of data  → Low probability
Bin 3 [20-30]:   3% of data  → Low probability
...
Bin 10 [90-100]: 0.1% of data → Very low probability

Result: Uneven probabilities → Uncontrolled entropy
```

#### 2. Quantile-Based Quantization (USED in this tool) ✓

**How it works:** Divide data into bins with equal number of values

```
Data: 1000 values
Bins: 10
Values per bin: 100 (equal)

Bin 1: values [0%-10%] → 100 values
Bin 2: values [10%-20%] → 100 values
...
Bin 10: values [90%-100%] → 100 values
```

**Advantage:** Works for ANY distribution!

**Example:**
```
Exponential distribution:
Bin 1: [0.0 - 0.5]    → 100 values (many small values)
Bin 2: [0.5 - 1.2]    → 100 values
Bin 3: [1.2 - 2.5]    → 100 values
...
Bin 10: [15.0 - 100.0] → 100 values (few large values)

Result: Equal probabilities → Controlled entropy!
```

### The Quantization Algorithm

**Step-by-step process in `adjust_entropy()`:**

```python
def adjust_entropy(data, target_entropy):
    # Step 1: Calculate how many bins needed
    n_bins = round(2 ** target_entropy)
    # Example: 6 bits → 64 bins

    # Step 2: Find quantile-based bin edges
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(data, quantiles)
    # Example: [0%, 1.56%, 3.12%, ..., 98.44%, 100%] for 64 bins

    # Step 3: Assign each value to a bin
    bin_indices = np.digitize(data, bin_edges)
    # Each value → bin number (0 to 63)

    # Step 4: Calculate bin centers
    for i in range(n_bins):
        bin_centers[i] = mean(values_in_bin[i])

    # Step 5: Replace all values with their bin centers
    data[:] = bin_centers[bin_indices]

    return data
```

**Visual Example:**

```
Before Quantization (continuous):
[7.23, 21.77, 36.46, 51.38, 66.56, 81.97, 97.54, 113.42, ...]
→ Millions of unique values, high entropy

After Quantization (discrete, 64 bins):
[7.23, 21.77, 36.46, 51.38, 66.56, 81.97, 97.54, 113.42, ...]
→ Exactly 64 unique values, controlled entropy ~6 bits
```

---

## The Complete Generation Pipeline

### Overview

```
User Input → Distribution → Quantization → HDF5 File
    ↓            ↓              ↓              ↓
  Target      Base Data    Adjust Entropy   Store +
  Entropy                                    Metadata
```

### Detailed Flow

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: User Specifies Parameters                           │
├─────────────────────────────────────────────────────────────┤
│ - Distribution: normal, uniform, exponential, bimodal       │
│ - Target Entropy: 1.0 - 8.0 bits                           │
│ - Shape: (1000, 1000) or --dataset-size 4MB                │
│ - Seed: for reproducibility                                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Generate Base Distribution                          │
├─────────────────────────────────────────────────────────────┤
│ File: attributes/distributions.py                           │
│                                                              │
│ def generate(distribution, shape, seed):                    │
│     if distribution == 'normal':                            │
│         data = np.random.randn(shape)                       │
│     elif distribution == 'uniform':                         │
│         data = np.random.uniform(0, 10000, shape)           │
│     ...                                                      │
│                                                              │
│ Result: Array with statistical distribution                 │
│         BUT uncontrolled entropy (could be 4-8+ bits)       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Adjust Entropy (Quantization)                       │
├─────────────────────────────────────────────────────────────┤
│ File: attributes/entropy.py                                 │
│                                                              │
│ def adjust_entropy(data, target_entropy):                   │
│     # Calculate bins needed                                 │
│     n_bins = round(2 ** target_entropy)                     │
│                                                              │
│     # Quantile-based binning                                │
│     quantiles = np.linspace(0, 100, n_bins + 1)             │
│     bin_edges = np.percentile(data, quantiles)              │
│                                                              │
│     # Assign values to bins                                 │
│     bin_indices = np.digitize(data, bin_edges)              │
│                                                              │
│     # Replace with bin centers                              │
│     data[:] = bin_centers[bin_indices]                      │
│                                                              │
│ Result: Data now has exactly 2^target_entropy unique values │
│         Entropy controlled to target                        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: Measure Actual Metrics                              │
├─────────────────────────────────────────────────────────────┤
│ File: utils/metrics.py                                      │
│                                                              │
│ def measure_all_attributes(data):                           │
│     actual_entropy = calculate_entropy(data, n_bins=512)    │
│     actual_mean = np.mean(data)                             │
│     actual_std = np.std(data)                               │
│     ...                                                      │
│                                                              │
│ Result: Measured values to verify generation                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: Write to HDF5 File                                  │
├─────────────────────────────────────────────────────────────┤
│ File: utils/hdf5_writer.py                                  │
│                                                              │
│ Structure:                                                   │
│   /data [float32 array]                                     │
│     Attributes:                                              │
│       param_distribution: "normal"                          │
│       param_entropy_target: 6.00000000                      │
│       param_shape: "(1000, 1000)"                           │
│       param_seed: 42                                        │
│       actual_entropy: 6.01234567                            │
│       actual_mean: 1000.18                                  │
│       actual_std: 992.08                                    │
│       generation_timestamp: "2026-02-04T..."                │
│                                                              │
│ Result: Self-documenting HDF5 file with data + metadata    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 6: Validation                                          │
├─────────────────────────────────────────────────────────────┤
│ File: utils/metrics.py                                      │
│                                                              │
│ def validate_attributes(data, target_entropy):              │
│     error = abs(actual - target) / target                   │
│     passed = (error < 0.15)  # 15% tolerance                │
│                                                              │
│ Output:                                                      │
│   Target:  6.00000000 bits                                  │
│   Actual:  6.01234567 bits                                  │
│   Error:   0.21%                                            │
│   Status:  ✓ PASS                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Example

Let's generate a dataset with **6 bits of entropy** using a **normal distribution**.

### Command

```bash
python generator.py generate \
  --distribution normal \
  --entropy 6.0 \
  --shape 1000 1000 \
  --seed 42 \
  --output example.h5
```

### Step 1: Parse Command Line

```python
# In generator.py:generate()
params = {
    'distribution': 'normal',
    'entropy_target': 6.0,
    'shape': (1000, 1000),
    'seed': 42
}
```

### Step 2: Generate Base Distribution

```python
# In attributes/distributions.py
data = distributions.generate(
    distribution='normal',
    shape=(1000, 1000),
    seed=42
)

# Result: 1000×1000 array
# Values follow normal distribution (Gaussian)
# Example values: [-2.34, 0.12, 1.87, -0.45, 2.91, ...]
# Entropy at this point: ~7-8 bits (many unique values)
```

**Visualization:**
```
Before quantization (continuous):
   Count
    │     ╱‾‾╲
    │   ╱      ╲
    │ ╱          ╲
    │╱____________╲_____  Value
   -3  -2  -1  0  1  2  3

Thousands of unique values
Entropy: ~7-8 bits
```

### Step 3: Adjust Entropy to Target (6 bits)

```python
# In attributes/entropy.py
entropy.adjust_entropy(data, target_entropy=6.0, seed=42)

# What happens:
# 1. Calculate bins needed
n_bins = round(2 ** 6.0)  # = 64 bins

# 2. Create quantile-based bin edges (65 edges for 64 bins)
quantiles = [0%, 1.56%, 3.12%, ..., 98.44%, 100%]
bin_edges = np.percentile(data, quantiles)
# Result: [-2.87, -2.13, -1.86, ..., 1.86, 2.13, 2.87]

# 3. Assign each value to a bin
# Original: -2.34 → Bin 2 (between -2.87 and -2.13)
# Original:  0.12 → Bin 32 (near middle)
# Original:  1.87 → Bin 62 (near top)

# 4. Calculate bin centers
bin_centers[2] = mean(all values in bin 2) = -2.50
bin_centers[32] = mean(all values in bin 32) = 0.15
bin_centers[62] = mean(all values in bin 62) = 1.93

# 5. Replace values with bin centers
# -2.34 → -2.50 (bin 2 center)
#  0.12 →  0.15 (bin 32 center)
#  1.87 →  1.93 (bin 62 center)
```

**After Quantization:**
```
After quantization (64 discrete values):
   Count
    │     ▄▄▄
    │   ▄▄███▄▄
    │ ▄▄███████▄▄
    │▄███████████▄___  Value
   -3  -2  -1  0  1  2  3

Exactly 64 unique values
Each value appears ~15,625 times (1,000,000 / 64)
Entropy: log₂(64) = 6.0 bits
```

### Step 4: Measure Actual Metrics

```python
# In utils/metrics.py
metrics = measure_all_attributes(data, n_entropy_bins=512)

# Results:
# {
#     'entropy': 5.99872,  # Close to 6.0!
#     'mean': 0.003,
#     'std': 0.998,
#     'min': -2.87,
#     'max': 2.87,
#     'shape': (1000, 1000),
#     'size': 1000000,
#     'dtype': 'float32'
# }
```

### Step 5: Write to HDF5

```python
# In utils/hdf5_writer.py
hdf5_writer.write_dataset(
    filename='example.h5',
    data=data,
    generation_params=params,
    measure_metrics=True
)

# File structure:
# example.h5
#   /data [1000×1000 float32]
#     Attributes:
#       param_distribution: "normal"
#       param_entropy_target: 6.0
#       param_shape: "(1000, 1000)"
#       param_seed: 42
#       actual_entropy: 5.99872
#       actual_mean: 0.003
#       actual_std: 0.998
#       generation_timestamp: "2026-02-04T..."
```

### Step 6: Validation

```python
# In utils/metrics.py
validation = validate_attributes(data, target_entropy=6.0)

# Output:
# Validation Results
# ==================
# Entropy: ✓ PASS
#   Target:  6.00000000 bits
#   Actual:  5.99872000 bits
#   Error:   0.02%
```

### Final Result

```bash
$ ls -lh example.h5
-rw-r--r-- 1 user user 4.0M Feb 4 10:00 example.h5

$ python calculate_entropy.py example.h5
======================================================================
Entropy Analysis: example.h5
======================================================================
Dataset:     data
Shape:       (1000, 1000)
Data type:   float32
Elements:    1,000,000

Calculated entropy:  5.9987 bits
Stored entropy:      5.9987 bits
                     ✓ Matches calculated entropy
Target entropy:      6.0000 bits
                     ✓ Within tolerance (Δ 0.0013)

Statistics:
  Min:       -2.870000
  Max:        2.870000
  Mean:       0.003000
  Std Dev:    0.998000
======================================================================
```

---

## Code Flow Through generator.py

### File: `generator.py`

This is the main entry point for dataset generation.

#### Function Call Stack

```
main()
  ↓
cli()
  ↓
generate() [command handler]
  ↓
  ├→ load_preset() [if using preset]
  │    ↓
  │    └→ Parse YAML config file
  │
  ├→ parse_size_to_bytes() [if using --dataset-size]
  │    ↓
  │    └→ Convert "4MB" → 4194304 bytes → (1024, 1024) shape
  │
  ├→ generate_dataset()
  │    ↓
  │    ├→ distributions.generate()
  │    │    ↓
  │    │    └→ Create base distribution array
  │    │
  │    └→ entropy.adjust_entropy()
  │         ↓
  │         └→ Quantize to target entropy
  │
  ├→ hdf5_writer.write_dataset()
  │    ↓
  │    ├→ metrics.measure_all_attributes()
  │    │    ↓
  │    │    └→ Measure actual entropy and statistics
  │    │
  │    └→ Write to HDF5 with metadata
  │
  └→ metrics.validate_attributes()
       ↓
       └→ Check if actual matches target
```

#### Detailed Code Flow

**1. Command Line Parsing (lines 187-210)**

```python
@cli.command()
@click.option('--preset', '-p', help='Preset name')
@click.option('--distribution', '-d', help='Distribution type')
@click.option('--entropy', '-e', 'entropy_target', type=float)
@click.option('--shape', '-s', multiple=True, type=int)
@click.option('--dataset-size', type=str, help='Size in bytes')
@click.option('--output', '-o', required=True)
def generate(preset, distribution, shape, dataset_size, entropy_target, ...):
```

**2. Load Parameters (lines 229-260)**

```python
# From preset or command line
if preset:
    params = load_preset(preset, config)
else:
    params = {}

# Override with command-line args
if distribution:
    params['distribution'] = distribution
if entropy_target is not None:
    params['entropy_target'] = entropy_target

# Handle dataset-size option
if dataset_size:
    size_bytes = parse_size_to_bytes(dataset_size)  # "4MB" → 4194304
    calculated_shape = calculate_square_shape(size_bytes)  # → (1024, 1024)
    params['shape'] = list(calculated_shape)
```

**3. Generate Dataset (lines 266-272)**

```python
data = generate_dataset(
    distribution=params['distribution'],
    shape=params['shape'],
    entropy_target=params['entropy_target'],
    seed=params.get('seed'),
    verbose=verbose
)
```

Inside `generate_dataset()` (lines 93-142):

```python
def generate_dataset(distribution, shape, entropy_target, seed, verbose):
    # Step 1: Generate base distribution
    data = distributions.generate(
        distribution=distribution,
        shape=shape,
        seed=seed
    )

    # Step 2: Adjust entropy
    entropy.adjust_entropy(data, entropy_target, seed=seed)

    return data
```

**4. Write to HDF5 (lines 295-302)**

```python
hdf5_writer.write_dataset(
    filename=output,
    data=data,
    generation_params=params,
    compression=compression_filter,
    chunks=chunks_param,
    measure_metrics=True
)
```

**5. Validation (lines 308-317)**

```python
# Measure actual attributes
measured = metrics.measure_all_attributes(data)
metrics.print_metrics(measured, title=f"Generated Dataset: {output}")

# Validate against target
if validate:
    validation = metrics.validate_attributes(
        data,
        target_entropy=params.get('entropy_target')
    )
    metrics.print_validation_results(validation)
```

---

## Advanced Topics

### Precision: Float32 vs Float64

**Data Values:** Stored as `float32`
- Precision: ~7 significant digits
- Memory: 4 bytes per value
- Example: `366.73233032` (shown with 8 decimals, but limited precision)

**Entropy Metrics:** Stored as `float64`
- Precision: ~15-17 significant digits
- Memory: 8 bytes per value
- Example: `6.01499683258858119927`

### Why Different Precisions?

```python
# Data (float32): Saves memory for large arrays
data = np.random.randn(1000, 1000).astype(np.float32)
# Memory: 1,000,000 × 4 bytes = 4 MB

# Metrics (float64): High precision for validation
entropy = calculate_entropy(data)  # Returns float64
# Memory: 1 × 8 bytes = 8 bytes (negligible)
```

### Viewing Full Precision

```bash
# Python
import h5py
with h5py.File('test.h5', 'r') as f:
    entropy = f['data'].attrs['actual_entropy']
    print(f'{entropy:.15f}')  # 6.014996832588581

# h5dump (command line)
h5dump -a /data/actual_entropy -m "%.15f" test.h5
# Output: 6.014996832588581
```

---

## Troubleshooting

### Issue: Entropy doesn't match target

**Problem:**
```
Target:  6.0000 bits
Actual:  5.2315 bits
Error:   12.8%  ← Too high!
```

**Possible Causes:**
1. Measurement bins too low (use 512+)
2. Data distribution very skewed
3. Target entropy too low (<1.0 bit)

**Solution:**
```bash
# Recalculate with more bins
python calculate_entropy.py file.h5 --bins 1024
```

### Issue: Stored entropy differs from calculated

**Problem:**
```
Stored entropy:    6.0150 bits
Calculated entropy: 6.1293 bits
Differs by 0.1143 bits
```

**Cause:** File was generated with old default (256 bins)

**Solution:** Regenerate dataset with updated code (512 bins default)

### Issue: Validation fails

**Problem:**
```
Entropy: ✗ FAIL
  Target:  2.0000 bits
  Actual:  2.3500 bits
  Error:   17.5%  ← Above 15% tolerance
```

**Cause:** Low entropy targets are harder to hit precisely

**Solution:** This is expected for very low entropy (<2 bits). The quantization algorithm has discrete steps.

---

## Summary

### Key Takeaways

1. **Entropy = Information Content**
   - Measured in bits
   - Higher entropy = more random = less compressible

2. **Two Types of Bins**
   - Generation bins: How many unique values to create
   - Measurement bins: How many histogram bins for measurement

3. **Quantization = Controlled Discretization**
   - Quantile-based approach works for any distribution
   - Creates exactly `2^target_entropy` unique values

4. **The Pipeline**
   - Generate base distribution → Quantize to target entropy → Measure and validate → Store in HDF5

5. **Validation**
   - Target vs Actual should be within ~5% for good generation
   - Use 512+ measurement bins for accuracy

### Quick Reference Commands

```bash
# Generate dataset
python generator.py generate -d normal -e 6.0 --dataset-size 4MB -o test.h5

# Check entropy
python calculate_entropy.py test.h5

# View with h5dump
h5dump -a /data/actual_entropy -m "%.15f" test.h5

# Batch generate
python generator.py batch -c config/presets_small.yaml -o datasets/

# List presets
python generator.py list-presets --config config/presets_small.yaml
```

---

## Further Reading

- **Shannon, C.E. (1948)** - "A Mathematical Theory of Communication"
- **README.md** - Quick start guide and examples
- **Code Documentation** - See docstrings in:
  - `attributes/entropy.py` - Entropy adjustment algorithm
  - `attributes/distributions.py` - Statistical distributions
  - `utils/metrics.py` - Measurement and validation
  - `utils/hdf5_writer.py` - HDF5 I/O operations

---

*Last Updated: 2026-02-04*
*Tool Version: Latest with 512-bin default*
