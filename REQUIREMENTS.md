# GPUCompress Requirements and Installation Guide

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Ubuntu 20.04 x86_64 | Ubuntu 22.04/24.04 x86_64 |
| GPU | NVIDIA GPU (Compute Capability >= 7.0) | NVIDIA A100/H100 |
| NVIDIA Driver | >= 525.60.13 | Latest stable |
| CUDA Toolkit | >= 12.0 | 12.6+ |
| RAM | 8 GB | 16 GB+ |
| Disk | 1 GB free | 5 GB+ free |

## Dependencies

### Required

| Package | Version | Purpose |
|---------|---------|---------|
| CUDA Toolkit | >= 12.0 | GPU compilation and runtime |
| nvcomp | 5.1.0 | GPU compression library |
| cuFile | (included with CUDA) | GPUDirect Storage |
| cmake | >= 3.18 | Build system |
| g++ | >= 9.0 | C++ compiler |

### Optional

| Package | Purpose |
|---------|---------|
| NVIDIA Nsight Systems | Performance profiling |
| GDS drivers | Direct GPU-to-storage I/O |

## Quick Installation

Run the automated installation script:

```bash
cd /home/cc/GPUCompress
chmod +x scripts/install_dependencies.sh
./scripts/install_dependencies.sh
```

This script will:
1. Verify system requirements
2. Install cmake and build tools
3. Download and install nvcomp 5.1.0
4. Build the project

## Manual Installation

### Step 1: Install CUDA Toolkit

If not already installed, download from:
https://developer.nvidia.com/cuda-downloads

Verify installation:
```bash
nvcc --version
nvidia-smi
```

### Step 2: Install Build Tools

```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential curl xz-utils
```

### Step 3: Install nvcomp

Download and extract nvcomp 5.1.0:

```bash
# Create directories
mkdir -p /tmp/include /tmp/lib

# Download nvcomp
curl -L -o /tmp/nvcomp.tar.xz \
  "https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-5.1.0.21_cuda12-archive.tar.xz"

# Extract
cd /tmp
tar -xf nvcomp.tar.xz

# Install
cp -r nvcomp-linux-x86_64-5.1.0.21_cuda12-archive/include/* /tmp/include/
cp -r nvcomp-linux-x86_64-5.1.0.21_cuda12-archive/lib/* /tmp/lib/
```

### Step 4: Build the Project

```bash
cd /home/cc/GPUCompress
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### Step 5: Set Up Environment

Before running executables, set the library path:

```bash
# Option A: Source the setup script
source scripts/setup_env.sh

# Option B: Set manually
export LD_LIBRARY_PATH=/tmp/lib:$LD_LIBRARY_PATH
```

For permanent setup, add to `~/.bashrc`:
```bash
echo 'export LD_LIBRARY_PATH=/tmp/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Verification

Test the installation:

```bash
# Check executables exist
ls -la build/gpu_compress build/gpu_decompress

# Show help
./build/gpu_compress --help

# Run quantization tests
./build/test_quantization
```

## Troubleshooting

### nvcomp library not found at runtime

```
error while loading shared libraries: libnvcomp.so.5: cannot open shared object file
```

**Solution**: Set LD_LIBRARY_PATH:
```bash
export LD_LIBRARY_PATH=/tmp/lib:$LD_LIBRARY_PATH
```

### CUDA not found during cmake

```
Could not find CUDAToolkit
```

**Solution**: Ensure CUDA is in PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### GDS/cuFile errors

If GPUDirect Storage is not available, the application will fall back to standard I/O.
For full GDS support:
1. Ensure filesystem supports O_DIRECT (ext4, xfs)
2. Install GDS drivers: https://docs.nvidia.com/gpudirect-storage/

Check GDS status:
```bash
/usr/local/cuda/gds/tools/gdscheck -p
```

### Build errors with nvcomp API

The code is tested with nvcomp 5.1.0. If using a different version, API changes
may cause compilation errors. Check `src/CompressionFactory.cpp` for the
compression manager initialization code.

## nvcomp Supported Algorithms

| Algorithm | Best For | Speed | Ratio |
|-----------|----------|-------|-------|
| lz4 | General purpose | Very Fast | Medium |
| snappy | Speed critical | Fastest | Low |
| deflate | Compatibility | Slow | High |
| gdeflate | GPU-optimized gzip | Slow | High |
| zstd | Best ratio | Medium | Very High |
| ans | Numerical data | Medium | High |
| cascaded | Floating-point | Medium | Very High |
| bitcomp | Scientific data | Fast | High |

## Project Structure

```
GPUCompress/
├── src/
│   ├── GPU_Compress.cpp      # Main compression executable
│   ├── GPU_Decompress.cpp    # Main decompression executable
│   ├── CompressionFactory.cpp/.hpp  # nvcomp algorithm factory
│   ├── byte_shuffle_kernels.cu      # Preprocessing kernels
│   └── quantization_kernels.cu      # Quantization kernels
├── scripts/
│   ├── install_dependencies.sh      # Automated setup
│   └── setup_env.sh                 # Environment setup
├── tests/
│   └── quantization/                # Test suite
├── build/                           # Build output (created by cmake)
├── CMakeLists.txt
├── REQUIREMENTS.md                  # This file
└── README.md
```

## References

- [nvcomp Documentation](https://docs.nvidia.com/cuda/nvcomp/)
- [nvcomp Downloads](https://developer.nvidia.com/nvcomp-download)
- [GPUDirect Storage](https://docs.nvidia.com/gpudirect-storage/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
