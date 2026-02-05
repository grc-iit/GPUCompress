#!/bin/bash
#
# GPUCompress Dependencies Installation Script
#
# This script installs all required dependencies for the GPUCompress project.
# Run with: ./scripts/install_dependencies.sh
#
# Requirements:
#   - Ubuntu 20.04/22.04/24.04 (x86_64)
#   - NVIDIA GPU with compute capability >= 7.0
#   - CUDA Toolkit >= 12.0 installed
#   - NVIDIA driver >= 525.60.13
#   - sudo privileges (for apt packages)
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
NVCOMP_VERSION="5.1.0.21"
NVCOMP_CUDA_VERSION="cuda12"
NVCOMP_INSTALL_DIR="/tmp"
NVCOMP_URL="https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-${NVCOMP_VERSION}_${NVCOMP_CUDA_VERSION}-archive.tar.xz"

# ============================================================================
# Pre-flight checks
# ============================================================================

echo_info "Running pre-flight checks..."

# Check if running on Linux x86_64
if [[ "$(uname -s)" != "Linux" ]] || [[ "$(uname -m)" != "x86_64" ]]; then
    echo_error "This script only supports Linux x86_64"
    exit 1
fi

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo_error "nvidia-smi not found. Please install NVIDIA drivers first."
    exit 1
fi

echo_info "Detected GPU:"
nvidia-smi --query-gpu=gpu_name,compute_cap,driver_version --format=csv,noheader

# Check for CUDA toolkit
if ! command -v nvcc &> /dev/null; then
    echo_error "nvcc not found. Please install CUDA Toolkit >= 12.0 first."
    echo_info "Download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
echo_info "Detected CUDA version: ${CUDA_VERSION}"

CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
if [[ "$CUDA_MAJOR" -lt 12 ]]; then
    echo_error "CUDA >= 12.0 required, found ${CUDA_VERSION}"
    exit 1
fi

# Check for cuFile (GDS support)
if [[ -f "/usr/local/cuda/lib64/libcufile.so" ]]; then
    echo_info "cuFile (GPUDirect Storage) library found"
else
    echo_warn "cuFile not found - GPUDirect Storage may not work"
fi

# ============================================================================
# Install system packages
# ============================================================================

echo_info "Installing system packages..."

sudo apt-get update
sudo apt-get install -y \
    cmake \
    build-essential \
    curl \
    xz-utils

echo_info "cmake version: $(cmake --version | head -1)"

# ============================================================================
# Download and install nvcomp
# ============================================================================

echo_info "Installing nvcomp ${NVCOMP_VERSION}..."

# Create install directories
mkdir -p "${NVCOMP_INSTALL_DIR}/include"
mkdir -p "${NVCOMP_INSTALL_DIR}/lib"

# Download nvcomp
NVCOMP_ARCHIVE="/tmp/nvcomp-${NVCOMP_VERSION}.tar.xz"
if [[ ! -f "${NVCOMP_ARCHIVE}" ]]; then
    echo_info "Downloading nvcomp from NVIDIA..."
    curl -L -o "${NVCOMP_ARCHIVE}" "${NVCOMP_URL}"
else
    echo_info "Using cached nvcomp archive"
fi

# Extract nvcomp
NVCOMP_EXTRACT_DIR="/tmp/nvcomp-linux-x86_64-${NVCOMP_VERSION}_${NVCOMP_CUDA_VERSION}-archive"
if [[ -d "${NVCOMP_EXTRACT_DIR}" ]]; then
    rm -rf "${NVCOMP_EXTRACT_DIR}"
fi

echo_info "Extracting nvcomp..."
cd /tmp
tar -xf "${NVCOMP_ARCHIVE}"

# Copy files to install location
echo_info "Installing nvcomp headers and libraries..."
cp -r "${NVCOMP_EXTRACT_DIR}/include/"* "${NVCOMP_INSTALL_DIR}/include/"
cp -r "${NVCOMP_EXTRACT_DIR}/lib/"* "${NVCOMP_INSTALL_DIR}/lib/"

# Verify installation
if [[ -f "${NVCOMP_INSTALL_DIR}/include/nvcomp.hpp" ]] && \
   [[ -f "${NVCOMP_INSTALL_DIR}/lib/libnvcomp.so" ]]; then
    echo_info "nvcomp installed successfully"
else
    echo_error "nvcomp installation failed"
    exit 1
fi

# ============================================================================
# Setup environment
# ============================================================================

echo_info "Setting up environment..."

# Create environment setup script
ENV_SCRIPT="/home/cc/GPUCompress/scripts/setup_env.sh"
cat > "${ENV_SCRIPT}" << 'ENVEOF'
#!/bin/bash
# Source this file to set up the GPUCompress environment
# Usage: source scripts/setup_env.sh

export LD_LIBRARY_PATH=/tmp/lib:${LD_LIBRARY_PATH}
export NVCOMP_INCLUDE_DIR=/tmp/include
export NVCOMP_LIB_DIR=/tmp/lib

echo "GPUCompress environment configured:"
echo "  LD_LIBRARY_PATH includes: /tmp/lib"
echo "  NVCOMP_INCLUDE_DIR: ${NVCOMP_INCLUDE_DIR}"
echo "  NVCOMP_LIB_DIR: ${NVCOMP_LIB_DIR}"
ENVEOF
chmod +x "${ENV_SCRIPT}"

# ============================================================================
# Build project
# ============================================================================

echo_info "Building GPUCompress..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

cd "${PROJECT_DIR}"
rm -rf build
mkdir -p build
cd build

cmake ..
make -j$(nproc)

# ============================================================================
# Verify build
# ============================================================================

echo_info "Verifying build..."

if [[ -x "./gpu_compress" ]] && [[ -x "./gpu_decompress" ]]; then
    echo_info "Build successful!"
    echo ""
    echo "============================================================"
    echo "  GPUCompress Installation Complete"
    echo "============================================================"
    echo ""
    echo "Built executables:"
    echo "  - build/gpu_compress"
    echo "  - build/gpu_decompress"
    echo "  - build/test_quantization"
    echo ""
    echo "Before running, set up the environment:"
    echo "  source scripts/setup_env.sh"
    echo ""
    echo "Example usage:"
    echo "  ./build/gpu_compress input.bin output.lz4 lz4"
    echo "  ./build/gpu_decompress output.lz4 restored.bin"
    echo ""
else
    echo_error "Build verification failed"
    exit 1
fi
