FROM gpucompress-base

ARG CUDA_ARCH=80

# Install PyTorch and torchvision matching CUDA 12.6
RUN pip3 install --break-system-packages -q \
        torch torchvision --index-url https://download.pytorch.org/whl/cu126 \
    && pip3 install --break-system-packages -q numpy
