# Build & Run 
##### (run from base directory level)

### Build
#### CPU build
`$ docker build \`<br>
`  --build-arg BASE_IMAGE=ubuntu:24.04 \`<br>
`  --build-arg ENABLE_GPU=0 \`<br>
`  --build-arg ENABLE_CUDA_CHECK=0 \`<br>
`  --build-arg TF_PACKAGE=tensorflow-cpu \`<br>
`  --build-arg TF_VERSION=2.15.1 \`<br>
`  -t nextgen-wireless-dl-demos:cpu .`
#### GPU build
`$ docker build \`<br>
`  --build-arg BASE_IMAGE=nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 \`<br>
`  --build-arg ENABLE_GPU=1 \`<br>
`  --build-arg TF_PACKAGE=tensorflow \`<br>
`  --build-arg TF_VERSION=2.15.1 \`<br>
`  -t nextgen-wireless-dl-demos:gpu-12.2 .`

### Run on a CPU-only host
`$ docker run -it -v "$PWD":/app -w /app nextgen-wireless-dl-demos:cpu bash`

### Run on a GPU host (with NVIDIA Container Toolkit installed)
`$ docker run -it --gpus all \`<br>
`  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \`<br>
`  -e DRJIT_LIBOPTIX_PATH=/usr/lib/x86_64-linux-gnu/libnvoptix.so.1 \`<br>
`  -v "$PWD":/app -w /app nextgen-wireless-dl-demos:gpu-12.2`