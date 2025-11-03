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
`  -t sionna-dl-6g:cpu .`
#### GPU build
`$ docker build \`<br>
`  --build-arg BASE_IMAGE=nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 \`<br>
`  --build-arg ENABLE_GPU=1 \`<br>
`  --build-arg TF_PACKAGE=tensorflow \`<br>
`  --build-arg TF_VERSION=2.15.1 \`<br>
`  -t sionna-dl-6g:gpu-12.2 .`

### Run on a CPU-only host
`$ docker run -it -v "$PWD":/app -w /app sionna-dl-6g:cpu bash`

### Run on a GPU host (with NVIDIA Container Toolkit installed)
`$ docker run -it --gpus all -v "$PWD":/app -w /app sionna-dl-6g:gpu-12.2 bash`