# Build & Run

### Build
#### CPU build
`$ docker build --build-arg BASE_IMAGE=ubuntu:24.04 --build-arg TF_PACKAGE=tensorflow-cpu -t sionna-dl-6g:cpu .`
#### GPU build
`$ docker build --build-arg BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 --build-arg TF_PACKAGE=tensorflow --build-arg TF_VERSION=2.15.* -t sionna-dl-6g:gpu-11.8 .`

### Run on a CPU-only host
`$ docker run -it -v "$PWD":/app -w /app sionna-dl-6g:cpu bash`

### Run on a GPU host (with NVIDIA Container Toolkit installed)
`$ docker run -it --gpus all -v "$PWD":/app -w /app sionna-dl-6g:gpu-11.8 bash`