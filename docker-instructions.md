# Build & Run

### Build
`$ docker build -t sionna-dl-6g .`

### Run on a CPU-only host
`$ docker run -it sionna-dl-6g`

### Run on a GPU host (with NVIDIA Container Toolkit installed)
`$ docker run -it --gpus all sionna-dl-6g`

### Override which script to run
#### If your entry script isn’t mimo_ofdm_over_cdl/mimo_ofdm_over_cdl.py, override as follows:
`$ docker run -it -e RUN_SCRIPT="folder/script.py" sionna-dl-6g`

#### To pass arguments to your script:
`$ docker run -it sionna-dl-6g -- --epochs 5 --batch-size 128`

#### Reuse the existing container and attach
`$ docker start -ai upbeat_wright`  
`-a` attaches to STDOUT/STDERR; `-i` keeps STDIN open.