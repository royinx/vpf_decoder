
# VPF decoder
Use VPF to decode video on NVIDIA GPU, and return cupy array / GPU pointer directly
### Quick Start
---
```bash
docker build -t vpf_trt -f trt.dockerfile .
docker run --rm \
           --runtime=nvidia \
           --name=vanjie_trt \
           --privileged \
           -it \
           -v $PWD:/py/ \
           -w /py \
           vpf_trt bash

python3 video_decoder.py
```