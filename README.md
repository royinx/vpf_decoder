
## Quick Start
### Environment setting
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