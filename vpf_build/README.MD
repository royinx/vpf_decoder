
> Download [Video_Codec_SDK](https://developer.nvidia.com/nvidia-video-codec-sdk)


```bash
# cp ../../yolox/VideoProcessingFramework/Video_Codec_SDK_11.1.5.tar  .

tar -xvf Video_Codec_SDK_11.1.5.tar
mv Video_Codec_SDK_11.1.5 Video_Codec_SDK

export VIDEO_CODEC_SDK=$PWD/Video_Codec_SDK
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

docker build -f docker/Dockerfile-tensorrt.gpu -t vpf .
docker run -it --runtime=nvidia --name vpf -v ${PWD}/test/:/py -w /py vpf 
python3 nvc.py
```