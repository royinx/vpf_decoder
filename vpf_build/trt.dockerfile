FROM nvcr.io/nvidia/tensorrt:22.07-py3

ARG DEBIAN_FRONTEND=noninteractive
# RUN apt-get update && apt-get install -y libgl1-mesa-glx \
#                                          libglib2.0-0 \
#                                         libssl-dev\
#                                         libavfilter-dev \
#                                         libavformat-dev \
#                                         libavcodec-dev \
#                                         libswresample-dev \
#                                         libavutil-dev\
#                                         ninja-build \
#                                         wget \
#                                         cmake \
#                                         build-essential \
#                                         libnvidia-encode-515 \
#                                         libnvidia-decode-515 \
#                                         git
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 build-essential ffmpeg

RUN python3 -m pip install cupy-cuda11x \
                           scikit-learn \
                           networkx \
                           lap \
                           opencv-python \
                           line_profiler \
                           loguru \
                           bbox-visualizer




RUN apt-get update && apt-get install -y libssl-dev\
                                          libavfilter-dev \
                                          libavformat-dev \
                                          libavcodec-dev \
                                          libswresample-dev \
                                          libavutil-dev\
                                          ninja-build \
                                          wget \
                                          cmake \
                                          build-essential \
                                          libnvidia-encode-515 \
                                          libnvidia-decode-515 \
                                          git
                    
RUN wget https://github.com/Kitware/CMake/releases/download/v3.25.2/cmake-3.25.2.tar.gz && \
        tar -zxvf cmake-3.25.2.tar.gz && \
        cd cmake-3.25.2 && \
        ./bootstrap && \
        make && \
        make install && \
        cd .. && rm -rf cmake-3.25.2 cmake-3.25.2.tar.gz
        
RUN pip3 install git+https://github.com/NVIDIA/VideoProcessingFramework
