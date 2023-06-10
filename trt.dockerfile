# syntax=docker/dockerfile:experimental
FROM docker.io/jrottenberg/ffmpeg:snapshot-ubuntu as ffmpeg
FROM nvcr.io/nvidia/tensorrt:22.07-py3 as build
# FROM nvcr.io/nvidia/tensorrt:21.08-py3 as build

ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_VERSION=1.1.13

ENV PATH="/opt/.venv/bin:$POETRY_HOME/bin:$PATH" \
    PYTHON_BINARY="/opt/.venv/bin/python"

ARG GEN_PYTORCH_EXT
ARG GEN_OPENGL_EXT

RUN apt update && DEBIAN_FRONTEND=noninteractive \
    apt -y install \
    git wget cmake \
    build-essential curl \
    libmp3lame-dev libtheora-dev libvorbis-dev \
    python3.8 python3-pip python3.8-venv && \
    rm -rf /var/lib/apt/lists/*

# https://python-poetry.org/docs/
WORKDIR $POETRY_HOME
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /opt

COPY vpf_build/docker/Makefile docker/Makefile
COPY vpf_build/poetry.lock poetry.lock
COPY vpf_build/pyproject.toml pyproject.toml
COPY vpf_build/docker/poetry-env.sh docker/poetry-env.sh

# Build poetry virtual environment
RUN bash docker/poetry-env.sh PYTHON_BINARY=$PYTHON_BINARY GEN_PYTORCH_EXT="$GEN_PYTORCH_EXT" GEN_OPENGL_EXT="$GEN_OPENGL_EXT"

# Build vpf
COPY vpf_build .
COPY --from=ffmpeg /usr/local /opt/ffmpeg

RUN make -f docker/Makefile vpf_built \
                            FFMPEG=/opt/ffmpeg \
                            PYTHON_BINARY=$PYTHON_BINARY \
                            VIDEO_CODEC_SDK=/opt/Video_Codec_SDK \
                            GEN_PYTORCH_EXT=$GEN_PYTORCH_EXT

FROM nvcr.io/nvidia/tensorrt:22.07-py3

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y install \
    git build-essential libbsd0 \
    wget cmake libtbb-dev \
    libjpeg8-dev libpng-dev libtiff-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libdc1394-22-dev libssl-dev \
    libxine2-dev libv4l-dev \
    libboost-all-dev libdc1394-22-dev \
    python3.8 python3-pip python3.8-venv curl

WORKDIR /opt

COPY --from=build /opt/dist /opt/dist
COPY --from=build /opt/poetry /opt/poetry
COPY --from=build /opt/ffmpeg /opt/ffmpeg
# Normally we'd reinstall the enviroment but pycuda is built with cuda headers
# which are not available in the nvidia runtime image
COPY --from=build /opt/.venv /opt/.venv
COPY . .

ENV POETRY_HOME="/opt/poetry"

ENV PATH="/opt/.venv/bin:$POETRY_HOME/bin:$PATH" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/ffmpeg/lib:/opt/dist/bin:/opt/.venv/lib/python3.8/site-packages/torch/lib" \
    PYTHONPATH=/opt/.venv/bin:/opt/dist/bin:/usr/local/lib/python3.8/dist-packages

RUN python3 -m pip install cupy-cuda11x \
                           opencv-python \
                           line_profiler \
                           loguru