# Trellis based generator
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ARG GITHUB_USER
ENV GITHUB_USER=${GITHUB_USER}

ARG GITHUB_TOKEN
ENV GITHUB_TOKEN=${GITHUB_TOKEN}

LABEL name="trellis-gen" maintainer="404gen"

# Install docker dependencies
# Installing python 3.11, enbaling channel
RUN apt update -y &&  \
    apt-get install software-properties-common -y &&  \
    apt update -y &&  \
    add-apt-repository ppa:deadsnakes/ppa -y &&  \
    apt update -y

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    gcc-14 \
    g++-14 \
    cmake \
    ninja-build \
    python3.11  \
    python3-pip  \
    python3-wheel \
    libcudnn9-cuda-12 \
    libjpeg-dev  \
    zlib1g-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# allow pip to install packages
RUN rm /usr/lib/python3*/EXTERNALLY-MANAGED

# rebinding names for python, pip, gcc, g++
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 120 &&  \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 120 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 100

# Set environment variables
ENV LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="8.9, 12.0"

# preparing project
RUN mkdir -p /app
WORKDIR /app

RUN git clone https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/404-Repo/404-base-miner.git
WORKDIR /app/trellis-generation
RUN git checkout initial_commit

RUN pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1 packaging
RUN pip install flash-attn==2.8.0.post2 --no-build-isolation --no-cache-dir
RUN pip install flashinfer-python==0.3.0 flashinfer-cubin==0.4.0rc2 --no-build-isolation
RUN pip install -r requiremetns.txt

EXPOSE 10006
CMD ["python",  "serve.py"]