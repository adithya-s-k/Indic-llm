ARG CUDA_VERSION="11.8.0"
ARG CUDNN_VERSION="8"
ARG UBUNTU_VERSION="22.04"
ARG MAX_JOBS=4

FROM nvidia/cuda:$CUDA_VERSION-cudnn$CUDNN_VERSION-devel-ubuntu$UBUNTU_VERSION as base-builder

ENV PATH="/root/miniconda3/bin:${PATH}"

ARG PYTHON_VERSION="3.9"
ARG PYTORCH_VERSION="2.0.1"
ARG CUDA="118"
ARG TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 9.0+PTX"

ENV PYTHON_VERSION=$PYTHON_VERSION
ENV TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST

RUN apt-get update \
    && apt-get install -y wget git build-essential ninja-build git-lfs libaio-dev && rm -rf /var/lib/apt/lists/* \
    && wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda create -n "py${PYTHON_VERSION}" python="${PYTHON_VERSION}"

ENV PATH="/root/miniconda3/envs/py${PYTHON_VERSION}/bin:${PATH}"

WORKDIR /workspace

RUN python3 -m pip install --upgrade pip && pip3 install packaging && \
    python3 -m pip install --no-cache-dir -U torch==${PYTORCH_VERSION}+cu${CUDA} deepspeed-kernels --extra-index-url https://download.pytorch.org/whl/cu$CUDA

RUN git lfs install --skip-repo && \
    pip3 install awscli && \
    # The base image ships with `pydantic==1.8.2` which is not working
    pip3 install -U --no-cache-dir pydantic==1.10.10

RUN pip install \
    numpy \
    torch \
    jupyterlab

# start jupyter lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
EXPOSE 8888