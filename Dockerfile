# Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

# Use NVIDIA's CUDA base image
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu18.04 as osml_model

# Set AWS to the maintainer
LABEL maintainer="Amazon Web Services"

# Enable sudo access for the build session
USER root

# Update package manager and install core build deps
RUN apt-get update -y \
    && apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --fix-missing --no-install-recommends \
            software-properties-common build-essential ca-certificates \
            git make cmake wget unzip libtool automake \
            zlib1g-dev libsqlite3-dev pkg-config sqlite3 libcurl4-gnutls-dev \
            libtiff5-dev

# Install miniconda to manage our venv
ARG MINICONDA_VERSION=Miniconda3-latest-Linux-x86_64
ARG MINICONDA_URL=https://repo.anaconda.com/miniconda/${MINICONDA_VERSION}.sh
ENV CONDA_TARGET_ENV=osml_model
RUN wget -c ${MINICONDA_URL} \
    && chmod +x ${MINICONDA_VERSION}.sh \
    && ./${MINICONDA_VERSION}.sh -b -f -p /opt/conda \
    && rm ${MINICONDA_VERSION}.sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# Set our new conda target lib dirs
ENV PATH=$PATH:/opt/conda/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib/
ENV PROJ_LIB=$PROJ_LIB:/opt/conda/share/proj

# Copy our conda env which includes Python 3.10, GDAL, and PROJ
COPY environment-py310.yml environment.yml

# Create the conda env
RUN conda env create

# Create /entry.sh which will be our new shell entry point
# that performs actions to configure the environment on each RUN
RUN     (echo '#!/bin/bash' \
    &&   echo '__conda_setup="$(/opt/conda/bin/conda shell.bash hook 2> /dev/null)"' \
    &&   echo 'eval "$__conda_setup"' \
    &&   echo 'conda activate "${CONDA_TARGET_ENV:-base}"' \
    &&   echo 'exec "$@"'\
        ) >> /entry.sh && chmod +x /entry.sh

# Tell the docker build process to use this for RUN
# The default shell on Linux is ["/bin/sh", "-c"], and on Windows is ["cmd", "/S", "/C"]
SHELL ["/entry.sh", "/bin/bash", "-c"]

# Configure .bashrc to drop into a conda env and immediately activate our TARGET env
# Note this makes python3 default to our conda managed python version
RUN conda init && echo 'conda activate "${CONDA_TARGET_ENV:-base}"' >>  ~/.bashrc

# Install detectron2 dependencies
RUN python3 -m pip install \
    "fvcore>=0.1.5,<0.1.6" \
    iopath==0.1.8 \
    pycocotools \
    omegaconf==2.1.1 \
    hydra-core==1.1.1 \
    black==21.4b2 \
    termcolor==1.1.0 \
    matplotlib==3.5.2 \
    yacs==0.1.8 \
    tabulate==0.8.9 \
    cloudpickle==2.0.0 \
    tqdm==4.62.3 \
    tensorboard==2.8.0 \
    opencv-contrib-python-headless==4.8.0.76

# Torch needs special GPU enabled versions
RUN python3 -m pip install \
    torch==1.12.0+cu116 \
    torchvision==0.13.0+cu116 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install detectron2 from source
ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
RUN python3 -m pip install --no-deps 'git+https://github.com/facebookresearch/detectron2.git'

# Cleanup to reduce the image size
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    conda clean -afy && \
    python -m pip cache purge

# Copy model source and install it
RUN mkdir /home/osml-models
COPY . /home/osml-models

# Install Detectron2
WORKDIR  /home/osml-models
RUN chmod 777 --recursive .
RUN python3 -m pip install .

# Make sure we expose our ports
EXPOSE 8080

# Set up a health check at that port
HEALTHCHECK NONE

RUN python3 -m pip install opencv-contrib-python-headless==4.8.0.76
# Set up a user to run the container as and assume it
RUN adduser --system --no-create-home --group model
RUN chown -R model:model ./
USER model

# Set the entry point script
ENTRYPOINT python3 src/aws/osml/models/$MODEL_SELECTION/app.py
