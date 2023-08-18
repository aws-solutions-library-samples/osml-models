FROM public.ecr.aws/amazonlinux/amazonlinux:2023 as osml_model

# only override if you're using a mirror with a cert pulled in using cert-base as a build parameter
ARG BUILD_CERT=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
ARG PIP_INSTALL_LOCATION=https://pypi.org/simple/

# give sudo permissions
USER root

# set working directory to home
WORKDIR /home/

# configure, update, and refresh yum enviornment
RUN yum update -y && yum clean all && yum makecache && yum install -y wget shadow-utils

# install dev tools and compiler resources
RUN yum groupinstall -y "Development Tools";

# install miniconda
ARG MINICONDA_VERSION=Miniconda3-latest-Linux-x86_64
ARG MINICONDA_URL=https://repo.anaconda.com/miniconda/${MINICONDA_VERSION}.sh
RUN wget -c ${MINICONDA_URL} \
    && chmod +x ${MINICONDA_VERSION}.sh \
    && ./${MINICONDA_VERSION}.sh -b -f -p /opt/conda \
    && rm ${MINICONDA_VERSION}.sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# set all the ENV vars needed for build
ENV LIB_PATHS=/opt/conda/bin:/opt/conda/lib/:/usr/include:/usr/local/
ENV PATH=$LIB_PATHS:$PATH
ENV CONDA_TARGET_ENV=osml_models
ENV TORCH_CUDA_ARCH_LIST=Volta
ENV FVCORE_CACHE="/tmp"
ENV CC="clang"
ENV CXX="clang++"
ENV ARCHFLAGS="-arch x86_64"
ENV LD_LIBRARY_PATH=$LIB_PATHS:$LD_LIBRARY_PATH
ENV PROJ_LIB=/opt/conda/share/proj
ENV FVCORE_CACHE="/tmp"
ENV FORCE_CUDA="1"
ENV USE_NNPACK=0

# copy conda env source for Python 3.11
COPY environment-py311.yml environment.yml

# create the conda env
RUN conda env create

# create /entry.sh which will be our new shell entry point
# this performs actions to configure the environment
# before starting a new shell (which inherits the env).
# the exec is important! this allows signals to pass
RUN     (echo '#!/bin/bash' \
    &&   echo '__conda_setup="$(/opt/conda/bin/conda shell.bash hook 2> /dev/null)"' \
    &&   echo 'eval "$__conda_setup"' \
    &&   echo 'conda activate "${CONDA_TARGET_ENV:-base}"' \
    &&   echo '>&2 echo "ENTRYPOINT: CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV}"' \
    &&   echo 'exec "$@"'\
        ) >> /entry.sh && chmod +x /entry.sh

# tell the docker build process to use this for RUN.
# the default shell on Linux is ["/bin/sh", "-c"], and on Windows is ["cmd", "/S", "/C"]
SHELL ["/entry.sh", "/bin/bash", "-c"]

# configure .bashrc to drop into a conda env and immediately activate our TARGET env
RUN conda init && echo 'conda activate "${CONDA_TARGET_ENV:-base}"' >>  ~/.bashrc

RUN python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# copy our application source
COPY . .

# install the application
RUN python3 -m pip install .

# this is a hotfix until the most recent detectron2 udpates reach conda-forge
# https://github.com/facebookresearch/detectron2/commit/ff53992b1985b63bd3262b5a36167098e3dada02
RUN sed -i "s|Image.LINEAR|Image.BILINEAR |g" /opt/conda/envs/osml_models/lib/python3.11/site-packages/detectron2/data/transforms/transform.py

# this is a hotfix until facebookresearch fixes their telemetry logging package
# https://github.com/facebookresearch/iopath/issues/21
RUN sed -i "s|handler.log_event()|pass|g" /opt/conda/envs/osml_models/lib/python3.11/site-packages/iopath/common/file_io.py

# make sure we expose our ports
EXPOSE 8080

# set up a health check at that port
HEALTHCHECK NONE

# set up a user to run the container as and assume it
RUN adduser model
RUN chown -R model:model ./
USER model

# set the entry point script
ENTRYPOINT ["/entry.sh", "/bin/bash", "-c", "python3 -m aws.osml.models.${MODEL_SELECTION}.app"]
