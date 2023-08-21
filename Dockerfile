FROM public.ecr.aws/amazonlinux/amazonlinux:2023 as osml_model

############# Set default cert information for pip #############
# Only override if you're using a mirror with a cert pulled in using cert-base as a build parameter
ARG BUILD_CERT=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
ARG PIP_INSTALL_LOCATION=https://pypi.org/simple/

# give sudo permissions
USER root

# set working directory to home
WORKDIR /home/

############# Install compilers and C/C++ tools for D2 deps #############
RUN yum groupinstall -y "Development Tools";

############# Install required yum packages for build #############
RUN yum install -y wget git

############# Install Miniconda3 ############
ARG MINICONDA_VERSION=Miniconda3-latest-Linux-x86_64
ARG MINICONDA_URL=https://repo.anaconda.com/miniconda/${MINICONDA_VERSION}.sh
RUN wget -c ${MINICONDA_URL} \
    && chmod +x ${MINICONDA_VERSION}.sh \
    && ./${MINICONDA_VERSION}.sh -b -f -p /opt/conda \
    && rm ${MINICONDA_VERSION}.sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# add conda and local installs to the path so we can execute them
ENV PATH=/usr/local/:/usr/local/bin:/opt/conda/bin:$PATH

# update the LD_LIBRARY_PATH to ensure the C++ libraries can be found
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/:/usr/include:/usr/local/:/usr/local/bin

# disable NNPACK since we don't do training with this container
ENV USE_NNPACK=0

# set local project directroy
ENV PROJ_LIB=/usr/local/share/proj

# copy our conda env configuration for Python 3.10
COPY environment-py311.yml environment.yml

# create the conda env
RUN conda env create

# create /entry.sh which will be our new shell entry point
# this performs actions to configure the environment
# before starting a new shell (which inherits the env).
# the exec is important as this allows signals to passpw
ENV CONDA_TARGET_ENV=osml_models
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

############# Installing latest D2 build dependencies if plane model is selected as the target ############
# Force cuda since it won't be available in Docker build env
ENV FORCE_CUDA="1"
# build D2 only for Volta architecture - V100 chips (ml.p3 AWS instances)
ENV TORCH_CUDA_ARCH_LIST="Volta"

RUN python3 -m pip install \
           --index-url ${PIP_INSTALL_LOCATION} \
           --cert ${BUILD_CERT} \
           --upgrade \
           --force-reinstall \
           torch==2.0.1 torchvision==0.15.2 cython==3.0.0 opencv-contrib-python-headless==4.8.0.76;

RUN python3 -m pip install \
            --index-url ${PIP_INSTALL_LOCATION} \
            --cert ${BUILD_CERT} \
             'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI';

RUN python3 -m pip install \
            --index-url ${PIP_INSTALL_LOCATION} \
            --cert ${BUILD_CERT} \
            'git+https://github.com/facebookresearch/fvcore';

RUN python3 -m pip install \
            --index-url ${PIP_INSTALL_LOCATION} \
            --cert ${BUILD_CERT} \
            'git+https://github.com/facebookresearch/detectron2.git';

# set a fixed model cache directory. Detectron2 requirement
ENV FVCORE_CACHE="/tmp"

############# Copy control model source code  ############
COPY . .
RUN chmod 777 --recursive .

############# Setting up application runtime layer #############
# hop in the home directory where we have copied the source files
RUN python3 -m pip install \
    --index-url ${PIP_INSTALL_LOCATION} \
    --cert ${BUILD_CERT} \
    .

# clean up any dangling conda resources
RUN conda clean -afy

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