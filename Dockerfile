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
# Grab wget to pull the miniconda installer
ARG MINICONDA_VERSION=Miniconda3-latest-Linux-x86_64
ARG MINICONDA_URL=https://repo.anaconda.com/miniconda/${MINICONDA_VERSION}.sh
RUN wget -c ${MINICONDA_URL} \
    && chmod +x ${MINICONDA_VERSION}.sh \
    && ./${MINICONDA_VERSION}.sh -b -f -p /usr/local \
    && rm ${MINICONDA_VERSION}.sh

# Update the LD_LIBRARY_PATH to ensure the C++ libraries can be found
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib/:/usr/include:/usr/local/"

# Update the PATH to ensure the user bins can be found
ENV PATH="${PATH}:/usr/local/"

############# Install GDAL and python venv to the user profile ############
# This sets the python3 alias to be the miniconda managed python3.10 ENV
ARG PYTHON_VERSION=3.11
ARG CUDA_VERSION=11.7.0
ARG GDAL_VERSION=3.7.1
ARG PROJ_VERSION=9.2.1

# Restrict the conda channel to reduce package incompatibility problems
RUN conda config --set channel_priority strict

RUN conda install -c "nvidia/label/cuda-${CUDA_VERSION}" -c conda-forge -q -y --prefix /usr/local \
    python=${PYTHON_VERSION} \
    gdal=${GDAL_VERSION} \
    proj=${PROJ_VERSION} \
    cuda=${CUDA_VERSION}

############# Set Proj installation metadata ############
ENV PROJ_LIB=/usr/local/share/proj
RUN chmod 777 --recursive ${PROJ_LIB}

############# Installing latest D2 build dependencies if plane model is selected as the target ############
# Force cuda since it won't be available in Docker build env
ENV FORCE_CUDA="1"
# Build D2 only for Volta architecture - V100 chips (ml.p3 AWS instances)
ENV TORCH_CUDA_ARCH_LIST="Volta"
# Disable NNPACK since we don't do training with this container
ENV USE_NNPACK=0

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

# Set a fixed model cache directory. Detectron2 requirement
ENV FVCORE_CACHE="/tmp"


############# Copy control model source code  ############
COPY . .
RUN chmod 777 --recursive .


############# Setting up application runtime layer #############
# Hop in the home directory where we have copied the source files
RUN python3 -m pip install \
    --index-url ${PIP_INSTALL_LOCATION} \
    --cert ${BUILD_CERT} \
    .

# Clean up any dangling conda resources
RUN conda clean -afy

# Make sure we expose our ports
EXPOSE 8080

############# Inject model selection build configuration parameters #############
# Ensure that a model selection was provided and set the entry point
ARG MODEL_SELECTION
ENV MODEL_SELECTION=$MODEL_SELECTION
ENV MODEL_ENTRY_POINT="aws.osml.models.${MODEL_SELECTION}.app"

# Create a script to pass command line args to python
RUN echo "python3 -m ${MODEL_ENTRY_POINT} \$@" >> /run_model.sh

# Set the entry point command to the bin we created
ENTRYPOINT ["bash", "/run_model.sh"]
