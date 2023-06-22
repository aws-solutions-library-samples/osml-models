# Set the base image to build from
ARG BASE_CONTAINER=public.ecr.aws/amazonlinux/amazonlinux:latest

# Swap BASE_CONTAINER to a container output while building cert-base if you need to override the pip mirror
FROM ${BASE_CONTAINER} as osml_model

############# Inject model selection build configuration parameters #############
# Ensure that a model selection was provided
ARG MODEL_SELECTION
ENV MODEL_SELECTION=$MODEL_SELECTION

# Exit if we didn't find a MODEL_SELECTION value set
RUN if [[ -z "${MODEL_SELECTION}" ]]; then echo 'Argument MODEL_SELECTION must be specified!'; exit 1; fi


############# Set default cert information for pip #############
# Only override if you're using a mirror with a cert pulled in using cert-base as a build parameter
ARG BUILD_CERT=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
ARG PIP_INSTALL_LOCATION=https://pypi.org/simple/

############# Install compilers and C/C++ tools for D2 deps #############
RUN yum groupinstall -y "Development Tools";

############# Install Miniconda3 ############
# Grab wget to pull the miniconda installer
RUN yum install -y wget
ARG MINICONDA_VERSION=Miniconda3-latest-Linux-x86_64
ARG MINICONDA_URL=https://repo.anaconda.com/miniconda/${MINICONDA_VERSION}.sh
RUN wget -c ${MINICONDA_URL} \
    && chmod +x ${MINICONDA_VERSION}.sh \
    && ./${MINICONDA_VERSION}.sh -b -f -p /usr/local

# Clean up installer file
RUN rm ${MINICONDA_VERSION}.sh

############# Install GDAL and python venv to the user profile ############
# This sets the python3 alias to be the miniconda managed python3.10 ENV
ARG PYTHON_VERSION=3.10
RUN conda install -c conda-forge -q -y --prefix /usr/local python=${PYTHON_VERSION} gdal proj

############# Set Proj installation metadata ############
ENV PROJ_LIB=/usr/local/share/proj
RUN chmod 777 --recursive ${PROJ_LIB}

############# Installing latest D2 build dependencies if plane model is selected as the target ############
# Force cuda since it won't be available in Docker build env
ENV FORCE_CUDA="1"
# Build D2 only for Volta architecture - V100 chips (ml.p3 AWS instances)
ENV TORCH_CUDA_ARCH_LIST="Volta"
ARG CUDA_VERSION="11.7.0"

RUN if [ "$MODEL_SELECTION" = "aircraft" ]; \
     then \
      conda install -q -y --prefix /usr/local --channel "nvidia/label/cuda-${CUDA_VERSION}" cuda; \
            python3 -m pip install \
               --index-url ${PIP_INSTALL_LOCATION} \
               --cert ${BUILD_CERT} \
               --upgrade \
               --force-reinstall \
               torch torchvision cython opencv-contrib-python-headless; \
            yum install -y git; \
            python3 -m pip install \
                --index-url ${PIP_INSTALL_LOCATION} \
                --cert ${BUILD_CERT} \
                 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'; \
            python3 -m pip install \
                --index-url ${PIP_INSTALL_LOCATION} \
                --cert ${BUILD_CERT} \
                'git+https://github.com/facebookresearch/fvcore'; \
            python3 -m pip install \
                --index-url ${PIP_INSTALL_LOCATION} \
                --cert ${BUILD_CERT} \
                'git+https://github.com/facebookresearch/detectron2.git'; \
    fi;
# Set a fixed model cache directory. Detectron2 requirement
ENV FVCORE_CACHE="/tmp"


############# Copy control model source code  ############
COPY . /home/
RUN chmod 777 --recursive /home/


############# Setting up application runtime layer #############
# Hop in the home directory where we have copied the source files
WORKDIR /home
RUN python3 -m pip install \
    --index-url ${PIP_INSTALL_LOCATION} \
    --cert ${BUILD_CERT} \
    -r requirements.txt

# Install package module to the instance
RUN python3 setup.py install

# Clean up any dangling conda resources
RUN conda clean -afy

# Make sure we expose our ports
EXPOSE 8080

# Update the PYTHONPATH to ensure the source directory is found
ENV PYTHONPATH="${PYTHONPATH}:./src/:/usr/local/"
# Update the LD_LIBRARY_PATH to ensure the C++ libraries can be found
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib/:/usr/include:/usr/local/"
# Update the PATH to ensure the user bins can be found
ENV PATH="${PATH}:/usr/sbin/:/usr/local/"

# Set the model entry point path
ENV MODEL_ENTRY_POINT="aws.osml.models.${MODEL_SELECTION}.app"

# Create a script to pass command line args to python
RUN echo "python3 -m ${MODEL_ENTRY_POINT} \$@" >> /run_model.sh

# Set the entry point command to the bin we created
ENTRYPOINT ["bash", "/run_model.sh"]

# Build the unit_test stage
FROM osml_model as unit_test

# Hop in the home directory
WORKDIR /home

# Set root user for dep installs
USER root

# Import the source directory to the generalized path
ENV PYTHONPATH="./src/"

# Set the entry point command to run unit tests
RUN echo "python3 -m pytest -vv --cov-report=term-missing --cov=aws.osml.models.${MODEL_SELECTION} test/aws/osml/models/${MODEL_SELECTION}/ \$@" > /run_pytest.sh
CMD ["/bin/bash", "/run_pytest.sh"]

ENTRYPOINT []
