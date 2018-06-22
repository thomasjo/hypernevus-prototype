# Intended to be used together with a custom TensorFlow container image;
# https://github.com/thomasjo/dockerfiles/tree/master/tensorflow
FROM tensorflow:latest AS default

# Ensure all base packages are up-to-date.
RUN apt-get update && apt-get upgrade --yes

# Install basic tools for a sensible workflow.
RUN apt-get update && apt-get install --yes \
    build-essential \
    ca-certificates \
    curl \
    git \
    software-properties-common \
&&  rm -rf /var/lib/apt/lists/*

# Install GDAL development dependencies.
RUN add-apt-repository -y ppa:ubuntugis/ubuntugis-unstable \
&&  apt-get update && apt-get install --yes \
    libgdal-dev=2.2.2+dfsg-1~xenial1 \
&&  rm -rf /var/lib/apt/lists/*

# Expose GDAL include paths to C/C++ toolchain.
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal

# Install Python 3 development headers, etc.
RUN apt-get update && apt-get install --yes \
    python3-dev \
&&  rm -rf /var/lib/apt/lists/*

# Install required Python 3 packages.
RUN apt-get update && apt-get install --yes \
    python3-pyqt5 \
&&  pip3 install --no-cache-dir \
    click \
    gdal==2.2.2 \
    h5py \
    keras \
    matplotlib \
    pandas \
    scikit-image \
    scikit-learn \
    scipy \
    spectral

# Install fpipy package from its GitHub repository.
RUN git clone https://github.com/silmae/fpipy.git \
&&  cd fpipy && python3 setup.py install \
&&  cd .. && rm -rf fpipy

# Install CMake from binary tarball.
RUN curl -Lo /tmp/cmake.sh https://cmake.org/files/v3.11/cmake-3.11.4-Linux-x86_64.sh
RUN sh /tmp/cmake.sh --prefix=/usr/local --skip-license

# Install Eigen3 library.
RUN apt-get update && apt-get install --yes \
    libeigen3-dev \
&&  rm -rf /var/lib/apt/lists/*

# Explictly set the working directory to something reasonable.
WORKDIR /root

# Copy project files into the image.
ADD python /root/python

# ---------------------------------------------------------------------------- #

# Extend the base image with Jupyter Notebook support.
FROM default AS notebook

# Install required Python 3 packages.
RUN pip3 install --no-cache-dir \
    jupyter

# Copy project files into the image.
ADD notebooks /root/notebooks

# Launch Jupyter Notebook by default.
# NOTE: We might want to extract this into a shell script.
CMD jupyter-notebook \
    --allow-root \
    --no-browser \
    --port=8888 \
    --notebook-dir="/root/notebooks"
