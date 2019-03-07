FROM ubuntu:18.04 AS default

# Ensure apt-get is in noninteractive mode during build.
ARG DEBIAN_FRONTEND=interactive

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

# Install CMake from binary tarball.
RUN curl -Lo /tmp/cmake.sh https://cmake.org/files/v3.12/cmake-3.12.4-Linux-x86_64.sh
RUN sh /tmp/cmake.sh --prefix=/usr/local --skip-license

# Install Eigen3 library.
RUN apt-get update && apt-get install --yes \
    libeigen3-dev \
&&  rm -rf /var/lib/apt/lists/*

# Install GDAL development dependencies.
RUN apt-get update && apt-get install --yes \
    libgdal-dev=2.2.3+dfsg-2 \
&&  rm -rf /var/lib/apt/lists/*

# Expose GDAL include paths to C/C++ toolchain.
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal

# Install Python 3 development headers, etc.
RUN apt-get update && apt-get install --yes \
    python3 \
    python3-dev \
&&  rm -rf /var/lib/apt/lists/*

# Install latest version of pip.
RUN curl https://bootstrap.pypa.io/get-pip.py | python3

# Install required Python 3 packages.
RUN apt-get update && apt-get install --yes \
    python3-pyqt5 \
&&  pip3 install --no-cache-dir \
    click \
    gdal==2.2.3 \
    h5py \
    matplotlib \
    pandas \
    pyamg \
    scikit-image \
    scikit-learn \
    scipy \
    spectral

# Install fpipy package from its GitHub repository.
RUN git clone https://github.com/silmae/fpipy.git \
# Patch a small bug in setup.py.
# TODO: Report upstream.
&&  sed -i "s/include=\['fpipy'\]/include=\['fpipy','fpipy\.\*'\]/g" fpipy/setup.py \
&&  cd fpipy && pip3 install --no-cache-dir . \
&&  cd .. && rm -rf fpipy

# Explictly set the working directory to something reasonable.
WORKDIR /root

# Copy project files into the image.
ADD python /root/python

# Prevent Python from writing byte code to disk.
ENV PYTHONDONTWRITEBYTECODE=true

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
