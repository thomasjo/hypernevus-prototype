# Intended to be used together with a custom TensorFlow container image;
# https://github.com/thomasjo/dockerfiles/tree/master/tensorflow
FROM tensorflow:latest

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
    jupyter \
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

# Explictly set the working directory to something reasonable.
WORKDIR /root

# Copy all project files into image.
COPY notebooks python /root/

# Set the default entrypoint to Bash for the time being. Might make sense to
# change this to /usr/bin/python3 unless we introduce a script to simplify the
# process of running experiments, etc.
ENTRYPOINT ["/bin/sh", "-c"]
CMD ["bash"]
