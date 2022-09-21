# Use an official Python runtime as a parent image
FROM python:3.9.10

# Set the working directory to /app
WORKDIR /app

# Copy ketos source code, setup file and requirements file
COPY ketos/ /app/ketos
COPY requirements.txt /app
COPY setup.py /app

# Update base container install
RUN apt-get update

# Install HDF5 C library
RUN apt-get install -y --no-install-recommends libhdf5-serial-dev

# Install sndfile (needed by librosa)
RUN apt-get install -y --no-install-recommends libsndfile1

# Clean
RUN rm -rf /var/lib/apt/lists/*

# Upgrade to latest version of pip
RUN pip install pip --upgrade

# Install packages specified in requirements.txt
RUN pip install -r requirements.txt

# Install Scipy>=1.6.0 and aural-features
RUN pip install scipy>=1.6.0
RUN pip install aural-features

# Install ketos
RUN python setup.py sdist
RUN pip install dist/ketos-2.6.1.tar.gz

