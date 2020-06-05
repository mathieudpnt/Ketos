# Use an official Python runtime as a parent image
FROM python:3.6-stretch

# Set the working directory to /app
WORKDIR /app

# Copy ketos source code, setup file and requirements file
COPY ketos/ /app/ketos
COPY requirements.txt /app
COPY setup.py /app
COPY sphinx_mer_rtd_theme-0.4.3.dev0.tar.gz /app

# Update base container install
RUN apt-get update

# Install HDF5 C library
RUN apt-get install -y --no-install-recommends libhdf5-serial-dev

# Install sndfile (needed by librosa)
RUN apt-get install -y --no-install-recommends libsndfile1

# Clean
RUN rm -rf /var/lib/apt/lists/*

# Install packages specified in requirements.txt
RUN pip install -r requirements.txt

# Install ketos
RUN python setup.py sdist
RUN pip install dist/ketos-2.0.0b5.tar.gz
RUN pip install sphinx_mer_rtd_theme-0.4.3.dev0.tar.gz

