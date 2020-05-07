.. _installation_instructions:

Installation
=============

Ketos is available on the Python package index repository and the latest version can be installed with pip: ::

    pip install ketos==2.0.0b4


Because Ketos uses TensorFlow as the deep learning framework, at this time it requires pip 20.0 or higher and python 3.6 or 3.7. 
If you have a GPU and would like to take advantage of it, install tensorflow-gpu after installing ketos: ::

    pip install tensorflow-gpu

Note that GPU support depends on CUDA®-enabled graphics cards and the necessary drivers and libraries. Refer to  https://www.tensorflow.org/install/gpu for more information further instructions.


Depending on your operating system, you might have to install other dependencies (like hdf5lib).
If you try the steps above and receive errors due to missing dependencies and don't want to install them yourself, you might find Anaconda helpful. 

Anaconda is freely available from https://docs.anaconda.com/anaconda/install/. 
Make sure you get the Python 3.7 version and make sure to pick the installer appropriate for your OS (Linux, macOS, Windows) 

Clone the Ketos repository: ::

    git clone --depth=1 https://gitlab.meridian.cs.dal.ca/public_projects/ketos.git
    cd ketos

Create and activate Anaconda environment: ::

    conda create --name ketos_env
    conda activate ketos_env
 
Install the PyPI package manager and Jupyter Notebook: ::
    
    conda install pip
    conda install jupyter #if you want to run the executable jupyter notebooks in the tutorials 

Install Ketos: ::
    
    python setup.py sdist
    pip install dist/ketos-2.0.0b4.tar.gz

Check that everything is working by running pytest: ::

    conda install pytest
    pytest ketos/ --doctest-modules

