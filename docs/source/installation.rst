.. _installation_instructions:

Installation
=============

Ketos is most easily installed using the Anaconda package manager.
Anaconda is freely available from `docs.anaconda.com/anaconda/install <https://docs.anaconda.com/anaconda/install/>`_. 
Make sure you get the Python 3.7 version and make sure to pick the installer appropriate for your OS (Linux, macOS, Windows) 

Clone the Ketos repository: ::

    git clone https://gitlab.meridian.cs.dal.ca/public_projects/ketos.git
    cd ketos

Create and activate Anaconda environment: ::

    conda create --name ketos_env
    conda activate ketos_env
 
Install the PyPI package manager and Jupyter Notebook: ::
    
    conda install pip
    conda install jupyter

Install Ketos: ::
    
    python setup.py sdist
    pip install dist/ketos-2.0.0b4.tar.gz

Check that everything is working by running pytest: ::

    pytest ketos/ --doctest-modules

