# Welcome to Ketos, a deep learning package for underwater acoustics

Ketos provides a unified, high-level interface for working with acoustic data and deep neural networks, 
its main purpose being to support the development of deep learning models for detection and 
classification problems in underwater acoustics.

Ketos is written in Python and utilizes a number of powerful software packages 
including NumPy, HDF5, and Tensorflow. It is licensed under the [GNU GPLv3 licens](https://www.gnu.org/licenses/) 
and hence freely available for anyone to use and modify. The project is hosted on GitLab at 
[gitlab.meridian.cs.dal.ca/public_projects/ketos](https://gitlab.meridian.cs.dal.ca/public_projects/ketos). 
For more information, please consult [Ketos' Documentation Page](https://docs.meridian.cs.dal.ca/ketos/).

Ketos was developed by the [MERIDIAN](http://meridian.cs.dal.ca/) Data Analytics Team at the 
[Institute for Big Data Analytics](https://bigdata.cs.dal.ca/) at Dalhousie University. 
We are greatful to Amalis Riera and Francis Juanes at the University of Victoria, 
Kim Davies and Chris Taggart at Dalhousie University, and Kristen Kanes at Ocean Networks Canada 
for providing us with annotated acoustic data sets, which played a key role in the development work.
The first version of Ketos was released in April 2019. 

The intended users of Ketos are primarily researchers and data scientists working with (underwater) acoustics data. 
While Ketos comes with complete documentation and comprehensive step-by-step tutorials, some familiarity with 
Python and especially the NumPy package would be beneficial. A basic understanding of 
the fundamentals of machine learning and neural networks would also be an advantage.

The name Ketos was chosen to highlight the package's main intended application, underwater acoustics.
In Ancient Greek, the word ketos denotes a large fish, whale, shark, or sea monster. The word ketos 
is also the origin of the scientific term for whales, cetacean.


## Installation

Ketos is most easily installed using the Anaconda package manager.
Anaconda is freely available from [docs.anaconda.com/anaconda/install](https://docs.anaconda.com/anaconda/install/). 
Make sure you get the Python 3.7 version and make sure to pick the installer appropriate for your OS (Linux, macOS, Windows) 

 1. Clone the Ketos repository:
    ```bash
    git clone https://gitlab.meridian.cs.dal.ca/public_projects/ketos.git
    cd ketos
    ```

 2. Create and activate a new Anaconda environment:
    ```bash
    conda create --name ketos_env
    conda activate ketos_env
    ```

 3. Install the PyPI package manager and Jupyter Notebook:
    ```bash
    conda install pip
    conda install jupyter
    ```

 4. Install Ketos:
    ```bash
    python setup.py sdist
    pip install dist/ketos-2.0.0.tar.gz
    ```

## Jupyter notebook tutorials

 1. [notebook 1](docs/source/tutorials/bla/bla.ipynb)