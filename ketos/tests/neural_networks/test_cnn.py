import pytest
import numpy as np
import tensorflow as tf
from ketos.neural_networks.nn_interface import RecipeCompat
from ketos.neural_networks.resnet import CNNArch, CNNInterface
from ketos.neural_networks.losses import FScoreLoss
from ketos.neural_networks.metrics import Precision, Recall, Accuracy, FScore
from ketos.data_handling.data_feeding import BatchGenerator
import os
import tables
import json



current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')
