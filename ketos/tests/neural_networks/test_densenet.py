import pytest
import numpy as np
import tensorflow as tf
from ketos.neural_networks.dev_utils.nn_interface import RecipeCompat
from ketos.neural_networks.densenet import ConvBlock, DenseBlock, TransitionBlock, DenseNetArch, DenseNetInterface
from ketos.neural_networks.dev_utils.losses import FScoreLoss
#from ketos.neural_networks.dev_utils.metrics import Precision, Recall, Accuracy, FScore
import os
import tables
import json

current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')


