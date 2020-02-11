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


@pytest.fixture
def recipe_simple_dict():
    recipe = {'conv_set':[(64, False), (128, True), (256, True)],
               'dense_set': [512, 256],
               'n_classes':2,
               'optimizer': {'name':'Adam', 'parameters': {'learning_rate':0.005}},
               'loss_function': {'name':'FScoreLoss', 'parameters':{}},  
               'metrics': [{'name':'CategoricalAccuracy', 'parameters':{}}]

    }

    return recipe


@pytest.fixture
def recipe_simple():
    recipe = {'conv_set':[(64, False), (128, True), (256, True)],
               'dense_set': [512, 256],
               'n_classes':2,        
               'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005),
               'loss_function': RecipeCompat('FScoreLoss', FScoreLoss),  
               'metrics': [RecipeCompat('CategoricalAccuracy',tf.keras.metrics.CategoricalAccuracy)]
        
    }
    return recipe
