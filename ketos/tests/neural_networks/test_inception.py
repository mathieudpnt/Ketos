import pytest
import numpy as np
import tensorflow as tf
from ketos.neural_networks.dev_utils.nn_interface import RecipeCompat
from ketos.neural_networks.densenet import ConvBlock, DenseBlock, TransitionBlock, DenseNetArch, DenseNetInterface
#from ketos.neural_networks.dev_utils.losses import FScoreLoss
from ketos.data_handling.data_feeding import BatchGenerator
#from ketos.neural_networks.dev_utils.metrics import Precision, Recall, Accuracy, FScore
import os
import tables
import json

current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')

@pytest.fixture
def recipe_dict():
    recipe = {'n_blocks':3,
                'n_classes':2,
                'initial_filters':16,
                'optimizer': {'recipe_name':'Adam', 'parameters': {'learning_rate':0.005}},
                'loss_function': {'recipe_name':'CategoricalCrossentropy', 'parameters':{}},  
                'metrics': [{'recipe_name':'CategoricalAccuracy', 'parameters':{}}]
    
    }

    return recipe

@pytest.fixture
def recipe():
    recipe = {'n_blocks':3,
                'n_classes':2,
                'initial_filters':16,,
                'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005),
                'loss_function': RecipeCompat('CategoricalCrossentropy', tf.keras.losses.CategoricalCrossentropy),  
                'metrics': [RecipeCompat('CategoricalAccuracy',tf.keras.metrics.CategoricalAccuracy),
                            ],
                }
                
    return recipe



