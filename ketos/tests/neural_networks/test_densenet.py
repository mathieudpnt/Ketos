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


@pytest.fixture
def recipe_dict():
    recipe = {'dense_blocks':[ 6, 12, 24, 16],
                    'growth_rate':32,
                    'compression_factor':0.5,
                    'n_classes':2,
                    'dropout_rate':0.2,
                    'optimizer': {'recipe_name':'Adam', 'parameters': {'learning_rate':0.005}},
                    'loss_function': {'recipe_name':'CategoricalCrossEntropy', 'parameters':{}},  
                    'metrics': [{'recipe_name':'CategoricalAccuracy', 'parameters':{}}]
        
    }
    return recipe
    
@pytest.fixture
def recipe():
    recipe = {'dense_blocks':[ 6, 12, 24, 16],
                    'growth_rate':32,
                    'compression_factor':0.5,
                    'n_classes':2,
                    'dropout_rate':0.2,
                    'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005),
                    'loss_function': RecipeCompat('CategoricalCrossentropy', tf.keras.losses.CategoricalCrossentropy),  
                    'metrics': [RecipeCompat('CategoricalAccuracy',tf.keras.metrics.CategoricalAccuracy),
                                ],
                    }
    return recipe
