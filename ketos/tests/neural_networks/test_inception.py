import pytest
import numpy as np
import tensorflow as tf
from ketos.neural_networks.dev_utils.nn_interface import RecipeCompat
from ketos.neural_networks.inception import ConvBatchNormRelu, InceptionBlock, InceptionArch, InceptionInterface
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
                'initial_filters':16,
                'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005),
                'loss_function': RecipeCompat('CategoricalCrossentropy', tf.keras.losses.CategoricalCrossentropy),  
                'metrics': [RecipeCompat('CategoricalAccuracy',tf.keras.metrics.CategoricalAccuracy),
                            ],
                }

    return recipe

def test_ConvBatchNormRelu():
    layer = ConvBatchNormRelu(n_filters=16, filter_shape=3, strides=1, padding='same')

    
    assert len(layer.layers[0].layers) == 3
    assert isinstance(layer.layers[0].layers[0], tf.keras.layers.Conv2D)
    assert isinstance(layer.layers[0].layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(layer.layers[0].layers[2], tf.keras.layers.ReLU)


def test_InceptionBlock():
    block = InceptionBlock(n_filters=16, strides=1)

    assert len(block.layers) == 6
    assert isinstance(block.layers[0], ConvBatchNormRelu)
    assert isinstance(block.layers[1], ConvBatchNormRelu)
    assert isinstance(block.layers[2], ConvBatchNormRelu)
    assert isinstance(block.layers[3], ConvBatchNormRelu)
    assert isinstance(block.layers[4], tf.keras.layers.MaxPooling2D)
    assert isinstance(block.layers[5], ConvBatchNormRelu)


def test_InceptionArch():
    inception = InceptionArch(n_blocks=3, n_classes=2, initial_filters=16)

    assert len(inception.layers) == 5
    assert isinstance(inception.layers[0], ConvBatchNormRelu)
    assert len(inception.layers[1].layers) == 6
    for block in inception.layers[1].layers:
        assert isinstance(block, InceptionBlock)
    assert isinstance(inception.layers[2], tf.keras.layers.GlobalAveragePooling2D)
    assert isinstance(inception.layers[3], tf.keras.layers.Dense)
    assert isinstance(inception.layers[4], tf.keras.layers.Softmax)



    



