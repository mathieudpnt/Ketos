import pytest
import numpy as np
import tensorflow as tf
from ketos.neural_networks.resnet import ResNetBlock, ResNetArch, ResNetInterface
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
def recipe_dict():
    recipe = {'block_list':[2,2,2],
               'n_classes':2,
               'initial_filters':16,        
               'optimizer': {'name':'Adam', 'parameters': {'learning_rate':0.005}},
               'loss_function': {'name':'FScoreLoss', 'parameters':{}},  
               'metrics': [{'name':'CategoricalAccuracy', 'parameters':{}}]
        
    }
    return recipe


def test_ResNetBlock():
    block = ResNetBlock(channels=1, strides=1, residual_path=False)

    assert len(block.layers) == 5
    assert isinstance(block.layers[0], tf.keras.layers.Conv2D)
    assert isinstance(block.layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(block.layers[2], tf.keras.layers.Conv2D)
    assert isinstance(block.layers[3], tf.keras.layers.BatchNormalization)
    assert isinstance(block.layers[4], tf.keras.layers.Dropout)



def test_ResNetBlock_residual():
    block = ResNetBlock(channels=1, strides=1, residual_path=True)

    assert len(block.layers) == 7
    assert isinstance(block.layers[0], tf.keras.layers.Conv2D)
    assert isinstance(block.layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(block.layers[2], tf.keras.layers.Conv2D)
    assert isinstance(block.layers[3], tf.keras.layers.BatchNormalization)
    assert isinstance(block.layers[4], tf.keras.layers.Conv2D)
    assert isinstance(block.layers[5], tf.keras.layers.BatchNormalization)
    assert isinstance(block.layers[6], tf.keras.layers.Dropout)

def test_ResNetArch():
    resnet = ResNetArch(block_list=[2,2,2], n_classes=2, initial_filters=16)

    assert len(resnet.layers) == 6
    assert isinstance(resnet.layers[0], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1], tf.keras.models.Sequential)
    assert isinstance(resnet.layers[2], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[3], tf.keras.layers.GlobalAveragePooling2D)
    assert isinstance(resnet.layers[4], tf.keras.layers.Dense)
    assert isinstance(resnet.layers[5], tf.keras.layers.Softmax)

    #ResNet blocks
    assert len(resnet.layers[1].layers) == 6
    assert isinstance(resnet.layers[1].layers[0], ResNetBlock)
    assert isinstance(resnet.layers[1].layers[1], ResNetBlock)
    assert isinstance(resnet.layers[1].layers[2], ResNetBlock)
    assert isinstance(resnet.layers[1].layers[3], ResNetBlock)
    assert isinstance(resnet.layers[1].layers[4], ResNetBlock)
    assert isinstance(resnet.layers[1].layers[5], ResNetBlock)

    #Block 1
    assert isinstance(resnet.layers[1].layers[0].layers[0], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1].layers[0].layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[0].layers[2], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1].layers[0].layers[3], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[0].layers[4], tf.keras.layers.Dropout)

    #Block 2
    assert isinstance(resnet.layers[1].layers[1].layers[0], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1].layers[1].layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[1].layers[2], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1].layers[1].layers[3], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[1].layers[4], tf.keras.layers.Dropout)

    #Block 3
    assert isinstance(resnet.layers[1].layers[2].layers[0], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1].layers[2].layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[2].layers[2], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1].layers[2].layers[3], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[2].layers[4], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1].layers[2].layers[5], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[2].layers[6], tf.keras.layers.Dropout)

    #Block 4
    assert isinstance(resnet.layers[1].layers[3].layers[0], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1].layers[3].layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[3].layers[2], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1].layers[3].layers[3], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[3].layers[4], tf.keras.layers.Dropout)

    #Block 5
    assert isinstance(resnet.layers[1].layers[4].layers[0], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1].layers[4].layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[4].layers[2], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1].layers[4].layers[3], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[4].layers[4], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1].layers[4].layers[5], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[4].layers[6], tf.keras.layers.Dropout)

    #Block 6
    assert isinstance(resnet.layers[1].layers[5].layers[0], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1].layers[5].layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[5].layers[2], tf.keras.layers.Conv2D)
    assert isinstance(resnet.layers[1].layers[5].layers[3], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[5].layers[4], tf.keras.layers.Dropout)










    






