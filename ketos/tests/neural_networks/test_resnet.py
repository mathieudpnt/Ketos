import pytest
import numpy as np
import tensorflow as tf
from ketos.neural_networks.dev_utils.nn_interface import RecipeCompat
from ketos.neural_networks.resnet import ResNetBlock, ResNet1DBlock, ResNetArch, ResNet1DArch, ResNetInterface, ResNet1DInterface
from ketos.neural_networks.dev_utils.losses import FScoreLoss
from ketos.data_handling.data_feeding import BatchGenerator
import os
import tables
import json

current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')


# ResNet 2D


@pytest.fixture
def recipe_dict():
    recipe = {'interface': 'ResNetInterface',
               'block_sets':[2,2,2],
               'n_classes':2,
               'initial_filters':16,        
               'initial_strides':1,
               'initial_kernel':[3,3],        
               'strides':2,
               'kernel':[3,3],        
               'optimizer': {'recipe_name':'Adam', 'parameters': {'learning_rate':0.005}},
               'loss_function': {'recipe_name':'FScoreLoss', 'parameters':{}},  
               'metrics': [{'recipe_name':'CategoricalAccuracy', 'parameters':{}}]
        
    }
    return recipe

@pytest.fixture
def recipe():
    recipe = {'interface': 'ResNetInterface',
               'block_sets':[2,2,2],
               'n_classes':2,
               'initial_filters':16,        
               'initial_strides':1,
               'initial_kernel':[3,3],        
               'strides':2,
               'kernel':[3,3],        
               'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005),
               'loss_function': RecipeCompat('FScoreLoss', FScoreLoss),  
               'metrics': [RecipeCompat('CategoricalAccuracy',tf.keras.metrics.CategoricalAccuracy)]   
    }
    return recipe

@pytest.fixture
def recipe_dict_simple():
    recipe = {'interface': 'ResNetInterface',
               'block_sets':[2,2,2],
               'n_classes':2,
               'initial_filters':16,        
               'optimizer': {'recipe_name':'Adam', 'parameters': {'learning_rate':0.005}},
               'loss_function': {'recipe_name':'FScoreLoss', 'parameters':{}},  
               'metrics': [{'recipe_name':'CategoricalAccuracy', 'parameters':{}}]
        
    }
    return recipe

@pytest.fixture
def recipe_simple():
    recipe = {'interface': 'ResNetInterface',
               'block_sets':[2,2,2],
               'n_classes':2,
               'initial_filters':16,        
               'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005),
               'loss_function': RecipeCompat('FScoreLoss', FScoreLoss),  
               'metrics': [RecipeCompat('CategoricalAccuracy',tf.keras.metrics.CategoricalAccuracy)]   
    }
    return recipe

def test_ResNetBlock():
    block = ResNetBlock(filters=1, strides=1, residual_path=False)

    assert len(block.layers) == 5
    assert isinstance(block.layers[0], tf.keras.layers.Conv2D)
    assert isinstance(block.layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(block.layers[2], tf.keras.layers.Conv2D)
    assert isinstance(block.layers[3], tf.keras.layers.BatchNormalization)
    assert isinstance(block.layers[4], tf.keras.layers.Dropout)



def test_ResNetBlock_residual():
    block = ResNetBlock(filters=1, strides=1, residual_path=True)

    assert len(block.layers) == 7
    assert isinstance(block.layers[0], tf.keras.layers.Conv2D)
    assert isinstance(block.layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(block.layers[2], tf.keras.layers.Conv2D)
    assert isinstance(block.layers[3], tf.keras.layers.BatchNormalization)
    assert isinstance(block.layers[4], tf.keras.layers.Conv2D)
    assert isinstance(block.layers[5], tf.keras.layers.BatchNormalization)
    assert isinstance(block.layers[6], tf.keras.layers.Dropout)

def test_ResNetArch():
    resnet = ResNetArch(block_sets=[2,2,2], n_classes=2, initial_filters=16)

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


def test_ResNetInterface_build_from_recipe(recipe):
    resnet = ResNetInterface._build_from_recipe(recipe)

    assert resnet.optimizer.recipe_name == recipe['optimizer'].recipe_name
    assert resnet.optimizer.instance.__class__ == recipe['optimizer'].instance.__class__
    assert resnet.optimizer.args == recipe['optimizer'].args

    assert resnet.loss_function.recipe_name == recipe['loss_function'].recipe_name
    assert resnet.loss_function.instance.__class__ == recipe['loss_function'].instance.__class__
    assert resnet.loss_function.args == recipe['loss_function'].args

    assert resnet.metrics[0].recipe_name == recipe['metrics'][0].recipe_name
    assert resnet.metrics[0].instance.__class__ == recipe['metrics'][0].instance.__class__
    assert resnet.metrics[0].args == recipe['metrics'][0].args

    assert resnet.initial_filters == recipe['initial_filters']
    assert resnet.block_sets == recipe['block_sets']
    assert resnet.n_classes ==  recipe['n_classes']

    assert resnet.initial_strides == recipe['initial_strides']
    assert resnet.initial_kernel == recipe['initial_kernel']
    assert resnet.strides == recipe['strides']
    assert resnet.kernel == recipe['kernel']

def test_ResNetInterface_build_from_recipe_simple(recipe_simple):
    resnet = ResNetInterface._build_from_recipe(recipe_simple)

    assert resnet.optimizer.recipe_name == recipe_simple['optimizer'].recipe_name
    assert resnet.optimizer.instance.__class__ == recipe_simple['optimizer'].instance.__class__
    assert resnet.optimizer.args == recipe_simple['optimizer'].args

    assert resnet.loss_function.recipe_name == recipe_simple['loss_function'].recipe_name
    assert resnet.loss_function.instance.__class__ == recipe_simple['loss_function'].instance.__class__
    assert resnet.loss_function.args == recipe_simple['loss_function'].args

    assert resnet.metrics[0].recipe_name == recipe_simple['metrics'][0].recipe_name
    assert resnet.metrics[0].instance.__class__ == recipe_simple['metrics'][0].instance.__class__
    assert resnet.metrics[0].args == recipe_simple['metrics'][0].args

    assert resnet.initial_filters == recipe_simple['initial_filters']
    assert resnet.block_sets == recipe_simple['block_sets']
    assert resnet.n_classes ==  recipe_simple['n_classes']

def test_extract_recipe_dict(recipe, recipe_dict):
    resnet = ResNetInterface._build_from_recipe(recipe)
    written_recipe = resnet._extract_recipe_dict()

    assert written_recipe == recipe_dict

def test_extract_recipe_dict_simple(recipe_simple, recipe_dict):
    resnet = ResNetInterface._build_from_recipe(recipe_simple)
    written_recipe = resnet._extract_recipe_dict()

    assert written_recipe == recipe_dict

def test_read_recipe_file(recipe, recipe_dict):
    path_to_recipe_file = os.path.join(path_to_tmp, "test_resnet_recipe.json")
    resnet = ResNetInterface._build_from_recipe(recipe)
    written_recipe = resnet._extract_recipe_dict()
    resnet.save_recipe_file(path_to_recipe_file)

    #Read recipe as a recipe dict
    read_recipe = resnet._read_recipe_file(path_to_recipe_file,return_recipe_compat=False)
    assert read_recipe == recipe_dict

    #Read recipe as a recipe dict with RecipeCompat objects
    read_recipe = resnet._read_recipe_file(path_to_recipe_file,return_recipe_compat=True)
    assert read_recipe['optimizer'].recipe_name ==recipe['optimizer'].recipe_name
    assert read_recipe['optimizer'].instance.__class__ == recipe['optimizer'].instance.__class__
    assert read_recipe['optimizer'].args == recipe['optimizer'].args

    assert read_recipe['loss_function'].recipe_name == recipe['loss_function'].recipe_name
    assert read_recipe['loss_function'].instance.__class__ == recipe['loss_function'].instance.__class__
    assert read_recipe['loss_function'].args == recipe['loss_function'].args
    
    assert read_recipe['metrics'][0].recipe_name == recipe['metrics'][0].recipe_name
    assert read_recipe['metrics'][0].instance.__class__ == recipe['metrics'][0].instance.__class__
    assert read_recipe['metrics'][0].args == recipe['metrics'][0].args

    assert read_recipe['block_sets'] == recipe['block_sets']
    assert read_recipe['n_classes'] ==  recipe['n_classes']
    assert read_recipe['initial_filters'] == recipe['initial_filters']

    assert read_recipe['initial_strides'] == recipe['initial_strides']
    assert read_recipe['initial_kernel'] == recipe['initial_kernel']
    assert read_recipe['strides'] == recipe['strides']
    assert read_recipe['kernel'] == recipe['kernel']

def test_read_recipe_file_simple(recipe, recipe_dict):
    path_to_recipe_file = os.path.join(path_to_tmp, "test_resnet_recipe.json")
    resnet = ResNetInterface._build_from_recipe(recipe)
    written_recipe = resnet._extract_recipe_dict()
    resnet.save_recipe_file(path_to_recipe_file)

    #Read recipe as a recipe dict
    read_recipe = resnet._read_recipe_file(path_to_recipe_file,return_recipe_compat=False)
    assert read_recipe == recipe_dict

    #Read recipe as a recipe dict with RecipeCompat objects
    read_recipe = resnet._read_recipe_file(path_to_recipe_file,return_recipe_compat=True)
    assert read_recipe['optimizer'].recipe_name ==recipe['optimizer'].recipe_name
    assert read_recipe['optimizer'].instance.__class__ == recipe['optimizer'].instance.__class__
    assert read_recipe['optimizer'].args == recipe['optimizer'].args

    assert read_recipe['loss_function'].recipe_name == recipe['loss_function'].recipe_name
    assert read_recipe['loss_function'].instance.__class__ == recipe['loss_function'].instance.__class__
    assert read_recipe['loss_function'].args == recipe['loss_function'].args
    
    assert read_recipe['metrics'][0].recipe_name == recipe['metrics'][0].recipe_name
    assert read_recipe['metrics'][0].instance.__class__ == recipe['metrics'][0].instance.__class__
    assert read_recipe['metrics'][0].args == recipe['metrics'][0].args

    assert read_recipe['initial_filters'] == recipe['initial_filters']
    assert read_recipe['block_sets'] == recipe['block_sets']
    assert read_recipe['n_classes'] ==  recipe['n_classes']

def test_train_ResNet(sample_data):
    data, labels = sample_data
    resnet = ResNetInterface() #default resnet
    train_generator = BatchGenerator(batch_size=5, x=data, y=labels, shuffle=True)
    val_generator = BatchGenerator(batch_size=5, x=data, y=labels, shuffle=True)

    resnet.train_generator = train_generator
    resnet.val_generator = val_generator

    resnet.train_loop(2, checkpoint_freq=0)

def test_freeze_and_unfreeze_block():
    """ Check that calling the freeze_block and unfreeze_block methods does not result in an error"""
    resnet = ResNetInterface() #default resnet
    resnet.model.freeze_block([1,2,5])
    resnet.model.unfreeze_block([5])

def test_set_batch_norm_momentum():
    """ Check that calling the set_batch_norm_momentum method does not result in an error"""
    resnet = ResNetInterface() #default resnet
    resnet.model.set_batch_norm_momentum(momentum=0.5)
    resnet1d = ResNet1DInterface() #default resnet
    resnet1d.model.set_batch_norm_momentum(momentum=0.5)

def test_set_dropout_rate():
    """ Check that calling the set_dropout_rate method does not result in an error"""
    resnet = ResNetInterface() #default resnet
    resnet.model.set_dropout_rate(rate=0.5)
    resnet1d = ResNet1DInterface() #default resnet
    resnet1d.model.set_dropout_rate(rate=0.5)


# ResNet 1D


@pytest.fixture
def recipe_dict_1d():
    recipe = {'interface': 'ResNet1DInterface',
               'block_sets':[2,2,2],
               'n_classes':2,
               'initial_filters':2,        
               'initial_strides':1,
               'initial_kernel':30,        
               'strides':2,
               'kernel':300,        
               'optimizer': {'recipe_name':'Adam', 'parameters': {'learning_rate':0.005}},
               'loss_function': {'recipe_name':'FScoreLoss', 'parameters':{}},  
               'metrics': [{'recipe_name':'CategoricalAccuracy', 'parameters':{}}]
        
    }
    return recipe

@pytest.fixture
def recipe_1d():
    recipe = {'interface': 'ResNet1DInterface',
               'block_sets':[2,2,2],
               'n_classes':2,
               'initial_strides':1,
               'initial_kernel':30,        
               'strides':2,
               'kernel':300,        
               'initial_filters':2,        
               'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005),
               'loss_function': RecipeCompat('FScoreLoss', FScoreLoss),  
               'metrics': [RecipeCompat('CategoricalAccuracy',tf.keras.metrics.CategoricalAccuracy)]
        
    }
    return recipe


def test_ResNetBlock_1d():
    block = ResNet1DBlock(filters=1, strides=1, residual_path=False)

    assert len(block.layers) == 5
    assert isinstance(block.layers[0], tf.keras.layers.Conv1D)
    assert isinstance(block.layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(block.layers[2], tf.keras.layers.Conv1D)
    assert isinstance(block.layers[3], tf.keras.layers.BatchNormalization)
    assert isinstance(block.layers[4], tf.keras.layers.Dropout)



def test_ResNetBlock_residual_1d():
    block = ResNet1DBlock(filters=1, strides=1, residual_path=True)

    assert len(block.layers) == 7
    assert isinstance(block.layers[0], tf.keras.layers.Conv1D)
    assert isinstance(block.layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(block.layers[2], tf.keras.layers.Conv1D)
    assert isinstance(block.layers[3], tf.keras.layers.BatchNormalization)
    assert isinstance(block.layers[4], tf.keras.layers.Conv1D)
    assert isinstance(block.layers[5], tf.keras.layers.BatchNormalization)
    assert isinstance(block.layers[6], tf.keras.layers.Dropout)

def test_ResNetArch_1d():
    resnet = ResNet1DArch(block_sets=[2,2,2], n_classes=2, initial_filters=2)

    assert len(resnet.layers) == 6
    assert isinstance(resnet.layers[0], tf.keras.layers.Conv1D)
    assert isinstance(resnet.layers[1], tf.keras.models.Sequential)
    assert isinstance(resnet.layers[2], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[3], tf.keras.layers.GlobalAveragePooling1D)
    assert isinstance(resnet.layers[4], tf.keras.layers.Dense)
    assert isinstance(resnet.layers[5], tf.keras.layers.Softmax)

    #ResNet blocks
    assert len(resnet.layers[1].layers) == 6
    assert isinstance(resnet.layers[1].layers[0], ResNet1DBlock)
    assert isinstance(resnet.layers[1].layers[1], ResNet1DBlock)
    assert isinstance(resnet.layers[1].layers[2], ResNet1DBlock)
    assert isinstance(resnet.layers[1].layers[3], ResNet1DBlock)
    assert isinstance(resnet.layers[1].layers[4], ResNet1DBlock)
    assert isinstance(resnet.layers[1].layers[5], ResNet1DBlock)

    #Block 1
    assert isinstance(resnet.layers[1].layers[0].layers[0], tf.keras.layers.Conv1D)
    assert isinstance(resnet.layers[1].layers[0].layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[0].layers[2], tf.keras.layers.Conv1D)
    assert isinstance(resnet.layers[1].layers[0].layers[3], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[0].layers[4], tf.keras.layers.Dropout)

    #Block 2
    assert isinstance(resnet.layers[1].layers[1].layers[0], tf.keras.layers.Conv1D)
    assert isinstance(resnet.layers[1].layers[1].layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[1].layers[2], tf.keras.layers.Conv1D)
    assert isinstance(resnet.layers[1].layers[1].layers[3], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[1].layers[4], tf.keras.layers.Dropout)

    #Block 3
    assert isinstance(resnet.layers[1].layers[2].layers[0], tf.keras.layers.Conv1D)
    assert isinstance(resnet.layers[1].layers[2].layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[2].layers[2], tf.keras.layers.Conv1D)
    assert isinstance(resnet.layers[1].layers[2].layers[3], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[2].layers[4], tf.keras.layers.Conv1D)
    assert isinstance(resnet.layers[1].layers[2].layers[5], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[2].layers[6], tf.keras.layers.Dropout)

    #Block 4
    assert isinstance(resnet.layers[1].layers[3].layers[0], tf.keras.layers.Conv1D)
    assert isinstance(resnet.layers[1].layers[3].layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[3].layers[2], tf.keras.layers.Conv1D)
    assert isinstance(resnet.layers[1].layers[3].layers[3], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[3].layers[4], tf.keras.layers.Dropout)

    #Block 5
    assert isinstance(resnet.layers[1].layers[4].layers[0], tf.keras.layers.Conv1D)
    assert isinstance(resnet.layers[1].layers[4].layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[4].layers[2], tf.keras.layers.Conv1D)
    assert isinstance(resnet.layers[1].layers[4].layers[3], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[4].layers[4], tf.keras.layers.Conv1D)
    assert isinstance(resnet.layers[1].layers[4].layers[5], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[4].layers[6], tf.keras.layers.Dropout)

    #Block 6
    assert isinstance(resnet.layers[1].layers[5].layers[0], tf.keras.layers.Conv1D)
    assert isinstance(resnet.layers[1].layers[5].layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[5].layers[2], tf.keras.layers.Conv1D)
    assert isinstance(resnet.layers[1].layers[5].layers[3], tf.keras.layers.BatchNormalization)
    assert isinstance(resnet.layers[1].layers[5].layers[4], tf.keras.layers.Dropout)


def test_ResNetInterface_build_from_recipe_1d(recipe_1d):
    resnet = ResNet1DInterface._build_from_recipe(recipe_1d)

    assert resnet.optimizer.recipe_name == recipe_1d['optimizer'].recipe_name
    assert resnet.optimizer.instance.__class__ == recipe_1d['optimizer'].instance.__class__
    assert resnet.optimizer.args == recipe_1d['optimizer'].args

    assert resnet.loss_function.recipe_name == recipe_1d['loss_function'].recipe_name
    assert resnet.loss_function.instance.__class__ == recipe_1d['loss_function'].instance.__class__
    assert resnet.loss_function.args == recipe_1d['loss_function'].args

    assert resnet.metrics[0].recipe_name == recipe_1d['metrics'][0].recipe_name
    assert resnet.metrics[0].instance.__class__ == recipe_1d['metrics'][0].instance.__class__
    assert resnet.metrics[0].args == recipe_1d['metrics'][0].args

    assert resnet.initial_filters == recipe_1d['initial_filters']
    assert resnet.block_sets == recipe_1d['block_sets']
    assert resnet.n_classes ==  recipe_1d['n_classes']


def test_extract_recipe_dict_1d(recipe_1d, recipe_dict_1d):
    resnet = ResNet1DInterface._build_from_recipe(recipe_1d)
    written_recipe = resnet._extract_recipe_dict()

    assert written_recipe == recipe_dict_1d


def test_read_recipe_file_1d(recipe_1d, recipe_dict_1d):
    path_to_recipe_file = os.path.join(path_to_tmp, "test_resnet_recipe_1d.json")
    resnet = ResNet1DInterface._build_from_recipe(recipe_1d)
    written_recipe = resnet._extract_recipe_dict()
    resnet.save_recipe_file(path_to_recipe_file)

    #Read recipe as a recipe dict
    read_recipe = resnet._read_recipe_file(path_to_recipe_file,return_recipe_compat=False)
    assert read_recipe == recipe_dict_1d

    #Read recipe as a recipe dict with RecipeCompat objects
    read_recipe = resnet._read_recipe_file(path_to_recipe_file,return_recipe_compat=True)
    assert read_recipe['optimizer'].recipe_name ==recipe_1d['optimizer'].recipe_name
    assert read_recipe['optimizer'].instance.__class__ == recipe_1d['optimizer'].instance.__class__
    assert read_recipe['optimizer'].args == recipe_1d['optimizer'].args

    assert read_recipe['loss_function'].recipe_name == recipe_1d['loss_function'].recipe_name
    assert read_recipe['loss_function'].instance.__class__ == recipe_1d['loss_function'].instance.__class__
    assert read_recipe['loss_function'].args == recipe_1d['loss_function'].args
    
    assert read_recipe['metrics'][0].recipe_name == recipe_1d['metrics'][0].recipe_name
    assert read_recipe['metrics'][0].instance.__class__ == recipe_1d['metrics'][0].instance.__class__
    assert read_recipe['metrics'][0].args == recipe_1d['metrics'][0].args

    assert read_recipe['initial_filters'] == recipe_1d['initial_filters']
    assert read_recipe['block_sets'] == recipe_1d['block_sets']
    assert read_recipe['n_classes'] ==  recipe_1d['n_classes']


def test_train_ResNet1D(sample_data_1d):
    data, labels = sample_data_1d
    resnet = ResNet1DInterface() #default resnet 1d
    train_generator = BatchGenerator(batch_size=5, x=data, y=labels, shuffle=True)
    val_generator = BatchGenerator(batch_size=5, x=data, y=labels, shuffle=True)

    resnet.train_generator = train_generator
    resnet.val_generator = val_generator

    resnet.train_loop(2, checkpoint_freq=0)
