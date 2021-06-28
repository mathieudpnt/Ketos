import numpy as np
np.random.seed(1000)

import tensorflow as tf
tf.random.set_seed(2000)

import pytest
from ketos.neural_networks.dev_utils.nn_interface import RecipeCompat
from ketos.neural_networks.cnn import CNNArch, CNN1DArch, CNNInterface, CNN1DInterface
from ketos.neural_networks.dev_utils.losses import FScoreLoss
from ketos.data_handling.data_feeding import BatchGenerator
import os
import tables
import json


current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')



#Tests for CNN 2D

@pytest.fixture
def recipe_simple_dict():
    recipe = {'interface': 'CNNInterface',
               'conv_set':[[64, False], [128, True], [256, True]],
               'dense_set': [512, 256],
               'n_classes':2,
               'optimizer': {'recipe_name':'Adam', 'parameters': {'learning_rate':0.005}},
               'loss_function': {'recipe_name':'FScoreLoss', 'parameters':{}},  
               'metrics': [{'recipe_name':'CategoricalAccuracy', 'parameters':{}}]

    }

    return recipe


@pytest.fixture
def recipe_simple():
    recipe = {'interface': 'CNNInterface',
               'conv_set':[[64, False], [128, True], [256, True]],
               'dense_set': [512, 256],
               'n_classes':2,        
               'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005),
               'loss_function': RecipeCompat('FScoreLoss', FScoreLoss),  
               'metrics': [RecipeCompat('CategoricalAccuracy',tf.keras.metrics.CategoricalAccuracy)]
        
    }
    return recipe


@pytest.fixture
def recipe_detailed_dict():
    recipe = {'interface': 'CNNInterface',
              'convolutional_layers':  [{'n_filters':64, "filter_shape":[3,3], 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':None, 'batch_normalization':True},
                                    {'n_filters':128, "filter_shape":[3,3], 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':{'pool_size':[2,2] , 'strides':[2,2]}, 'batch_normalization':True},
                                    {'n_filters':256, "filter_shape":[3,3], 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':{'pool_size':[2,2] , 'strides':[2,2]}, 'batch_normalization':True}],
              'dense_layers':[{'n_hidden':512, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                                    {'n_hidden':256, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                                    ],
               'n_classes':2,
               'optimizer': {'recipe_name':'Adam', 'parameters': {'learning_rate':0.005}},
               'loss_function': {'recipe_name':'FScoreLoss', 'parameters':{}},  
               'metrics': [{'recipe_name':'CategoricalAccuracy', 'parameters':{}}]

    }

    return recipe

@pytest.fixture
def recipe_detailed():
    recipe = {'interface': 'CNNInterface',
              'convolutional_layers':  [{'n_filters':64, "filter_shape":[3,3], 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':None, 'batch_normalization':True},
                                    {'n_filters':128, "filter_shape":[3,3], 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':{'pool_size':[2,2] , 'strides':[2,2]}, 'batch_normalization':True},
                                    {'n_filters':256, "filter_shape":[3,3], 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':{'pool_size':[2,2] , 'strides':[2,2]}, 'batch_normalization':True}],
              'dense_layers':[{'n_hidden':512, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                                    {'n_hidden':256, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                                    ],
               'n_classes':2,
               'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005),
               'loss_function': RecipeCompat('FScoreLoss', FScoreLoss),  
               'metrics': [RecipeCompat('CategoricalAccuracy',tf.keras.metrics.CategoricalAccuracy)]
        
    }

    return recipe


def test_CNNArch():
    conv_layers =  [{'n_filters':64, "filter_shape":[3,3], 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':None, 'batch_normalization':True},
                                    {'n_filters':128, "filter_shape":[3,3], 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':{'pool_size':[2,2] , 'strides':[2,2]}, 'batch_normalization':True},
                                    {'n_filters':256, "filter_shape":[3,3], 'strides':1, 'padding':'valid', 'activation':'relu', 'max_pool':{'pool_size':[2,2] , 'strides':[2,2]}, 'batch_normalization':True}]

    dense_layers = [{'n_hidden':512, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                                    {'n_hidden':256, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                                    ]
    
    cnn = CNNArch(convolutional_layers=conv_layers, dense_layers=dense_layers, n_classes=2)
    assert len(cnn.layers) == 3

    #convolutional block
    assert len(cnn.layers[0].layers) == 8
    assert isinstance(cnn.layers[0].layers[0], tf.keras.layers.Conv2D)
    assert isinstance(cnn.layers[0].layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(cnn.layers[0].layers[2], tf.keras.layers.Conv2D)
    assert isinstance(cnn.layers[0].layers[3], tf.keras.layers.MaxPooling2D)
    assert isinstance(cnn.layers[0].layers[4], tf.keras.layers.BatchNormalization)
    assert isinstance(cnn.layers[0].layers[5], tf.keras.layers.Conv2D)
    assert isinstance(cnn.layers[0].layers[6], tf.keras.layers.MaxPooling2D)
    assert isinstance(cnn.layers[0].layers[7], tf.keras.layers.BatchNormalization)

    #Flatten layer
    assert isinstance(cnn.layers[1], tf.keras.layers.Flatten)
    #Dense block
    assert len(cnn.layers[2].layers) == 8
    assert isinstance(cnn.layers[2].layers[0], tf.keras.layers.Dense)
    assert isinstance(cnn.layers[2].layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(cnn.layers[2].layers[2], tf.keras.layers.Dropout)
    assert isinstance(cnn.layers[2].layers[3], tf.keras.layers.Dense)
    assert isinstance(cnn.layers[2].layers[4], tf.keras.layers.BatchNormalization)
    assert isinstance(cnn.layers[2].layers[5], tf.keras.layers.Dropout)
    assert isinstance(cnn.layers[2].layers[6], tf.keras.layers.Dense)
    assert isinstance(cnn.layers[2].layers[7], tf.keras.layers.Softmax)
   

def test_convolutional_layers_from_conv_set(recipe_simple, recipe_detailed):
    detailed_layers = CNNInterface._convolutional_layers_from_conv_set(recipe_simple['conv_set'])
    assert detailed_layers == recipe_detailed['convolutional_layers']
    

def test_dense_layers_from_dense_set(recipe_simple, recipe_detailed):
    detailed_layers = CNNInterface._dense_layers_from_dense_set(recipe_simple['dense_set'])
    assert detailed_layers == recipe_detailed['dense_layers']


 
def test_CNNInterface_build_from_recipe_simple(recipe_simple, recipe_detailed):
    cnn = CNNInterface._build_from_recipe(recipe_simple)

    assert cnn.optimizer.recipe_name == recipe_simple['optimizer'].recipe_name
    assert cnn.optimizer.instance.__class__ == recipe_simple['optimizer'].instance.__class__
    assert cnn.optimizer.args == recipe_simple['optimizer'].args

    assert cnn.loss_function.recipe_name == recipe_simple['loss_function'].recipe_name
    assert cnn.loss_function.instance.__class__ == recipe_simple['loss_function'].instance.__class__
    assert cnn.loss_function.args == recipe_simple['loss_function'].args

    assert cnn.metrics[0].recipe_name == recipe_simple['metrics'][0].recipe_name
    assert cnn.metrics[0].instance.__class__ == recipe_simple['metrics'][0].instance.__class__
    assert cnn.metrics[0].args == recipe_simple['metrics'][0].args

    assert cnn.conv_set == recipe_simple['conv_set']
    assert cnn.dense_set == recipe_simple['dense_set']
    assert cnn.n_classes ==  recipe_simple['n_classes']


def test_CNNInterface_build_from_recipe_simple_dict(recipe_simple_dict, recipe_simple, recipe_detailed):
    cnn = CNNInterface._build_from_recipe(recipe_simple_dict, recipe_compat=False)

    assert cnn.optimizer.recipe_name == recipe_simple['optimizer'].recipe_name
    assert cnn.optimizer.instance.__class__ == recipe_simple['optimizer'].instance.__class__
    assert cnn.optimizer.args == recipe_simple['optimizer'].args

    assert cnn.loss_function.recipe_name == recipe_simple['loss_function'].recipe_name
    assert cnn.loss_function.instance.__class__ == recipe_simple['loss_function'].instance.__class__
    assert cnn.loss_function.args == recipe_simple['loss_function'].args

    assert cnn.metrics[0].recipe_name == recipe_simple['metrics'][0].recipe_name
    assert cnn.metrics[0].instance.__class__ == recipe_simple['metrics'][0].instance.__class__
    assert cnn.metrics[0].args == recipe_simple['metrics'][0].args

    assert cnn.conv_set == recipe_simple['conv_set']
    assert cnn.dense_set == recipe_simple['dense_set']
    assert cnn.n_classes ==  recipe_simple['n_classes']


def test_CNNInterface_build_from_recipe_detailed(recipe_detailed):
    cnn = CNNInterface._build_from_recipe(recipe_detailed)

    assert cnn.optimizer.recipe_name == recipe_detailed['optimizer'].recipe_name
    assert cnn.optimizer.instance.__class__ == recipe_detailed['optimizer'].instance.__class__
    assert cnn.optimizer.args == recipe_detailed['optimizer'].args

    assert cnn.loss_function.recipe_name == recipe_detailed['loss_function'].recipe_name
    assert cnn.loss_function.instance.__class__ == recipe_detailed['loss_function'].instance.__class__
    assert cnn.loss_function.args == recipe_detailed['loss_function'].args

    assert cnn.metrics[0].recipe_name == recipe_detailed['metrics'][0].recipe_name
    assert cnn.metrics[0].instance.__class__ == recipe_detailed['metrics'][0].instance.__class__
    assert cnn.metrics[0].args == recipe_detailed['metrics'][0].args

    assert cnn.convolutional_layers == recipe_detailed['convolutional_layers']
    assert cnn.dense_layers == recipe_detailed['dense_layers']
    assert cnn.n_classes ==  recipe_detailed['n_classes']

def test_CNNInterface_build_from_recipe_detailed_dict(recipe_detailed, recipe_detailed_dict):
    cnn = CNNInterface._build_from_recipe(recipe_detailed_dict, recipe_compat=False)

    assert cnn.optimizer.recipe_name == recipe_detailed['optimizer'].recipe_name
    assert cnn.optimizer.instance.__class__ == recipe_detailed['optimizer'].instance.__class__
    assert cnn.optimizer.args == recipe_detailed['optimizer'].args

    assert cnn.loss_function.recipe_name == recipe_detailed['loss_function'].recipe_name
    assert cnn.loss_function.instance.__class__ == recipe_detailed['loss_function'].instance.__class__
    assert cnn.loss_function.args == recipe_detailed['loss_function'].args

    assert cnn.metrics[0].recipe_name == recipe_detailed['metrics'][0].recipe_name
    assert cnn.metrics[0].instance.__class__ == recipe_detailed['metrics'][0].instance.__class__
    assert cnn.metrics[0].args == recipe_detailed['metrics'][0].args

    assert cnn.convolutional_layers == recipe_detailed['convolutional_layers']
    assert cnn.dense_layers == recipe_detailed['dense_layers']
    assert cnn.n_classes ==  recipe_detailed['n_classes']

def test_write_recipe_simple(recipe_simple, recipe_simple_dict, recipe_detailed):
    cnn = CNNInterface._build_from_recipe(recipe_simple)
    written_recipe = cnn._extract_recipe_dict()

    #Even when the model is built from a simplified recipe, the detailed form will be included when writing the recipe again

    recipe_simple_dict['convolutional_layers'] = recipe_detailed['convolutional_layers']
    recipe_simple_dict['dense_layers'] = recipe_detailed['dense_layers']

    assert written_recipe == recipe_simple_dict
    

def test_read_recipe_simple_file(recipe_simple, recipe_simple_dict, recipe_detailed):
    path_to_recipe_file = os.path.join(path_to_tmp, "test_cnn_recipe_simple.json")
    cnn = CNNInterface._build_from_recipe(recipe_simple)
    #written_recipe = resnet.write_recipe()
    cnn.save_recipe_file(path_to_recipe_file)

    #Read recipe as a recipe dict
    read_recipe = cnn._read_recipe_file(path_to_recipe_file,return_recipe_compat=False)
    recipe_simple_dict['convolutional_layers'] = recipe_detailed['convolutional_layers']
    recipe_simple_dict['dense_layers'] = recipe_detailed['dense_layers']
    assert read_recipe == recipe_simple_dict
    

    #Read recipe as a recipe dict with RecipeCompat objects
    recipe_simple['convolutional_layers'] = recipe_detailed['convolutional_layers']
    recipe_simple['dense_layers'] = recipe_detailed['dense_layers']

    read_recipe = cnn._read_recipe_file(path_to_recipe_file,return_recipe_compat=True)
    assert read_recipe['optimizer'].recipe_name == recipe_simple['optimizer'].recipe_name
    assert read_recipe['optimizer'].instance.__class__ == recipe_simple['optimizer'].instance.__class__
    assert read_recipe['optimizer'].args == recipe_simple['optimizer'].args

    assert read_recipe['loss_function'].recipe_name == recipe_simple['loss_function'].recipe_name
    assert read_recipe['loss_function'].instance.__class__ == recipe_simple['loss_function'].instance.__class__
    assert read_recipe['loss_function'].args == recipe_simple['loss_function'].args
    
    assert read_recipe['metrics'][0].recipe_name == recipe_simple['metrics'][0].recipe_name
    assert read_recipe['metrics'][0].instance.__class__ == recipe_simple['metrics'][0].instance.__class__
    assert read_recipe['metrics'][0].args == recipe_simple['metrics'][0].args

    assert read_recipe['conv_set'] == recipe_simple['conv_set']
    assert read_recipe['dense_set'] == recipe_simple['dense_set']
    assert read_recipe['convolutional_layers'] == recipe_simple['convolutional_layers']
    assert read_recipe['dense_layers'] == recipe_simple['dense_layers']


def test_read_recipe_simple_detailed(recipe_detailed, recipe_detailed_dict):
    path_to_recipe_file = os.path.join(path_to_tmp, "test_cnn_recipe_detailed.json")
    cnn = CNNInterface._build_from_recipe(recipe_detailed)
    #written_recipe = resnet.write_recipe()
    cnn.save_recipe_file(path_to_recipe_file)

    #Read recipe as a recipe dict
    read_recipe = cnn._read_recipe_file(path_to_recipe_file,return_recipe_compat=False)
    recipe_detailed_dict['conv_set'] = None
    recipe_detailed_dict['dense_set'] = None
    assert read_recipe == recipe_detailed_dict
    
    #Read recipe as a recipe dict with RecipeCompat objects
    recipe_detailed['conv_set'] = None
    recipe_detailed['dense_set'] = None

    read_recipe = cnn._read_recipe_file(path_to_recipe_file,return_recipe_compat=True)
    assert read_recipe['optimizer'].recipe_name == recipe_detailed['optimizer'].recipe_name
    assert read_recipe['optimizer'].instance.__class__ == recipe_detailed['optimizer'].instance.__class__
    assert read_recipe['optimizer'].args == recipe_detailed['optimizer'].args

    assert read_recipe['loss_function'].recipe_name == recipe_detailed['loss_function'].recipe_name
    assert read_recipe['loss_function'].instance.__class__ == recipe_detailed['loss_function'].instance.__class__
    assert read_recipe['loss_function'].args == recipe_detailed['loss_function'].args
    
    assert read_recipe['metrics'][0].recipe_name == recipe_detailed['metrics'][0].recipe_name
    assert read_recipe['metrics'][0].instance.__class__ == recipe_detailed['metrics'][0].instance.__class__
    assert read_recipe['metrics'][0].args == recipe_detailed['metrics'][0].args

    assert read_recipe['conv_set'] == recipe_detailed['conv_set']
    assert read_recipe['dense_set'] == recipe_detailed['dense_set']
    assert read_recipe['convolutional_layers'] == recipe_detailed['convolutional_layers']
    assert read_recipe['dense_layers'] == recipe_detailed['dense_layers']


def test_train_CNN(sample_data):
    data, labels = sample_data
    cnn = CNNInterface() #default cnn
    train_generator = BatchGenerator(batch_size=5, x=data, y=labels, shuffle=True)
    val_generator = BatchGenerator(batch_size=5, x=data, y=labels, shuffle=True)

    cnn.train_generator = train_generator
    cnn.val_generator = val_generator

    cnn.train_loop(2, checkpoint_freq=0)
    





#Tests for CNN 1D


@pytest.fixture
def recipe_simple_dict_1d():
    recipe = {'interface': 'CNN1DInterface',
               'conv_set':[[8, False], [16, True], [32, True], [64, False], [128, False], [256, True]],
               'dense_set': [512, 128],
               'n_classes':2,        
               'optimizer': {'recipe_name':'Adam', 'parameters': {'beta_1': 0.9, 'beta_2': 0.999, 'decay': 0.01, 'lr': 0.01}},
               'loss_function': {'recipe_name':'CategoricalCrossentropy', 'parameters':{'from_logits':True}},  
               'metrics': [{'recipe_name':'CategoricalAccuracy', 'parameters':{}}]

    }

    return recipe


@pytest.fixture
def recipe_simple_1d():
    recipe = {'interface': 'CNN1DInterface',
               'conv_set':[[8, False], [16, True], [32, True], [64, False], [128, False], [256, True]],
               'dense_set': [512, 128],
               'n_classes':2,        
               'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01),
               'loss_function': RecipeCompat('CategoricalCrossentropy', tf.keras.losses.CategoricalCrossentropy, from_logits=True),  
               'metrics': [RecipeCompat('CategoricalAccuracy',tf.keras.metrics.CategoricalAccuracy)],   
        
    }
    return recipe


@pytest.fixture
def recipe_detailed_dict_1d():
    recipe = {'interface': 'CNN1DInterface',
              'convolutional_layers':  [{'n_filters':8, "filter_shape":64, 'strides':2, 'padding':'causal', 'activation':'relu', 'max_pool': None, 'batch_normalization':True},
                                    {'n_filters':16, "filter_shape":64, 'strides':2, 'padding':'causal', 'activation':'relu', 'max_pool': {'pool_size': 8 , 'strides':8}, 'batch_normalization':True},
                                    {'n_filters':32, "filter_shape":64, 'strides':2, 'padding':'causal', 'activation':'relu', 'max_pool': {'pool_size': 8 , 'strides':8}, 'batch_normalization':True},
                                    {'n_filters':64, "filter_shape":64, 'strides':2, 'padding':'causal','activation':'relu', 'max_pool':None, 'batch_normalization':True, },
                                    {'n_filters':128, "filter_shape":64, 'strides':2, 'padding':'causal','activation':'relu', 'max_pool':None, 'batch_normalization':True},
                                    {'n_filters':256, "filter_shape":64, 'strides':2, 'padding':'causal', 'activation':'relu', 'max_pool':{'pool_size': 8 , 'strides': 8}, 'batch_normalization':True, },
                                    ],

                'dense_layers':[{'n_hidden':512, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                                    {'n_hidden':128, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                                    ],

                'n_classes': 2 ,
               'optimizer': {'recipe_name':'Adam', 'parameters': {'beta_1': 0.9, 'beta_2': 0.999, 'decay': 0.01, 'lr': 0.01}},
               'loss_function': {'recipe_name':'CategoricalCrossentropy', 'parameters':{'from_logits':True}},  
               'metrics': [{'recipe_name':'CategoricalAccuracy', 'parameters':{}}]

    }

    return recipe

@pytest.fixture
def recipe_detailed_1d():
    recipe = {'interface': 'CNN1DInterface',
               'convolutional_layers':  [{'n_filters':8, "filter_shape":64, 'strides':2, 'padding':'causal', 'activation':'relu', 'max_pool': None, 'batch_normalization':True},
                                    {'n_filters':16, "filter_shape":64, 'strides':2, 'padding':'causal', 'activation':'relu', 'max_pool': {'pool_size': 8 , 'strides':8}, 'batch_normalization':True},
                                    {'n_filters':32, "filter_shape":64, 'strides':2, 'padding':'causal', 'activation':'relu', 'max_pool': {'pool_size': 8 , 'strides':8}, 'batch_normalization':True},
                                    {'n_filters':64, "filter_shape":64, 'strides':2, 'padding':'causal','activation':'relu', 'max_pool':None, 'batch_normalization':True, },
                                    {'n_filters':128, "filter_shape":64, 'strides':2, 'padding':'causal','activation':'relu', 'max_pool':None, 'batch_normalization':True},
                                    {'n_filters':256, "filter_shape":64, 'strides':2, 'padding':'causal', 'activation':'relu', 'max_pool':{'pool_size': 8 , 'strides': 8}, 'batch_normalization':True, },
                                    ],

                  'dense_layers':[{'n_hidden':512, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                                    {'n_hidden':128, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                                    ],

                  'n_classes': 2 ,
                  'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01),
                  'loss_function': RecipeCompat('CategoricalCrossentropy', tf.keras.losses.CategoricalCrossentropy, from_logits=True),  
                  'metrics': [RecipeCompat('CategoricalAccuracy',tf.keras.metrics.CategoricalAccuracy)],   
                               
                    }

        

    return recipe


def test_CNN1DArch():
    conv_layers =   [{'n_filters':8, "filter_shape":128, 'strides':2, 'padding':'causal', 'activation':'relu', 'max_pool': None, 'batch_normalization':True},
                                    {'n_filters':16, "filter_shape":64, 'strides':2, 'padding':'causal', 'activation':'relu', 'max_pool': {'pool_size': 8 , 'strides':8}, 'batch_normalization':True},
                                    {'n_filters':32, "filter_shape":32, 'strides':2, 'padding':'causal', 'activation':'relu', 'max_pool': {'pool_size': 8 , 'strides':8}, 'batch_normalization':True},
                                    {'n_filters':64, "filter_shape":16, 'strides':2, 'padding':'causal','activation':'relu', 'max_pool':None, 'batch_normalization':True, },
                                    {'n_filters':128, "filter_shape":8, 'strides':2, 'padding':'causal','activation':'relu', 'max_pool':None, 'batch_normalization':True},
                                    {'n_filters':256, "filter_shape":4, 'strides':2, 'padding':'causal', 'activation':'relu', 'max_pool':{'pool_size': 8 , 'strides': 8}, 'batch_normalization':True, },
                                    ]

    dense_layers = [{'n_hidden':512, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                                    {'n_hidden':128, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},
                                    ]
    
    cnn = CNN1DArch(convolutional_layers=conv_layers, dense_layers=dense_layers, n_classes=2)
    assert len(cnn.layers) == 3

    #convolutional block
    assert len(cnn.layers[0].layers) == 15
    assert isinstance(cnn.layers[0].layers[0], tf.keras.layers.Conv1D)
    assert isinstance(cnn.layers[0].layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(cnn.layers[0].layers[2], tf.keras.layers.Conv1D)
    assert isinstance(cnn.layers[0].layers[3], tf.keras.layers.MaxPooling1D)
    assert isinstance(cnn.layers[0].layers[4], tf.keras.layers.BatchNormalization)
    assert isinstance(cnn.layers[0].layers[5], tf.keras.layers.Conv1D)
    assert isinstance(cnn.layers[0].layers[6], tf.keras.layers.MaxPooling1D)
    assert isinstance(cnn.layers[0].layers[7], tf.keras.layers.BatchNormalization)
    assert isinstance(cnn.layers[0].layers[8], tf.keras.layers.Conv1D)
    assert isinstance(cnn.layers[0].layers[9], tf.keras.layers.BatchNormalization)
    assert isinstance(cnn.layers[0].layers[10], tf.keras.layers.Conv1D)
    assert isinstance(cnn.layers[0].layers[11], tf.keras.layers.BatchNormalization)
    assert isinstance(cnn.layers[0].layers[12], tf.keras.layers.Conv1D)
    assert isinstance(cnn.layers[0].layers[13], tf.keras.layers.MaxPooling1D)
    assert isinstance(cnn.layers[0].layers[14], tf.keras.layers.BatchNormalization)

    #Flatten layer
    assert isinstance(cnn.layers[1], tf.keras.layers.Flatten)
    #Dense block
    assert len(cnn.layers[2].layers) == 8
    assert isinstance(cnn.layers[2].layers[0], tf.keras.layers.Dense)
    assert isinstance(cnn.layers[2].layers[1], tf.keras.layers.BatchNormalization)
    assert isinstance(cnn.layers[2].layers[2], tf.keras.layers.Dropout)
    assert isinstance(cnn.layers[2].layers[3], tf.keras.layers.Dense)
    assert isinstance(cnn.layers[2].layers[4], tf.keras.layers.BatchNormalization)
    assert isinstance(cnn.layers[2].layers[5], tf.keras.layers.Dropout)
    assert isinstance(cnn.layers[2].layers[6], tf.keras.layers.Dense)
    assert isinstance(cnn.layers[2].layers[7], tf.keras.layers.Softmax)
   

def test_convolutional_layers_from_conv_set_1d(recipe_simple_1d, recipe_detailed_1d):
    detailed_layers = CNN1DInterface._convolutional_layers_from_conv_set(recipe_simple_1d['conv_set'])
    assert detailed_layers == recipe_detailed_1d['convolutional_layers']
    

def test_dense_layers_from_dense_set_1d(recipe_simple_1d, recipe_detailed_1d):
    detailed_layers = CNNInterface._dense_layers_from_dense_set(recipe_simple_1d['dense_set'])
    assert detailed_layers == recipe_detailed_1d['dense_layers']


 
def test_CNN1DInterface_build_from_recipe_simple(recipe_simple_1d, recipe_detailed_1d):
    cnn = CNN1DInterface._build_from_recipe(recipe_simple_1d)

    assert cnn.optimizer.recipe_name == recipe_simple_1d['optimizer'].recipe_name
    assert cnn.optimizer.instance.__class__ == recipe_simple_1d['optimizer'].instance.__class__
    assert cnn.optimizer.args == recipe_simple_1d['optimizer'].args

    assert cnn.loss_function.recipe_name == recipe_simple_1d['loss_function'].recipe_name
    assert cnn.loss_function.instance.__class__ == recipe_simple_1d['loss_function'].instance.__class__
    assert cnn.loss_function.args == recipe_simple_1d['loss_function'].args

    assert cnn.metrics[0].recipe_name == recipe_simple_1d['metrics'][0].recipe_name
    assert cnn.metrics[0].instance.__class__ == recipe_simple_1d['metrics'][0].instance.__class__
    assert cnn.metrics[0].args == recipe_simple_1d['metrics'][0].args

    assert cnn.conv_set == recipe_simple_1d['conv_set']
    assert cnn.dense_set == recipe_simple_1d['dense_set']
    assert cnn.n_classes ==  recipe_simple_1d['n_classes']


def test_CNN1DInterface_build_from_recipe_simple_dict(recipe_simple_dict_1d, recipe_simple_1d, recipe_detailed_1d):
    cnn = CNN1DInterface._build_from_recipe(recipe_simple_dict_1d, recipe_compat=False)

    assert cnn.optimizer.recipe_name == recipe_simple_1d['optimizer'].recipe_name
    assert cnn.optimizer.instance.__class__ == recipe_simple_1d['optimizer'].instance.__class__
    assert cnn.optimizer.args == recipe_simple_1d['optimizer'].args

    assert cnn.loss_function.recipe_name == recipe_simple_1d['loss_function'].recipe_name
    assert cnn.loss_function.instance.__class__ == recipe_simple_1d['loss_function'].instance.__class__
    assert cnn.loss_function.args == recipe_simple_1d['loss_function'].args

    assert cnn.metrics[0].recipe_name == recipe_simple_1d['metrics'][0].recipe_name
    assert cnn.metrics[0].instance.__class__ == recipe_simple_1d['metrics'][0].instance.__class__
    assert cnn.metrics[0].args == recipe_simple_1d['metrics'][0].args

    assert cnn.conv_set == recipe_simple_1d['conv_set']
    assert cnn.dense_set == recipe_simple_1d['dense_set']
    assert cnn.n_classes ==  recipe_simple_1d['n_classes']


def test_CNN1DInterface_build_from_recipe_detailed(recipe_detailed_1d):
    cnn = CNN1DInterface._build_from_recipe(recipe_detailed_1d)

    assert cnn.optimizer.recipe_name == recipe_detailed_1d['optimizer'].recipe_name
    assert cnn.optimizer.instance.__class__ == recipe_detailed_1d['optimizer'].instance.__class__
    assert cnn.optimizer.args == recipe_detailed_1d['optimizer'].args

    assert cnn.loss_function.recipe_name == recipe_detailed_1d['loss_function'].recipe_name
    assert cnn.loss_function.instance.__class__ == recipe_detailed_1d['loss_function'].instance.__class__
    assert cnn.loss_function.args == recipe_detailed_1d['loss_function'].args

    assert cnn.metrics[0].recipe_name == recipe_detailed_1d['metrics'][0].recipe_name
    assert cnn.metrics[0].instance.__class__ == recipe_detailed_1d['metrics'][0].instance.__class__
    assert cnn.metrics[0].args == recipe_detailed_1d['metrics'][0].args

    assert cnn.convolutional_layers == recipe_detailed_1d['convolutional_layers']
    assert cnn.dense_layers == recipe_detailed_1d['dense_layers']
    assert cnn.n_classes ==  recipe_detailed_1d['n_classes']

def test_CNN1DInterface_build_from_recipe_detailed_dict(recipe_detailed_1d, recipe_detailed_dict_1d):
    cnn = CNN1DInterface._build_from_recipe(recipe_detailed_dict_1d, recipe_compat=False)

    assert cnn.optimizer.recipe_name == recipe_detailed_1d['optimizer'].recipe_name
    assert cnn.optimizer.instance.__class__ == recipe_detailed_1d['optimizer'].instance.__class__
    assert cnn.optimizer.args == recipe_detailed_1d['optimizer'].args

    assert cnn.loss_function.recipe_name == recipe_detailed_1d['loss_function'].recipe_name
    assert cnn.loss_function.instance.__class__ == recipe_detailed_1d['loss_function'].instance.__class__
    assert cnn.loss_function.args == recipe_detailed_1d['loss_function'].args

    assert cnn.metrics[0].recipe_name == recipe_detailed_1d['metrics'][0].recipe_name
    assert cnn.metrics[0].instance.__class__ == recipe_detailed_1d['metrics'][0].instance.__class__
    assert cnn.metrics[0].args == recipe_detailed_1d['metrics'][0].args

    assert cnn.convolutional_layers == recipe_detailed_1d['convolutional_layers']
    assert cnn.dense_layers == recipe_detailed_1d['dense_layers']
    assert cnn.n_classes ==  recipe_detailed_1d['n_classes']

def test_write_recipe_simple_1d(recipe_simple_1d, recipe_simple_dict_1d, recipe_detailed_1d):
    cnn = CNN1DInterface._build_from_recipe(recipe_simple_1d)
    written_recipe = cnn._extract_recipe_dict()

    #Even when the model is built from a simplified recipe, the detailed form will be included when writing the recipe again

    recipe_simple_dict_1d['convolutional_layers'] = recipe_detailed_1d['convolutional_layers']
    recipe_simple_dict_1d['dense_layers'] = recipe_detailed_1d['dense_layers']

    assert written_recipe == recipe_simple_dict_1d
    

def test_read_recipe_simple_file_1d(recipe_simple_1d, recipe_simple_dict_1d, recipe_detailed_1d):
    path_to_recipe_file = os.path.join(path_to_tmp, "test_cnn_recipe_simple_1d.json")
    cnn = CNN1DInterface._build_from_recipe(recipe_simple_1d)
    #written_recipe = resnet.write_recipe()
    cnn.save_recipe_file(path_to_recipe_file)

    #Read recipe as a recipe dict
    read_recipe = cnn._read_recipe_file(path_to_recipe_file,return_recipe_compat=False)
    recipe_simple_dict_1d['convolutional_layers'] = recipe_detailed_1d['convolutional_layers']
    recipe_simple_dict_1d['dense_layers'] = recipe_detailed_1d['dense_layers']
    assert read_recipe == recipe_simple_dict_1d
    

    #Read recipe as a recipe dict with RecipeCompat objects
    recipe_simple_1d['convolutional_layers'] = recipe_detailed_1d['convolutional_layers']
    recipe_simple_1d['dense_layers'] = recipe_detailed_1d['dense_layers']

    read_recipe = cnn._read_recipe_file(path_to_recipe_file,return_recipe_compat=True)
    assert read_recipe['optimizer'].recipe_name == recipe_simple_1d['optimizer'].recipe_name
    assert read_recipe['optimizer'].instance.__class__ == recipe_simple_1d['optimizer'].instance.__class__
    assert read_recipe['optimizer'].args == recipe_simple_1d['optimizer'].args

    assert read_recipe['loss_function'].recipe_name == recipe_simple_1d['loss_function'].recipe_name
    assert read_recipe['loss_function'].instance.__class__ == recipe_simple_1d['loss_function'].instance.__class__
    assert read_recipe['loss_function'].args == recipe_simple_1d['loss_function'].args
    
    assert read_recipe['metrics'][0].recipe_name == recipe_simple_1d['metrics'][0].recipe_name
    assert read_recipe['metrics'][0].instance.__class__ == recipe_simple_1d['metrics'][0].instance.__class__
    assert read_recipe['metrics'][0].args == recipe_simple_1d['metrics'][0].args

    assert read_recipe['conv_set'] == recipe_simple_1d['conv_set']
    assert read_recipe['dense_set'] == recipe_simple_1d['dense_set']
    assert read_recipe['convolutional_layers'] == recipe_simple_1d['convolutional_layers']
    assert read_recipe['dense_layers'] == recipe_simple_1d['dense_layers']


def test_read_recipe_simple_detailed_1d(recipe_detailed_1d, recipe_detailed_dict_1d):
    path_to_recipe_file = os.path.join(path_to_tmp, "test_cnn_recipe_detailed_1d.json")
    cnn = CNN1DInterface._build_from_recipe(recipe_detailed_1d)
    #written_recipe = resnet.write_recipe()
    cnn.save_recipe_file(path_to_recipe_file)

    #Read recipe as a recipe dict
    read_recipe = cnn._read_recipe_file(path_to_recipe_file,return_recipe_compat=False)
    recipe_detailed_dict_1d['conv_set'] = None
    recipe_detailed_dict_1d['dense_set'] = None
    assert read_recipe == recipe_detailed_dict_1d
    
    #Read recipe as a recipe dict with RecipeCompat objects
    recipe_detailed_1d['conv_set'] = None
    recipe_detailed_1d['dense_set'] = None

    read_recipe = cnn._read_recipe_file(path_to_recipe_file,return_recipe_compat=True)
    assert read_recipe['optimizer'].recipe_name == recipe_detailed_1d['optimizer'].recipe_name
    assert read_recipe['optimizer'].instance.__class__ == recipe_detailed_1d['optimizer'].instance.__class__
    assert read_recipe['optimizer'].args == recipe_detailed_1d['optimizer'].args

    assert read_recipe['loss_function'].recipe_name == recipe_detailed_1d['loss_function'].recipe_name
    assert read_recipe['loss_function'].instance.__class__ == recipe_detailed_1d['loss_function'].instance.__class__
    assert read_recipe['loss_function'].args == recipe_detailed_1d['loss_function'].args
    
    assert read_recipe['metrics'][0].recipe_name == recipe_detailed_1d['metrics'][0].recipe_name
    assert read_recipe['metrics'][0].instance.__class__ == recipe_detailed_1d['metrics'][0].instance.__class__
    assert read_recipe['metrics'][0].args == recipe_detailed_1d['metrics'][0].args

    assert read_recipe['conv_set'] == recipe_detailed_1d['conv_set']
    assert read_recipe['dense_set'] == recipe_detailed_1d['dense_set']
    assert read_recipe['convolutional_layers'] == recipe_detailed_1d['convolutional_layers']
    assert read_recipe['dense_layers'] == recipe_detailed_1d['dense_layers']


def test_train_CNN1D(sample_data_1d):
    data, labels = sample_data_1d
    cnn = CNN1DInterface() #default cnn 1d
    train_generator = BatchGenerator(batch_size=5, x=data, y=labels, shuffle=True)
    val_generator = BatchGenerator(batch_size=5, x=data, y=labels, shuffle=True)

    cnn.train_generator = train_generator
    cnn.val_generator = val_generator

    cnn.train_loop(2, checkpoint_freq=0)
    




