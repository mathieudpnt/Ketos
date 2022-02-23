import numpy as np
np.random.seed(1000)

import tensorflow as tf
tf.random.set_seed(2000)



import pytest
#import numpy as np
#import tensorflow as tf
from ketos.neural_networks.dev_utils.nn_interface import RecipeCompat, NNInterface
from ketos.neural_networks.dev_utils.losses import FScoreLoss
#from ketos.neural_networks.metrics import Precision, Recall, Accuracy, FScore
from ketos.data_handling.data_feeding import BatchGenerator
from unittest.mock import Mock, patch
import os
import shutil
import tables
import json
from zipfile import ZipFile


current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')


@pytest.fixture
def batch_generator():
    data = np.vstack([np.zeros((10,512)), np.ones((10,512))])
    labels = np.concatenate([np.array([[1,0] for i in range(10)]), np.array([[0,1] for i in range(10)])])
    generator = BatchGenerator(batch_size=5, x=data, y=labels, shuffle=True)

    return generator


@pytest.fixture
def recipe_dict():
    recipe = {'optimizer': {'recipe_name':'Adam', 'parameters': {'learning_rate':0.005}},
               'loss_function': {'recipe_name':'FScoreLoss', 'parameters':{}},  
               'metrics': [{'recipe_name':'CategoricalAccuracy', 'parameters':{}}]
        
    }
    return recipe

@pytest.fixture
def MLPInterface_subclass():
    """ A simple MLP inheriting from NNInterface
    """
    
    class MLP(tf.keras.Model):
        def __init__(self, n_neurons, activation):
            super(MLP, self).__init__()

            self.dense = tf.keras.layers.Dense(n_neurons, activation=activation)
            self.final_node = tf.keras.layers.Dense(2, 'softmax')

        def call(self, inputs):
            print(inputs.shape)
            # output = self.dense(inputs)
            # output = tf.expand_dims(output, -1)
            # print(output.shape)
            output = self.dense(inputs)
            print(output.shape)
            output = self.final_node(output)
            print(output.shape)
            return output

            
    class MLPInterface(NNInterface):

        @classmethod
        def _transform_input(cls, input):
            if input.ndim == 1:
                transformed_input = input.reshape(1,input.shape[0])
            elif input.ndim == 2:
                transformed_input = input.reshape(input.shape[0],input.shape[1])
            else:
                raise ValueError("Expected input to have 1 or 2 dimensions, got {}({}) instead".format(input.ndims, input.shape))

            return transformed_input

        @classmethod
        def _transform_output(cls,output):
            max_class = np.argmax(output, axis=-1)
            if output.shape[0] == 1:
                max_class_conf = output[0][max_class]
                transformed_output = (max_class[0], max_class_conf[0])
            elif output.shape[0] > 1:
                max_class_conf = np.array([output[i][c] for i, c in enumerate(max_class)])

            transformed_output = (max_class, max_class_conf)
            
            return transformed_output


        # @classmethod
        # def transform_train_batch(cls, x, y, n_classes=2):
        #     X = x
        #     Y = np.array([cls._to1hot(class_label=label, n_classes=n_classes) for label in y['label']])
        #     return (X, Y)



        @classmethod
        def build_from_recipe(cls, recipe):
            n_neurons = recipe['n_neurons']
            activation = recipe['activation']
            optimizer = recipe['optimizer']
            loss_function = recipe['loss_function']
            metrics = recipe['metrics']

            instance = cls(n_neurons=n_neurons, activation=activation, optimizer=optimizer, loss_function=loss_function, metrics=metrics, secondary_metrics=secondary_metrics)

            return instance

        @classmethod
        def _read_recipe_file(cls, json_file, return_recipe_compat=True):
            
            with open(json_file, 'r') as json_recipe:
                recipe_dict = json.load(json_recipe)
            
            
            optimizer = cls._optimizer_from_recipe(recipe_dict['optimizer'])
            loss_function = cls._loss_function_from_recipe(recipe_dict['loss_function'])
            metrics = cls._metrics_from_recipe(recipe_dict['metrics'])
            

            if return_recipe_compat == True:
                recipe_dict['optimizer'] = optimizer
                recipe_dict['loss_function'] = loss_function
                recipe_dict['metrics'] = metrics
            
            else:
                recipe_dict['optimizer'] = cls._optimizer_to_recipe(optimizer)
                recipe_dict['loss_function'] = cls._loss_function_to_recipe(loss_function)
                recipe_dict['metrics'] = cls._metrics_to_recipe(metrics)
            
            recipe_dict['n_neurons'] = recipe_dict['n_neurons']
            recipe_dict['activation'] = recipe_dict['activation']
            return recipe_dict

        def __init__(self, n_neurons, activation, optimizer, loss_function, metrics):
            super(MLPInterface, self).__init__(optimizer, loss_function, metrics)
            self.n_neurons = n_neurons
            self.activation = activation

            # self.optimizer=optimizer
            # self.loss_function=loss_function
            # self.metrics=metrics
            
            self.model = MLP(n_neurons=n_neurons, activation=activation)
            

        def _extract_recipe_dict(self):
        
            recipe = {}
            recipe['optimizer'] = self._optimizer_to_recipe(self.optimizer)
            recipe['loss_function'] = self._loss_function_to_recipe(self.loss_function)
            recipe['metrics'] = self._metrics_to_recipe(self.metrics)
            recipe['n_neurons'] = self.n_neurons
            recipe['activation'] = self.activation

            return recipe

    return MLPInterface


@pytest.fixture
def instance_of_MLPInterface(MLPInterface_subclass):
    np.random.seed(1000)
    tf.random.set_seed(2000)
    path_to_file = os.path.join(path_to_assets, "recipes/basic_recipe.json")
    recipe = NNInterface._read_recipe_file(path_to_file)

    h5 = tables.open_file(os.path.join(path_to_assets, "vectors_1_0.h5"), 'r')
    train_table = h5.get_node("/train")
    val_table = h5.get_node("/val")
    test_table = h5.get_node("/test")

    data = np.vstack([np.zeros((10,512)), np.ones((10,512))])
    labels = np.concatenate([np.array([[1,0] for i in range(10)]), np.array([[0,1] for i in range(10)])])
    train_generator = BatchGenerator(batch_size=5, x=data, y=labels, shuffle=True)
    val_generator = BatchGenerator(batch_size=5, x=data, y=labels, shuffle=True)
    test_generator = BatchGenerator(batch_size=5, x=data, y=labels, shuffle=True)
    # train_generator = BatchGenerator(batch_size=5, data_table=train_table, output_transform_func=MLPInterface_subclass.transform_train_batch, x_field='data', y_field='label')
    # val_generator = BatchGenerator(batch_size=5, data_table=val_table, output_transform_func=MLPInterface_subclass.transform_train_batch, x_field='data', y_field='label')
    # test_generator = BatchGenerator(batch_size=5, data_table=test_table, output_transform_func=MLPInterface_subclass.transform_train_batch, x_field='data', y_field='label')
    

    instance = MLPInterface_subclass(activation='relu', n_neurons=64, optimizer=recipe['optimizer'],
                         loss_function=recipe['loss_function'], metrics=recipe['metrics']) 

    instance.train_generator = train_generator
    instance.val_generator = val_generator
    instance.test_generator = test_generator
    
    return instance


def test_RecipeCompat():
    opt = RecipeCompat("sgd", tf.keras.optimizers.SGD, learning_rate=0.008, momentum=0.1)
    assert opt.recipe_name == "sgd"
    assert opt.args ==  {'learning_rate': 0.008, 'momentum': 0.1}
    assert isinstance(opt.instance, tf.keras.optimizers.SGD) 
    assert opt.instance.learning_rate == 0.008
    assert opt.instance.momentum == 0.1

    metric = RecipeCompat("accuracy", tf.keras.metrics.Accuracy, name='acc')
    assert metric.recipe_name == "accuracy"
    assert metric.args == {'name': 'acc'}
    assert isinstance(metric.instance, tf.keras.metrics.Accuracy) 
    assert metric.instance.name == 'acc'

 

@pytest.mark.parametrize("class_label,n_classes,expected",[
    (0,2,np.array([1.0,0.0])),
    (1,2,np.array([0.0,1.0])),
    (1,3,np.array([0.0,1.0, 0.0])),
    (1,5,np.array([0.0,1.0, 0.0, 0.0, 0.0])),
    ])
def test_to1hot(class_label, n_classes, expected):
    encoded =  NNInterface._to1hot(class_label, n_classes) 
    assert(encoded == expected).all()
    with pytest.raises(IndexError):
        assert NNInterface._to1hot(class_label=1.2, n_classes=2) 
    with pytest.raises(TypeError):
        assert NNInterface._to1hot(class_label=1, n_classes=2.0)



def test_transform_batch():
    inputs = np.random.rand(10,5,5)
    labels = np.array([1,0,0,0,1,0,0,1,1,1])

    X,Y = NNInterface.transform_batch(inputs, labels)

    
    assert X.shape == (10,5,5,1)
    assert np.array_equal(X[:,:,:,0], inputs)
    assert np.array_equal(Y, np.array([[0., 1.],
                            [1., 0.],
                            [1., 0.],
                            [1., 0.],
                            [0., 1.],
                            [1., 0.],
                            [1., 0.],
                            [0., 1.],
                            [0., 1.],
                            [0., 1.]]))


def test_transform_input():
     input1 = np.random.rand(5,5)        
     input2 = np.random.rand(1,5,5)

     output1 = NNInterface._transform_input(input1)
     output2 = NNInterface._transform_input(input2)

     assert output1.shape == (1,5,5,1)
     assert output2.shape == (1,5,5,1)

     assert np.array_equal(output1[0,:,:,0], input1)
     assert np.array_equal(output2[0,:,:,0], input2[0,:,:])


def test_transform_output():
    output1 = np.array([[0.05,0.05,0.7, 0.1, 0.1]])
    output2 = np.array([[0.05,0.05,0.7, 0.1, 0.1],[0.05,0.15,0.15,0.65,0.1]])

    class1, score1 = NNInterface._transform_output(output1)
    classes2, scores2 = NNInterface._transform_output(output2)

    assert np.array_equal(class1, np.array([2])) 
    assert np.array_equal(score1, np.array([0.7]))
    
    assert np.array_equal(classes2, np.array([2, 3])) 
    assert np.array_equal(scores2, np.array([0.7, 0.65]))
    


def test_optimizer_from_recipe(recipe_dict):
    built_opt = NNInterface._optimizer_from_recipe(recipe_dict['optimizer'])
    assert isinstance(built_opt, RecipeCompat)
    assert built_opt.recipe_name == 'Adam'
    assert built_opt.args == {'learning_rate':0.005}
    assert isinstance(built_opt.instance, tf.keras.optimizers.Adam)
    
    
def test_loss_function_from_recipe(recipe_dict):
    built_loss = NNInterface._loss_function_from_recipe(recipe_dict['loss_function'])
    assert isinstance(built_loss, RecipeCompat)
    assert built_loss.recipe_name == 'FScoreLoss'
    assert built_loss.args == {}
    assert isinstance(built_loss.instance, FScoreLoss)
    
def test_metrics_from_recipe(recipe_dict):
    built_metrics = NNInterface._metrics_from_recipe(recipe_dict['metrics'])
    assert isinstance(built_metrics[0], RecipeCompat)
    assert built_metrics[0].recipe_name == 'CategoricalAccuracy'
    assert built_metrics[0].args == {}
    assert isinstance(built_metrics[0].instance, tf.keras.metrics.CategoricalAccuracy)
    
    
def test_optimizer_to_recipe():
    recipe_compat_opt = RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005)
    optimizer_dict = NNInterface._optimizer_to_recipe(recipe_compat_opt)

    assert optimizer_dict['recipe_name'] == 'Adam'
    assert optimizer_dict['parameters'] == {'learning_rate':0.005}


def test_loss_function_to_recipe():
    recipe_compat_loss = RecipeCompat('FScoreLoss', FScoreLoss, beta=0.5)
    loss_dict = NNInterface._loss_function_to_recipe(recipe_compat_loss)

    assert loss_dict['recipe_name'] == 'FScoreLoss'
    assert loss_dict['parameters'] == {'beta':0.5}

    
def test_metrics_to_recipe():
    recipe_compat_metric = [RecipeCompat('CategoricalAccuracy', tf.keras.metrics.CategoricalAccuracy)]
    metrics_dicts = NNInterface._metrics_to_recipe(recipe_compat_metric)

    assert isinstance(metrics_dicts, list)
    assert metrics_dicts[0]['recipe_name'] == 'CategoricalAccuracy'
    assert metrics_dicts[0]['parameters'] == {}


def test_read_recipe_file():
    path_to_file = os.path.join(path_to_assets, "recipes/basic_recipe.json")
    recipe = NNInterface._read_recipe_file(path_to_file)

    assert isinstance(recipe, dict)
    assert isinstance(recipe['optimizer'], RecipeCompat)
    assert isinstance(recipe['loss_function'], RecipeCompat)
    assert isinstance(recipe['metrics'], list)
    assert isinstance(recipe['metrics'][0], RecipeCompat)

    opt = recipe['optimizer']
    assert opt.recipe_name == 'Adam'
    assert isinstance(opt.instance, tf.keras.optimizers.Adam)
    assert opt.args == {"learning_rate": 0.001, "beta_1": 0.9, "beta_2": 0.999, "decay": 0.01}

    loss = recipe['loss_function']
    assert loss.recipe_name == 'FScoreLoss'
    assert isinstance(loss.instance, FScoreLoss)
    assert loss.args == {}

    metric = recipe['metrics'][0]
    assert metric.recipe_name == 'CategoricalAccuracy'
    assert isinstance(metric.instance, tf.keras.metrics.CategoricalAccuracy)
    assert metric.args == {}


def test_write_recipe_file(recipe_dict):
    destination = os.path.join(path_to_tmp, "test_write_recipe_file.json")
    NNInterface._write_recipe_file(destination, recipe_dict)

    read_recipe =  NNInterface._read_recipe_file(destination, return_recipe_compat=False)
    #If return_recipe_compat is False, the result should be a recipe dictionary just like the recipe_dict used to write the file
    assert read_recipe == recipe_dict
    # assert read_recipe['optimizer'] == recipe_dict['optimizer']
    # assert read_recipe['loss_function'] == recipe_dict['loss_function']
    # assert read_recipe['metrics'] == recipe_dict['metrics']

    read_recipe =  NNInterface._read_recipe_file(destination, return_recipe_compat=True)
    #If return_recipe_compat is True, the result will be a dictionary with RecipeCompat objects for the optimizer, loss_functions and metrics (actually a lis of RecipCompat objects) 

    assert read_recipe['optimizer'].recipe_name == NNInterface._optimizer_from_recipe(recipe_dict['optimizer']).recipe_name
    assert read_recipe['optimizer'].instance.__class__ == NNInterface._optimizer_from_recipe(recipe_dict['optimizer']).instance.__class__
    assert read_recipe['optimizer'].args == NNInterface._optimizer_from_recipe(recipe_dict['optimizer']).args

    assert read_recipe['loss_function'].recipe_name == NNInterface._loss_function_from_recipe(recipe_dict['loss_function']).recipe_name
    assert read_recipe['loss_function'].instance.__class__ == NNInterface._loss_function_from_recipe(recipe_dict['loss_function']).instance.__class__
    assert read_recipe['loss_function'].args == NNInterface._loss_function_from_recipe(recipe_dict['loss_function']).args
    
    assert read_recipe['metrics'][0].recipe_name == NNInterface._metrics_from_recipe(recipe_dict['metrics'])[0].recipe_name
    assert read_recipe['metrics'][0].instance.__class__ == NNInterface._metrics_from_recipe(recipe_dict['metrics'])[0].instance.__class__
    assert read_recipe['metrics'][0].args == NNInterface._metrics_from_recipe(recipe_dict['metrics'])[0].args
    

    
def test_instantiate_nn(MLPInterface_subclass):
    path_to_file = os.path.join(path_to_assets, "recipes/basic_recipe.json")
    recipe = NNInterface._read_recipe_file(path_to_file)
    
    MLPInterface_subclass(activation='relu', n_neurons=64, optimizer=recipe['optimizer'],
                         loss_function=recipe['loss_function'], metrics=recipe['metrics'])    

def test_save_recipe_file(instance_of_MLPInterface):
    path_to_file = os.path.join(path_to_assets, "recipes/basic_recipe.json")
    recipe = NNInterface._read_recipe_file(path_to_file)

    instance = instance_of_MLPInterface

    path_to_saved_recipe = os.path.join(path_to_tmp, "test_save_recipe.json")
    instance.save_recipe_file(path_to_saved_recipe)

    read_recipe = instance._read_recipe_file(path_to_saved_recipe)

    #assert read_recipe == recipe

    assert read_recipe['n_neurons'] == 64
    assert read_recipe['activation'] == 'relu'
    assert read_recipe['optimizer'].recipe_name ==recipe['optimizer'].recipe_name
    assert read_recipe['optimizer'].instance.__class__ == recipe['optimizer'].instance.__class__
    assert read_recipe['optimizer'].args == recipe['optimizer'].args

    assert read_recipe['loss_function'].recipe_name == recipe['loss_function'].recipe_name
    assert read_recipe['loss_function'].instance.__class__ == recipe['loss_function'].instance.__class__
    assert read_recipe['loss_function'].args == recipe['loss_function'].args
    
    assert read_recipe['metrics'][0].recipe_name == recipe['metrics'][0].recipe_name
    assert read_recipe['metrics'][0].instance.__class__ == recipe['metrics'][0].instance.__class__
    assert read_recipe['metrics'][0].args == recipe['metrics'][0].args


def test_extract_recipe_dict(instance_of_MLPInterface):
    recipe = instance_of_MLPInterface._extract_recipe_dict()
    
    assert recipe['n_neurons'] == 64
    assert recipe['activation'] == 'relu'

    assert recipe['optimizer'] == {'recipe_name':'Adam', 'parameters': {'beta_1': 0.9, 'beta_2': 0.999, 'decay': 0.01, 'learning_rate': 0.001}}
    assert recipe['loss_function'] == {'recipe_name':'FScoreLoss', 'parameters':{}}
    assert recipe['metrics'] == [{'recipe_name':'CategoricalAccuracy', 'parameters':{}}]


def test_train_loop_early_stop_metric_decrease(instance_of_MLPInterface):
   
    train_loss_values = iter([0.9,0.9,0.9,0.9,
                              0.8,0.8,0.8,0.8,
                              0.7,0.7,0.7,0.7,
                              0.7,0.7,0.7,0.7,
                              0.7,0.7,0.7,0.7,
                              0.7,0.7,0.7,0.7,
                              0.7,0.7,0.7,0.7,
                              0.7,0.7,0.7,0.7,
                              0.6,0.6,0.6,0.6,
                              0.5,0.5,0.5,0.5,
                              0.4,0.4,0.4,0.4,
                              0.3,0.3,0.3,0.3,
                              0.2,0.2,0.2,0.2,
                              0.1,0.1,0.1,0.1,])

    def mock_train_step(self, inputs, labels):
       self._train_loss = Mock()
       loss = next(train_loss_values)
       print(loss)
       self._train_loss.result.return_value = loss#next(train_loss_values)

    patcher = patch('ketos.neural_networks.dev_utils.nn_interface.NNInterface._train_step', new=mock_train_step)
    patcher.start()
    
    instance_of_MLPInterface.checkpoint_dir = os.path.join(path_to_tmp, "test_train_loop_checkpoints")
    

    instance_of_MLPInterface._early_stopping_monitor = {"metric": 'train_loss',
                                        "decreasing": True,
                                        "period":3,
                                        "min_epochs": 3,
                                        "max_epochs": None,
                                        "delta" : 0.1,
                                        "baseline":0.5}
    
    instance_of_MLPInterface.train_loop(n_epochs=8, early_stopping=True)
    assert instance_of_MLPInterface.last_epoch_with_improvement == 3
 
    patcher.stop()
    
def test_train_loop_early_stop_metric_decrease_baseline(instance_of_MLPInterface):
   
    train_loss_values = iter([0.9,0.9,0.9,0.9,
                              0.8,0.8,0.8,0.8,
                              0.7,0.7,0.7,0.7,
                              0.7,0.7,0.7,0.7,
                              0.5,0.5,0.5,0.5,
                              0.4,0.4,0.4,0.4,
                              0.3,0.3,0.3,0.3,
                              0.2,0.2,0.2,0.2,
                              0.1,0.1,0.1,0.1,])

    def mock_train_step(self, inputs, labels):
       self._train_loss = Mock()
       loss = next(train_loss_values)
       print(loss)
       self._train_loss.result.return_value = loss#next(train_loss_values)

    patcher = patch('ketos.neural_networks.dev_utils.nn_interface.NNInterface._train_step', new=mock_train_step)
    patcher.start()
    
    instance_of_MLPInterface.checkpoint_dir = os.path.join(path_to_tmp, "test_train_loop_checkpoints")

    instance_of_MLPInterface._early_stopping_monitor = {"metric": 'train_loss',
                                        "decreasing": True,
                                        "period":3,
                                        "min_epochs": 3,
                                        "max_epochs": None,
                                        "delta" : 0.1,
                                        "baseline":0.5}
    
    instance_of_MLPInterface.train_loop(n_epochs=8, early_stopping=True)
    assert instance_of_MLPInterface.last_epoch_with_improvement == 4
 
    patcher.stop()
    
def test_train_loop_early_stop_metric_increase(instance_of_MLPInterface):
   
    train_loss_values = iter([0.1,0.1,0.1,0.1,
                             0.2,0.2,0.2,0.2,
                             0.3,0.3,0.3,0.3,
                             0.3,0.3,0.3,0.3,
                             0.3,0.3,0.3,0.3,
                             0.3,0.3,0.3,0.3,
                             0.3,0.3,0.3,0.3,
                             0.3,0.3,0.3,0.3,
                             0.3,0.3,0.3,0.3,
                             0.3,0.3,0.3,0.3,
                             0.4,0.4,0.4,0.4,
                             0.4,0.4,0.4,0.4,
                             0.5,0.5,0.5,0.5,
                             0.6,0.6,0.6,0.6,
                                                        ])

    def mock_train_step(self, inputs, labels):
       self._train_loss = Mock()
       loss = next(train_loss_values)
       print(loss)
       self._train_loss.result.return_value = loss#next(train_loss_values)

    patcher = patch('ketos.neural_networks.dev_utils.nn_interface.NNInterface._train_step', new=mock_train_step)
    patcher.start()
    
    instance_of_MLPInterface.checkpoint_dir = os.path.join(path_to_tmp, "test_train_loop_checkpoints")

    instance_of_MLPInterface._early_stopping_monitor = {"metric": 'train_loss',
                                        "decreasing": True,
                                        "period":3,
                                        "min_epochs": 3,
                                        "max_epochs": None,
                                        "delta" : 0.1,
                                        "baseline":1.0}
    
    instance_of_MLPInterface.train_loop(n_epochs=8, early_stopping=True)
    assert instance_of_MLPInterface.last_epoch_with_improvement == 3
 
    patcher.stop()
    
def test_train_loop(instance_of_MLPInterface):
    instance_of_MLPInterface.checkpoint_dir = os.path.join(path_to_tmp, "test_train_loop_checkpoints")
    instance_of_MLPInterface.train_loop(n_epochs=5)

def test_save_model_latest(instance_of_MLPInterface):
    instance_of_MLPInterface.checkpoint_dir = os.path.join(path_to_tmp, "test_save_model_latest_checkpoints")
    instance_of_MLPInterface.train_loop(n_epochs=5, checkpoint_freq=1)
    instance_of_MLPInterface.save_model(os.path.join(path_to_tmp, "test_save_model_latest.kt"))
    extraction_path = os.path.join(path_to_tmp, "test_save_model_latest_tmp_folder")
    with ZipFile(os.path.join(path_to_tmp, "test_save_model_latest.kt"), 'r') as zip:
            zip.extractall(path=extraction_path)
    
    assert os.path.exists(os.path.join(extraction_path,"checkpoints", "cp-0005.ckpt.data-00000-of-00001" ))
    assert os.path.exists(os.path.join(extraction_path, "checkpoints", "cp-0005.ckpt.index"))
    assert os.path.exists(os.path.join(extraction_path, "checkpoints", "checkpoint"))

def test_save_model_checkpoint_name(instance_of_MLPInterface):
    instance_of_MLPInterface.checkpoint_dir = os.path.join(path_to_tmp, "test_save_model_latest_checkpoints")
    instance_of_MLPInterface.train_loop(n_epochs=5, checkpoint_freq=1)
    instance_of_MLPInterface.save_model(os.path.join(path_to_tmp, "test_save_model_latest.kt"), checkpoint_name="cp-0003.ckpt")
    extraction_path = os.path.join(path_to_tmp, "test_save_model_latest_tmp_folder")
    with ZipFile(os.path.join(path_to_tmp, "test_save_model_latest.kt"), 'r') as zip:
            zip.extractall(path=extraction_path)
    
    assert os.path.exists(os.path.join(extraction_path,"checkpoints", "cp-0003.ckpt.data-00000-of-00001" ))
    assert os.path.exists(os.path.join(extraction_path, "checkpoints", "cp-0003.ckpt.index"))
    assert os.path.exists(os.path.join(extraction_path, "checkpoints", "checkpoint"))

def test_save_model_with_audio_repr(instance_of_MLPInterface, spectr_settings):
    instance_of_MLPInterface.checkpoint_dir = os.path.join(path_to_tmp, "test_save_model_latest_checkpoints")
    instance_of_MLPInterface.train_loop(n_epochs=5, checkpoint_freq=1)
    instance_of_MLPInterface.save_model(os.path.join(path_to_tmp, "test_save_model_latest.kt"), audio_repr=spectr_settings)
    extraction_path = os.path.join(path_to_tmp, "test_save_model_latest_tmp_folder")
    with ZipFile(os.path.join(path_to_tmp, "test_save_model_latest.kt"), 'r') as zip:
            zip.extractall(path=extraction_path)
    
    assert os.path.exists(os.path.join(extraction_path,"checkpoints", "cp-0005.ckpt.data-00000-of-00001" ))
    assert os.path.exists(os.path.join(extraction_path, "checkpoints", "cp-0005.ckpt.index"))
    assert os.path.exists(os.path.join(extraction_path, "checkpoints", "checkpoint"))
    assert os.path.exists(os.path.join(extraction_path, "audio_repr.json"))

def test_save_model_with_audio_repr_file(instance_of_MLPInterface):
    audio_repr_file = os.path.join(path_to_assets, "audio_repr.json")
    instance_of_MLPInterface.checkpoint_dir = os.path.join(path_to_tmp, "test_save_model_latest_checkpoints")
    instance_of_MLPInterface.train_loop(n_epochs=5, checkpoint_freq=1)
    instance_of_MLPInterface.save_model(os.path.join(path_to_tmp, "test_save_model_latest.kt"), audio_repr_file=audio_repr_file)
    extraction_path = os.path.join(path_to_tmp, "test_save_model_latest_tmp_folder")
    with ZipFile(os.path.join(path_to_tmp, "test_save_model_latest.kt"), 'r') as zip:
            zip.extractall(path=extraction_path)
    
    assert os.path.exists(os.path.join(extraction_path,"checkpoints", "cp-0005.ckpt.data-00000-of-00001" ))
    assert os.path.exists(os.path.join(extraction_path, "checkpoints", "cp-0005.ckpt.index"))
    assert os.path.exists(os.path.join(extraction_path, "checkpoints", "checkpoint"))
    assert os.path.exists(os.path.join(extraction_path, "audio_repr.json"))

def test_train_loop_log_csv(MLPInterface_subclass,batch_generator):
    recipe = { 'n_neurons':64,
               'activation':'relu',
               'optimizer': RecipeCompat("Adam",tf.keras.optimizers.Adam,learning_rate=0.005),
               'loss_function': RecipeCompat("FScoreLoss",FScoreLoss),  
               'metrics': [RecipeCompat('CategoricalAccuracy', tf.keras.metrics.CategoricalAccuracy),
                            RecipeCompat('Precision', tf.keras.metrics.Precision),
                            RecipeCompat('Recall', tf.keras.metrics.Recall)],
            }

    h5 = tables.open_file(os.path.join(path_to_assets, "vectors_1_0.h5"), 'r')
    train_table = h5.get_node("/train")
    val_table = h5.get_node("/val")
    test_table = h5.get_node("/test")

    instance = MLPInterface_subclass(activation='relu', n_neurons=64, optimizer=recipe['optimizer'],
                         loss_function=recipe['loss_function'], metrics=recipe['metrics']) 

    instance.train_generator = batch_generator
    instance.val_generator = batch_generator
    instance.log_dir = os.path.join(path_to_tmp, "test_log_dir")
    instance.checkpoint_dir = os.path.join(path_to_tmp, "test_train_loop_log_csv_checkpoints")

    instance.train_loop(15, log_csv=True)

    assert os.path.isfile(os.path.join(instance.log_dir, "log.csv"))
    os.remove(os.path.join(instance.log_dir, "log.csv"))

def test_train_loop_log_tensorboard(MLPInterface_subclass):
    recipe = { 'n_neurons':64,
               'activation':'relu',
               'optimizer': RecipeCompat("Adam",tf.keras.optimizers.Adam,learning_rate=0.005),
               'loss_function': RecipeCompat("FScoreLoss",FScoreLoss),  
               'metrics': [RecipeCompat('CategoricalAccuracy', tf.keras.metrics.CategoricalAccuracy),
                            RecipeCompat('Precision', tf.keras.metrics.Precision),
                            RecipeCompat('Recall', tf.keras.metrics.Recall)],
               
            }

    h5 = tables.open_file(os.path.join(path_to_assets, "vectors_1_0.h5"), 'r')
    train_table = h5.get_node("/train")
    val_table = h5.get_node("/val")
    test_table = h5.get_node("/test")

    data = np.vstack([np.zeros((10,512)), np.ones((10,512))])
    labels = np.concatenate([np.array([[1,0] for i in range(10)]), np.array([[0,1] for i in range(10)])])
    train_generator = BatchGenerator(batch_size=5, x=data, y=labels, shuffle=True)
    val_generator = BatchGenerator(batch_size=5, x=data, y=labels, shuffle=True)
    test_generator = BatchGenerator(batch_size=5, x=data, y=labels, shuffle=True)
    
    instance = MLPInterface_subclass(activation='relu', n_neurons=64, optimizer=recipe['optimizer'],
                         loss_function=recipe['loss_function'], metrics=recipe['metrics']) 

    instance.train_generator = train_generator
    instance.val_generator = val_generator
    instance.log_dir = os.path.join(path_to_tmp, "test_log_dir")
    instance.checkpoint_dir = os.path.join(path_to_tmp, "test_train_loop_log_tensorboard_checkpoints")

    instance.train_loop(15, log_tensorboard=True)

    assert os.path.isdir(os.path.join(instance.log_dir, "tensorboard_metrics"))
    shutil.rmtree(os.path.join(instance.log_dir, "tensorboard_metrics"))


def test_run_on_batch(instance_of_MLPInterface):
    instance_of_MLPInterface.checkpoint_dir = os.path.join(path_to_tmp, "test_train_loop_checkpoints")
    instance_of_MLPInterface.train_loop(n_epochs=5)
    instance_of_MLPInterface.test_generator.batch_count = 0 #reset batch generator
    X,y = next(instance_of_MLPInterface.test_generator)

    pred, scores = instance_of_MLPInterface.run_on_batch(X, transform_input=True, return_raw_output=False)
    
    assert pred.shape == (5,)
    assert np.array_equal(pred, np.array([0,1,1,0,0]))

    assert scores.shape == (5,)
    np.testing.assert_array_almost_equal(scores, np.array([0.5086525,1.,1., 0.5086525, 0.5086525], dtype=scores.dtype), decimal=7)

def test_run_on_batch_raw_output(instance_of_MLPInterface):
    instance_of_MLPInterface.checkpoint_dir = os.path.join(path_to_tmp, "test_train_loop_checkpoints")
    instance_of_MLPInterface.train_loop(n_epochs=5)
    X,_ = next(instance_of_MLPInterface.test_generator)

    scores = instance_of_MLPInterface.run_on_batch(X, transform_input=True, return_raw_output=True)

    assert scores.shape == (5,2)
    assert np.allclose(scores, np.array([[5.0865251e-01, 4.9134743e-01],
                                            [7.2768415e-12, 1.0000000e+00],
                                            [7.2768415e-12, 1.0000000e+00],
                                            [5.0865251e-01, 4.9134743e-01],
                                            [5.0865251e-01, 4.9134743e-01]], dtype=scores.dtype), atol=1e-08)

def test_run_on_instance(instance_of_MLPInterface):
    instance_of_MLPInterface.checkpoint_dir = os.path.join(path_to_tmp, "test_train_loop_checkpoints")
    instance_of_MLPInterface.train_loop(n_epochs=5)
    instance_of_MLPInterface.test_generator.batch_count = 0 #reset batch generator
    X,_ = next(instance_of_MLPInterface.test_generator)
    x = X[0:1]
    
    pred, scores = instance_of_MLPInterface.run_on_instance(x, return_raw_output=False)

    assert pred.shape == (1,)
    assert pred == 0
    assert scores.shape == (1,)
    np.testing.assert_array_almost_equal(scores, np.array([0.5086525], dtype=scores.dtype), decimal=7)

def test_run_on_instance_raw_output(instance_of_MLPInterface):
    instance_of_MLPInterface.checkpoint_dir = os.path.join(path_to_tmp, "test_train_loop_checkpoints")
    instance_of_MLPInterface.train_loop(n_epochs=5)
    instance_of_MLPInterface.test_generator.batch_count = 0 #reset batch generator
    X,_ = next(instance_of_MLPInterface.test_generator)
    x = X[0]
    
    scores = instance_of_MLPInterface.run_on_instance(x, return_raw_output=True)
    assert scores.shape == (1,2)
    np.testing.assert_array_almost_equal(scores, np.array([[0.5086525 , 0.49134743]], dtype=scores.dtype), decimal=7)
     
def test_run_on_test_generator(instance_of_MLPInterface):
    instance_of_MLPInterface.checkpoint_dir = os.path.join(path_to_tmp, "test_train_loop_checkpoints")
    instance_of_MLPInterface.train_loop(n_epochs=5)
    pred, scores = instance_of_MLPInterface.run_on_test_generator(compute_val_metrics=True, return_raw_output=False)

    assert pred.shape == (20,)
    assert np.array_equal(pred, np.array([0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1]))

    assert scores.shape == (20,)
    np.testing.assert_array_almost_equal(scores, np.array([0.5086525, 1.,1., 0.5086525, 0.5086525,
                                          0.5086525, 1., 0.5086525, 1.,0.5086525,
                                          0.5086525, 1., 0.5086525, 1., 1.,
                                          1., 0.5086525, 1., 0.5086525, 1.], dtype=scores.dtype), decimal=7)

def test_run_on_test_generator_metrics(instance_of_MLPInterface):
    """ Test that the run_on_test_generator method returns validation metrics """
    instance_of_MLPInterface.checkpoint_dir = os.path.join(path_to_tmp, "test_train_loop_checkpoints")
    instance_of_MLPInterface.train_loop(n_epochs=5)
    _, metrics = instance_of_MLPInterface.run_on_test_generator(compute_val_metrics=True, return_raw_output=False, return_metrics=True)
    
    assert metrics['CategoricalAccuracy'] == 1.0
    assert metrics['loss'].numpy() == pytest.approx(0.2456744, abs=0.00001)

def test_run_on_test_generator_raw_output(instance_of_MLPInterface):
    instance_of_MLPInterface.checkpoint_dir = os.path.join(path_to_tmp, "test_train_loop_checkpoints")
    instance_of_MLPInterface.train_loop(n_epochs=5)
    scores = instance_of_MLPInterface.run_on_test_generator(compute_val_metrics=True, return_raw_output=True)

    assert scores.shape == (4,5,2,)
    assert np.allclose(scores, np.array([[[5.0865251e-01, 4.9134743e-01],
                                            [7.2768415e-12, 1.0000000e+00],
                                            [7.2768415e-12, 1.0000000e+00],
                                            [5.0865251e-01, 4.9134743e-01],
                                            [5.0865251e-01, 4.9134743e-01]],

                                            [[5.0865251e-01, 4.9134743e-01],
                                            [7.2768415e-12, 1.0000000e+00],
                                            [5.0865251e-01, 4.9134743e-01],
                                            [7.2768415e-12, 1.0000000e+00],
                                            [5.0865251e-01, 4.9134743e-01]],

                                            [[5.0865251e-01, 4.9134743e-01],
                                            [7.2768415e-12, 1.0000000e+00],
                                            [5.0865251e-01, 4.9134743e-01],
                                            [7.2768415e-12, 1.0000000e+00],
                                            [7.2768415e-12, 1.0000000e+00]],

                                            [[7.2768415e-12, 1.0000000e+00],
                                            [5.0865251e-01, 4.9134743e-01],
                                            [7.2768415e-12, 1.0000000e+00],
                                            [5.0865251e-01, 4.9134743e-01],
                                            [7.2768415e-12, 1.0000000e+00]]], dtype=scores.dtype), atol=1e-08)
