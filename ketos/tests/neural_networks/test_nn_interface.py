import pytest
import numpy as np
import tensorflow as tf
from ketos.neural_networks.nn_interface import RecipeCompat, NNInterface
from ketos.neural_networks.losses import FScoreLoss
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')

@pytest.fixture
def recipe_dict():
    recipe = {'optimizer': {'name':'Adam', 'parameters': {'learning_rate':0.005}},
               'loss_function': {'name':'FScoreLoss', 'parameters':{}},  
               'metrics': [{'name':'CategoricalAccuracy', 'parameters':{}}]
        
    }
    return recipe

@pytest.fixture
def NNInterface_subclass():
    
        class MLP(tf.keras.Model):
            def __init__(self, n_neurons, activation):
                super(MLP, self).__init__()

                self.dense = tf.keras.layers.Dense(n_neurons, activation=activation)
                self.final_node = tf.keras.layers.Dense(1)

            def call(self, inputs):
                output = self.dense(inputs)
                output = self.dense(output)
                output = self.final_node(output)

               
        class MLPInterface(NNInterface):

            @classmethod
            def build_from_recipe(cls, recipe):
                n_neurons = recipe['n_neurons']
                activation = recipe['activation']
                optimizer = recipe['optimizer']
                loss_function = recipe['loss_function']
                metrics = recipe['metrics']

                instance = cls(n_neurons=n_neurons, activation=activation, optimizer=optimizer, loss_function=loss_function, metrics=metrics)

                return instance

            def __init__(self, n_neurons, activation, optimizer, loss_function, metrics):
                #super(MLPInterface, self).__init__(optimizer, loss_function, metrics)
                self.model = MLP(n_neurons=n_neurons, activation=activation)
                self.optimizer=optimizer
                self.loss_function=loss_function
                self.metrics=metrics

        return MLPInterface



def test_RecipeCompat():
    opt = RecipeCompat("sgd", tf.keras.optimizers.SGD, learning_rate=0.008, momentum=0.1)
    assert opt.name == "sgd"
    assert opt.args ==  {'learning_rate': 0.008, 'momentum': 0.1}
    assert isinstance(opt.func, tf.keras.optimizers.SGD) 

    metric = RecipeCompat("accuracy", tf.keras.metrics.Accuracy)
    assert metric.name == "accuracy"
    assert metric.args == {}
    assert isinstance(metric.func, tf.keras.metrics.Accuracy) 

 

@pytest.mark.parametrize("class_label,n_classes,expected",[
    (0,2,np.array([1.0,0.0])),
    (1,2,np.array([0.0,1.0])),
    (1,3,np.array([0.0,1.0, 0.0])),
    (1,5,np.array([0.0,1.0, 0.0, 0.0, 0.0])),
    ])
def test_to1hot(class_label, n_classes, expected):
    encoded =  NNInterface.to1hot(class_label, n_classes) 
    assert(encoded == expected).all()
    with pytest.raises(IndexError):
        assert NNInterface.to1hot(class_label=1.2, n_classes=2) 
    with pytest.raises(TypeError):
        assert NNInterface.to1hot(class_label=1, n_classes=2.0)



def test_transform_train_batch():
    inputs = np.random.rand(10,5,5)
    labels = np.array([1,0,0,0,1,0,0,1,1,1])

    X,Y = NNInterface.transform_train_batch(inputs, labels)

    
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

     output1 = NNInterface.transform_input(input1)
     output2 = NNInterface.transform_input(input2)

     assert output1.shape == (1,5,5,1)
     assert output2.shape == (1,5,5,1)

     assert np.array_equal(output1[0,:,:,0], input1)
     assert np.array_equal(output2[0,:,:,0], input2[0,:,:])


def test_transform_output():
    output1 = np.array([[0.05,0.05,0.7, 0.1, 0.1]])
    output2 = np.array([[0.05,0.05,0.7, 0.1, 0.1],[0.05,0.15,0.15,0.65,0.1]])

    class1, score1 = NNInterface.transform_output(output1)
    classes2, scores2 = NNInterface.transform_output(output2)

    assert np.array_equal(class1, np.array([2])) 
    assert np.array_equal(score1, np.array([0.7]))
    
    assert np.array_equal(classes2, np.array([2, 3])) 
    assert np.array_equal(scores2, np.array([0.7, 0.65]))
    


def test_optimizer_from_recipe(recipe_dict):
    built_opt = NNInterface.optimizer_from_recipe(recipe_dict['optimizer'])
    assert isinstance(built_opt, RecipeCompat)
    assert built_opt.name == 'Adam'
    assert built_opt.args == {'learning_rate':0.005}
    assert isinstance(built_opt.func, tf.keras.optimizers.Adam)
    
    
def test_loss_function_from_recipe(recipe_dict):
    built_loss = NNInterface.loss_function_from_recipe(recipe_dict['loss_function'])
    assert isinstance(built_loss, RecipeCompat)
    assert built_loss.name == 'FScoreLoss'
    assert built_loss.args == {}
    assert isinstance(built_loss.func, FScoreLoss)
    
def test_metrics_from_recipe(recipe_dict):
    built_metrics = NNInterface.metrics_from_recipe(recipe_dict['metrics'])
    assert isinstance(built_metrics[0], RecipeCompat)
    assert built_metrics[0].name == 'CategoricalAccuracy'
    assert built_metrics[0].args == {}
    assert isinstance(built_metrics[0].func, tf.keras.metrics.CategoricalAccuracy)
    
    
def test_optimizer_to_recipe():
    recipe_compat_opt = RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005)
    optimizer_dict = NNInterface.optimizer_to_recipe(recipe_compat_opt)

    assert optimizer_dict['name'] == 'Adam'
    assert optimizer_dict['parameters'] == {'learning_rate':0.005}


def test_loss_function_to_recipe():
    recipe_compat_loss = RecipeCompat('FScoreLoss', FScoreLoss, beta=0.5)
    loss_dict = NNInterface.loss_function_to_recipe(recipe_compat_loss)

    assert loss_dict['name'] == 'FScoreLoss'
    assert loss_dict['parameters'] == {'beta':0.5}

    
def test_metrics_to_recipe():
    recipe_compat_metric = [RecipeCompat('CategoricalAccuracy', tf.keras.metrics.CategoricalAccuracy)]
    metrics_dicts = NNInterface.metrics_to_recipe(recipe_compat_metric)

    assert isinstance(metrics_dicts, list)
    assert metrics_dicts[0]['name'] == 'CategoricalAccuracy'
    assert metrics_dicts[0]['parameters'] == {}


def test_read_recipe_file():
    path_to_file = os.path.join(path_to_assets, "recipes/basic_recipe.json")
    recipe = NNInterface.read_recipe_file(path_to_file)

    assert isinstance(recipe, dict)
    assert isinstance(recipe['optimizer'], RecipeCompat)
    assert isinstance(recipe['loss_function'], RecipeCompat)
    assert isinstance(recipe['metrics'], list)
    assert isinstance(recipe['metrics'][0], RecipeCompat)

    opt = recipe['optimizer']
    assert opt.name == 'Adam'
    assert isinstance(opt.func, tf.keras.optimizers.Adam)
    assert opt.args == {"learning_rate": 0.001, "beta_1": 0.9, "beta_2": 0.999, "decay": 0.01}

    loss = recipe['loss_function']
    assert loss.name == 'FScoreLoss'
    assert isinstance(loss.func, FScoreLoss)
    assert loss.args == {}

    metric = recipe['metrics'][0]
    assert metric.name == 'CategoricalAccuracy'
    assert isinstance(metric.func, tf.keras.metrics.CategoricalAccuracy)
    assert metric.args == {}


def test_write_recipe_file(recipe_dict):
    destination = os.path.join(path_to_tmp, "test_write_recipe_file.json")
    NNInterface.write_recipe_file(destination, recipe_dict)

    read_recipe =  NNInterface.read_recipe_file(destination, return_recipe_compat=False)
    #If return_recipe_compat is False, the result should be a recipe dictionary just like the recipe_dict used to write the file
    assert read_recipe == recipe_dict
    # assert read_recipe['optimizer'] == recipe_dict['optimizer']
    # assert read_recipe['loss_function'] == recipe_dict['loss_function']
    # assert read_recipe['metrics'] == recipe_dict['metrics']

    read_recipe =  NNInterface.read_recipe_file(destination, return_recipe_compat=True)
    #If return_recipe_compat is True, the result will be a dictionary with RecipeCompat objects for the optimizer, loss_functions and metrics (actually a lis of RecipCompat objects) 

    assert read_recipe['optimizer'].name == NNInterface.optimizer_from_recipe(recipe_dict['optimizer']).name
    assert read_recipe['optimizer'].func.__class__ == NNInterface.optimizer_from_recipe(recipe_dict['optimizer']).func.__class__
    assert read_recipe['optimizer'].args == NNInterface.optimizer_from_recipe(recipe_dict['optimizer']).args

    assert read_recipe['loss_function'].name == NNInterface.loss_function_from_recipe(recipe_dict['loss_function']).name
    assert read_recipe['loss_function'].func.__class__ == NNInterface.loss_function_from_recipe(recipe_dict['loss_function']).func.__class__
    assert read_recipe['loss_function'].args == NNInterface.loss_function_from_recipe(recipe_dict['loss_function']).args
    
    assert read_recipe['metrics'][0].name == NNInterface.metrics_from_recipe(recipe_dict['metrics'])[0].name
    assert read_recipe['metrics'][0].func.__class__ == NNInterface.metrics_from_recipe(recipe_dict['metrics'])[0].func.__class__
    assert read_recipe['metrics'][0].args == NNInterface.metrics_from_recipe(recipe_dict['metrics'])[0].args
    

    
def test_instantiate_nn(NNInterface_subclass):
    path_to_file = os.path.join(path_to_assets, "recipes/basic_recipe.json")
    recipe = NNInterface.read_recipe_file(path_to_file)
    
    NNInterface_subclass(activation='relu', n_neurons=64, optimizer=recipe['optimizer'],
                         loss_function=recipe['loss_function'], metrics=recipe['metrics'])    


