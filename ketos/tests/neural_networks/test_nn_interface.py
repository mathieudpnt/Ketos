import pytest
import numpy as np
import tensorflow as tf
from ketos.neural_networks.nn_interface import RecipeCompat, NNInterface
from ketos.neural_networks.losses import FScoreLoss


@pytest.fixture
def recipe_dict():
    recipe = {'optimizer': {'name':'Adam', 'parameters': {'learning_rate':0.005}},
               'loss_function': {'name':'FScoreLoss', 'parameters':{}},  
               'metrics': [{'name':'CategoricalAccuracy', 'parameters':{}}]
        
    }
    return recipe


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

    assert type(metrics_dicts) == list
    assert metrics_dicts[0]['name'] == 'CategoricalAccuracy'
    assert metrics_dicts[0]['parameters'] == {}
