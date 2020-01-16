import pytest
import numpy as np
import tensorflow as tf
from ketos.neural_networks.nn_interface import RecipeCompat, NNInterface


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



    