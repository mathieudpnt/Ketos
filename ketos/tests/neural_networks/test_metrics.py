import pytest
import numpy as np
import tensorflow as tf
from ketos.neural_networks.metrics import FScore, Accuracy, Precision, Recall


@pytest.mark.parametrize("beta, onehot, y_pred, y_true, expected", [(1.0, False, [1., 1., 1., 1.], [1., 1., 1., 1.], 1.0 ),
                                                                    (1.0, False, [1., 1., 1., 1.], [1., 0., 1., 0.], 0.66 ),
                                                                    (1.0, False, [1., 0., 0., 1.], [1., 0., 1., 0.], 0.5 ),

                                                                    (1.0, False, [1., 1., 1., 1.], [1., 0., 1., 0.], 0.66 ),
                                                                    (2.0, False, [1., 1., 1., 1.], [1., 0., 1., 0.], 0.83 ),
                                                                    (0.5, False, [1., 1., 1., 1.], [1., 0., 1., 0.], 0.55 ),
                                                                    
                                                                    (1.0, True, np.array([[0.3, 0.7],[0.1,0.9]]), np.array([[0., 1.],[0., 1.]]), 1.0 ),
                                                                    (1.0, True, np.array([[0.3, 0.7],[0.1,0.9]]), np.array([[1., 0.],[1., 0.]]), 0.0 ),])
def test_fscore(beta,onehot, y_pred, y_true, expected):
    metric_function = FScore(beta=beta, onehot=onehot)
    metric_value = metric_function(y_pred=y_pred, y_true=y_true)
    #loss = loss.numpy()

    assert metric_value == pytest.approx(expected, rel=1e-2, abs=1e-2)



@pytest.mark.parametrize("onehot, y_pred, y_true, expected", [(False, [1., 1., 1., 1.], [1., 1., 1., 1.], 1.0 ),
                                                            (False, [1., 1., 1., 1.], [1., 0., 1., 0.], 0.5 ),
                                                            (False, [1., 0., 0., 1.], [1., 0., 1., 0.], 0.5 ),
                                                                
                                                            (True, np.array([[0.3, 0.7],[0.1,0.9]]), np.array([[0., 1.],[0., 1.]]), 1.0 ),
                                                            (True, np.array([[0.3, 0.7],[0.1,0.9]]), np.array([[1., 0.],[1., 0.]]), 0.0 ),
                                                            (True, np.array([[0.3, 0.7],[0.1,0.9]]), np.array([[0., 1.],[1., 0.]]), 0.5 ),])
def test_precision(onehot, y_pred, y_true, expected):
    metric_function = Precision(onehot=onehot)
    metric_value = metric_function(y_pred=y_pred, y_true=y_true)
    
    assert metric_value == pytest.approx(expected, rel=1e-2, abs=1e-2)



@pytest.mark.parametrize("onehot, y_pred, y_true, expected", [(False, [1., 1., 1., 1.], [1., 1., 1., 1.], 1.0 ),
                                                            (False, [1., 1., 1., 1.], [1., 0., 1., 0.], 1.0 ),
                                                            (False, [1., 0., 0., 1.], [1., 0., 1., 0.], 0.5 ),
                                                                
                                                            (True, np.array([[0.3, 0.7],[0.1,0.9]]), np.array([[0., 1.],[0., 1.]]), 1.0 ),
                                                            (True, np.array([[0.3, 0.7],[0.1,0.9]]), np.array([[1., 0.],[1., 0.]]), 0.0 ),
                                                            (True, np.array([[0.3, 0.7],[0.1,0.9]]), np.array([[0., 1.],[1., 0.]]), 1.0 ),])
def test_recall(onehot, y_pred, y_true, expected):
    metric_function = Recall(onehot=onehot)
    metric_value = metric_function(y_pred=y_pred, y_true=y_true)
    
    assert metric_value == pytest.approx(expected, rel=1e-2, abs=1e-2)


@pytest.mark.parametrize("onehot, y_pred, y_true, expected", [(False, [1., 1., 1., 1.], [1., 1., 1., 1.], 1.0 ),
                                                            (False, [1., 1., 1., 1.], [1., 0., 1., 0.], 0.5 ),
                                                            (False, [1., 0., 0., 1.], [1., 0., 1., 0.], 0.5 ),
                                                                
                                                            (True, np.array([[0.3, 0.7],[0.1,0.9]]), np.array([[0., 1.],[0., 1.]]), 1.0 ),
                                                            (True, np.array([[0.3, 0.7],[0.1,0.9]]), np.array([[1., 0.],[1., 0.]]), 0.0 ),
                                                            (True, np.array([[0.3, 0.7],[0.1,0.9]]), np.array([[0., 1.],[1., 0.]]), 0.5 ),])
def test_accuracy(onehot, y_pred, y_true, expected):
    metric_function = Accuracy(onehot=onehot)
    metric_value = metric_function(y_pred=y_pred, y_true=y_true)
    
    assert metric_value == pytest.approx(expected, rel=1e-2, abs=1e-2)
