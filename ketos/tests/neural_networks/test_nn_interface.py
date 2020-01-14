import pytest
import numpy as np
from ketos.neural_networks.nn_interface import RecipeCompat, NNInterface


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

