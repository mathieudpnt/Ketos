import pytest
import numpy as np
import tensorflow as tf
from ketos.neural_networks.losses import FScoreLoss


# current_dir = os.path.dirname(os.path.realpath(__file__))
# path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
# path_to_tmp = os.path.join(path_to_assets,'tmp')

@pytest.mark.parametrize("beta, y_pred, y_true, expected", [(1.0, [0., 1.], [0., 1.], 0.0 ),
                                                           (1.0, [1., 0.], [1., 0.], 0.0 ),
                                                           (1.0, [1., 0.], [0., 1.], 1.0 ),
                                                           (1.0, [0., 1.], [1., 0.], 1.0 ),
                                                           (1.0, [1., 1., 1., 1.], [1., 0., 1., 0.], 0.333333 ),
                                                           (2.0, [1., 1., 1., 1.], [1., 0., 1., 0.], 0.166666 ),
                                                           (0.5, [1., 1., 1., 1.], [1., 0., 1., 0.], 0.44444 ),
                                                           (1.0, [0.3, 0.7], [0., 1.], 0.3 ),
                                                           (1.0, np.array([[0.3, 0.7],[0.1,0.9]]), np.array([[0., 1.],[0., 1.]]), 0.2 ),
                                                           (1.0, np.array([[0.3, 0.7],[0.1,0.9]]), np.array([[1., 0.],[1., 0.]]), 0.8 ),])
def test_FscoreLoss(beta, y_pred, y_true, expected):
    loss_function = FScoreLoss(beta=beta)
    loss = loss_function(y_pred=y_pred, y_true=y_true)
    loss = loss.numpy()
    print(loss)

    assert loss == pytest.approx(expected, rel=1e-5, abs=1e-5)
