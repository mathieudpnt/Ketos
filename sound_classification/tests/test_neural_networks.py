""" Unit tests for the the 'data_handling' module in the 'sound_classification' package


    Authors: Fabio Frazao and Oliver Kirsebom
    contact: fsfrazao@dal.ca and oliver.kirsebom@dal.ca
    Organization: MERIDIAN-Intitute for Big Data Analytics
    Team: Acoustic data Analytics, Dalhousie University
    Project: packages/sound_classification
             Project goal: Package code internally used in projects applying Deep Learning to sound classification
     
    License:

"""

import pytest
import numpy as np
import pandas as pd
import sound_classification.data_handling as dh
from sound_classification.neural_networks import MNet, DataUse
from tensorflow import reset_default_graph


@pytest.mark.test_MNet
def test_initialize_MNet(database_prepared_for_NN_2_classes):
    d = database_prepared_for_NN_2_classes
    x = d["train_x"]
    y = d["train_y"]
    _ = MNet(train_x=x, train_y=y)

@pytest.mark.test_MNet
def test_set_data_to_MNet(database_prepared_for_NN_2_classes):
    d = database_prepared_for_NN_2_classes
    x = d["train_x"]
    y = d["train_y"]
    network = MNet(train_x=x, train_y=y)
    # set training data
    network.set_training_data(x=x, y=y)
    assert np.all(x == network.images[DataUse.TRAINING])
    assert np.all(y == network.labels[DataUse.TRAINING])
    # set validation data
    network.set_validation_data(x=x, y=y)
    assert np.all(x == network.images[DataUse.VALIDATION])
    assert np.all(y == network.labels[DataUse.VALIDATION])
    # set test data
    network.set_test_data(x=x, y=y)
    assert np.all(x == network.images[DataUse.TEST])
    assert np.all(y == network.labels[DataUse.TEST])    

@pytest.mark.test_MNet
def test_add_data_to_MNet(database_prepared_for_NN_2_classes):
    d = database_prepared_for_NN_2_classes
    x = d["train_x"]
    y = d["train_y"]
    network = MNet(train_x=x, train_y=y)
    # add training data
    network.add_training_data(x=x, y=y)
    assert 2 * x.shape[0] == network.images[DataUse.TRAINING].shape[0]
    assert x.shape[1:] == network.images[DataUse.TRAINING].shape[1:]
    assert 2 * y.shape[0] == network.labels[DataUse.TRAINING].shape[0]
    assert y.shape[1:] == network.labels[DataUse.TRAINING].shape[1:]
