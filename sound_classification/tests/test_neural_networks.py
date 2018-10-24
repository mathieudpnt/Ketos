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
from sound_classification.neural_networks import MNet, DataUse, remove_correct_predictions, get_class_confidences, get_correct_class_confidences
from tensorflow import reset_default_graph


@pytest.mark.test_MNet
def test_initialize_MNet(database_prepared_for_NN_2_classes):
    d = database_prepared_for_NN_2_classes
    x = d["train_x"]
    y = d["train_y"]
    _ = MNet(train_x=x, train_y=y, verbosity=0)

@pytest.mark.test_MNet
def test_set_data_to_MNet(database_prepared_for_NN_2_classes):
    d = database_prepared_for_NN_2_classes
    x = d["train_x"]
    y = d["train_y"]
    network = MNet(train_x=x, train_y=y, verbosity=0)
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
    network = MNet(train_x=x, train_y=y, verbosity=0)
    # add training data
    network.add_training_data(x=x, y=y)
    assert 2 * x.shape[0] == network.images[DataUse.TRAINING].shape[0]
    assert x.shape[1:] == network.images[DataUse.TRAINING].shape[1:]
    assert 2 * y.shape[0] == network.labels[DataUse.TRAINING].shape[0]
    assert y.shape[1:] == network.labels[DataUse.TRAINING].shape[1:]

@pytest.mark.test_remove_correct_predictions
def test_remove_correct_predictions(data_classified_by_nn):
    x,y,w = data_classified_by_nn
    x_trim, y_trim = remove_correct_predictions(x=x, y=y, class_weights=w, confidence_cut=0)
    assert len(x_trim) == 2
    assert x_trim[0] == 5
    assert x_trim[1] == 6
    x_trim, y_trim = remove_correct_predictions(x=x, y=y, class_weights=w, confidence_cut=0.5)
    assert len(x_trim) == 3
    assert x_trim[0] == 4
    assert x_trim[1] == 5
    assert x_trim[2] == 6
    
@pytest.mark.test_get_class_confidences
def test_get_class_confidences(data_classified_by_nn):
    x,y,w = data_classified_by_nn
    conf = get_class_confidences(class_weights=w)
    assert len(conf) == 6
    assert conf[0] == pytest.approx(0.6, abs=0.001)
    assert conf[1] == pytest.approx(0.8, abs=0.001)
    
@pytest.mark.test_get_correct_class_confidences
def test_get_correct_class_confidences(data_classified_by_nn):
    x,y,w = data_classified_by_nn
    conf = get_correct_class_confidences(class_weights=w, y=y)
    assert len(conf) == 4
    assert conf[0] == pytest.approx(0.6, abs=0.001)
    assert conf[1] == pytest.approx(0.8, abs=0.001)
