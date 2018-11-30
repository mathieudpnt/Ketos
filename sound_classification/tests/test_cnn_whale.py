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
from sound_classification.cnn_whale import CNNWhale
from tensorflow import reset_default_graph


@pytest.mark.test_CNNWhale
def test_initialize_CNNWhale_with_class_method_and_default_args(database_prepared_for_NN):
    d = database_prepared_for_NN
    _ = CNNWhale.from_prepared_data(d, verbosity=0)

@pytest.mark.test_CNNWhale
def test_initialize_CNNWhale_with_default_constructor_and_default_args(database_prepared_for_NN):
    d = database_prepared_for_NN
    train_x = d["train_x"]
    train_y = d["train_y"]
    _ = CNNWhale(train_x, train_y, verbosity=0)
    reset_default_graph()

@pytest.mark.test_CNNWhale
def test_train_CNNWhale_with_default_args(database_prepared_for_NN):
    d = database_prepared_for_NN
    train_x = d["train_x"]
    train_y = d["train_y"]
    validation_x = d["validation_x"]
    validation_y = d["validation_y"]
    test_x = d["test_x"]
    test_y = d["test_y"]
    network = CNNWhale(train_x, train_y, validation_x, validation_y, test_x, test_y, num_labels=1, verbosity=0)
    network.create()
    network.train()
    reset_default_graph()

@pytest.mark.test_CNNWhale
def test_train_CNNWhale_with_default_args2(database_prepared_for_NN_2_classes):
    d = database_prepared_for_NN_2_classes
    train_x = d["train_x"]
    train_y = d["train_y"]
    validation_x = d["validation_x"]
    validation_y = d["validation_y"]
    test_x = d["test_x"]
    test_y = d["test_y"]
    network = CNNWhale(train_x, train_y, validation_x, validation_y, test_x, test_y, num_labels=2, verbosity=0)
    network.create()
    network.train()
    reset_default_graph()

@pytest.mark.test_CNNWhale
def test_load_CNNWhale_model(database_prepared_for_NN_2_classes, trained_CNNWhale):
    d = database_prepared_for_NN_2_classes
    train_x = d["train_x"]
    train_y = d["train_y"]
    validation_x = d["validation_x"]
    validation_y = d["validation_y"]
    test_x = d["test_x"]
    test_y = d["test_y"]
    network = CNNWhale(train_x, train_y, validation_x, validation_y,
                          test_x, test_y, num_labels=2, verbosity=0)
    path_to_meta, path_to_saved_model, test_acc = trained_CNNWhale
    network.load(path_to_meta, path_to_saved_model)
    assert test_acc == network.accuracy_on_test()
    reset_default_graph()
    
@pytest.mark.test_CNNWhale
def test_compute_class_weights_with_CNNWhale(database_prepared_for_NN_2_classes):
    d = database_prepared_for_NN_2_classes
    x = d["train_x"]
    y = d["train_y"]
    network = CNNWhale(train_x=x, train_y=y,  num_labels=2, verbosity=0, seed=41)
    network.create()
    network.train()
    img = np.zeros((20, 20))
    result = network.get_class_weights(x=[img])
    weights = result[0]
    assert weights[0] == pytest.approx(0.5, abs=0.1)
    assert weights[1] == pytest.approx(0.5, abs=0.1)
    