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
import sound_classification.neural_networks as nn
from tensorflow import reset_default_graph


@pytest.mark.test_CNNWhale
def test_initialize_CNNWhale_with_class_method_and_default_args(database_prepared_for_NN):
    d = database_prepared_for_NN
    network = nn.CNNWhale.from_prepared_data(d, batch_size=1, num_channels=1, num_labels=1)

@pytest.mark.test_CNNWhale
def test_initialize_CNNWhale_with_default_constructor_and_default_args(database_prepared_for_NN):
    d = database_prepared_for_NN
    train_x = d["train_x"]
    train_y = d["train_y"]
    validation_x = d["validation_x"]
    validation_y = d["validation_y"]
    test_x = d["test_x"]
    test_y = d["test_y"]
    network = nn.CNNWhale(train_x, train_y, validation_x, validation_y, test_x, test_y, batch_size=1, num_channels=1, num_labels=1)
    reset_default_graph()
    

@pytest.mark.test_CNNWhale
def test_train_neural_net_with_default_args(database_prepared_for_NN):
    d = database_prepared_for_NN
    train_x = d["train_x"]
    train_y = d["train_y"]
    validation_x = d["validation_x"]
    validation_y = d["validation_y"]
    test_x = d["test_x"]
    test_y = d["test_y"]
    network = nn.CNNWhale(train_x, train_y, validation_x, validation_y, test_x, test_y, batch_size=1, num_channels=2, num_labels=1)
    tf_nodes = network.create_net_structure()
    network.set_tf_nodes(tf_nodes)
    network.train()
    reset_default_graph()


@pytest.mark.test_CNNWhale
def test_train_neural_net_with_default_args2(database_prepared_for_NN_2_classes):
    d = database_prepared_for_NN_2_classes
    train_x = d["train_x"]
    train_y = d["train_y"]
    validation_x = d["validation_x"]
    validation_y = d["validation_y"]
    test_x = d["test_x"]
    test_y = d["test_y"]

    network = nn.CNNWhale(train_x, train_y, validation_x, validation_y, test_x, test_y, batch_size=1, num_channels=2, num_labels=2)
    tf_nodes = network.create_net_structure()
    network.set_tf_nodes(tf_nodes)
    network.train()
    reset_default_graph()

@pytest.mark.test_CNNWhale
def test_load_model(database_prepared_for_NN_2_classes, trained_CNNWhale):
    d = database_prepared_for_NN_2_classes
    

    train_x = d["train_x"]
    train_y = d["train_y"]
    validation_x = d["validation_x"]
    validation_y = d["validation_y"]
    test_x = d["test_x"]
    test_y = d["test_y"]
    network = nn.CNNWhale(train_x, train_y, validation_x, validation_y,
                          test_x, test_y, batch_size=1, num_channels=2, num_labels=2)
    
    

    path_to_meta, path_to_saved_model, test_acc = trained_CNNWhale
    tf_nodes = network.load_net_structure(path_to_meta, path_to_saved_model)
    network.set_tf_nodes(tf_nodes)

    assert test_acc == network.accuracy_on_test()
    reset_default_graph()
    

