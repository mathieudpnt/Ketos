""" Unit tests for the 'cnn' module within the ketos library

    Authors: Fabio Frazao and Oliver Kirsebom
    Contact: fsfrazao@dal.ca, oliver.kirsebom@dal.ca
    Organization: MERIDIAN (https://meridian.cs.dal.ca/)
    Team: Acoustic data analytics, Institute for Big Data Analytics, Dalhousie University
    Project: ketos
             Project goal: The ketos library provides functionalities for handling data, processing audio signals and
             creating deep neural networks for sound detection and classification projects.
     
    License: GNU GPLv3

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

import pytest
import numpy as np
import ketos.data_handling.data_handling as dh
from ketos.neural_networks.cnn import CNNWhale
from tensorflow import reset_default_graph


@pytest.mark.test_CNNWhale
def test_initialize_CNNWhale_with_default_constructor_and_default_args(database_prepared_for_NN):
    d = database_prepared_for_NN
    train_x = d["train_x"]
    train_y = d["train_y"]
    _ = CNNWhale(train_x=train_x, train_y=train_y, verbosity=0)
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
    network = CNNWhale(train_x=train_x, train_y=train_y, validation_x=validation_x, validation_y=validation_y, test_x=test_x, test_y=test_y, num_labels=2, verbosity=0)
    _ = network.create()
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
    network = CNNWhale(train_x=train_x, train_y=train_y, validation_x=validation_x, validation_y=validation_y, test_x=test_x, test_y=test_y, num_labels=2, verbosity=0)
    _ = network.create()
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
    network = CNNWhale(train_x=train_x, train_y=train_y, validation_x=validation_x, validation_y=validation_y, test_x=test_x, test_y=test_y, num_labels=2, verbosity=0)
    path_to_meta, path_to_saved_model, test_acc = trained_CNNWhale
    _ = network.load(path_to_meta, path_to_saved_model)
    assert test_acc == network.accuracy_on_test()
    reset_default_graph()
    
@pytest.mark.test_CNNWhale
def test_compute_class_weights_with_CNNWhale(database_prepared_for_NN_2_classes):
    d = database_prepared_for_NN_2_classes
    x = d["train_x"]
    y = d["train_y"]
    network = CNNWhale(train_x=x, train_y=y, num_labels=2, verbosity=0, seed=41)
    _ = network.create()
    network.train()
    img = np.zeros((20, 20))
    result = network.get_class_weights(x=[img])
    weights = result[0]
    assert weights[0] + weights[1] == pytest.approx(1.000, abs=0.001)
    reset_default_graph()

@pytest.mark.test_CNNWhale
def test_compute_features_with_CNNWhale(database_prepared_for_NN_2_classes):
    d = database_prepared_for_NN_2_classes
    x = d["train_x"]
    y = d["train_y"]
    network = CNNWhale(train_x=x, train_y=y,  num_labels=2, verbosity=0, seed=41)
    _ = network.create()
    network.train()
    img = np.zeros((20, 20))
    result = network.get_features(x=[img], layer_name='dense_1')
    f = result[0]
    assert f.shape == (512,)
    reset_default_graph()
