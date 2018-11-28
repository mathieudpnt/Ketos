""" Unit tests for the the 'EDTCN' module in the 'sound_classification' package


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
from sound_classification.edtcn import EDTCN


@pytest.mark.test_EDTCN
def test_initialize_EDTCN_with_default_constructor_and_default_args(data_for_TCN):
    train_x, train_y, val_x, val_y, test_x, test_y = data_for_TCN
    _ = EDTCN(train_x=train_x, train_y=train_y, validation_x=val_x, validation_y=val_y, test_x=test_x, test_y=test_y)

def test_create_EDTCN_network_with_default_args(data_for_TCN):
    train_x, train_y, val_x, val_y, test_x, test_y = data_for_TCN
    net = EDTCN(train_x=train_x, train_y=train_y, validation_x=val_x, validation_y=val_y, test_x=test_x, test_y=test_y)
    net.create()

def test_train_EDTCN_network_with_default_args(data_for_TCN):
    train_x, train_y, val_x, val_y, test_x, test_y = data_for_TCN
    net = EDTCN(train_x=train_x, train_y=train_y, validation_x=val_x, validation_y=val_y, test_x=test_x, test_y=test_y)
    net.create()
    net.train()

def test_create_EDTCN_network_with_max_len_not_divisible_by_four(data_for_TCN):
    train_x, train_y, val_x, val_y, test_x, test_y = data_for_TCN
    net = EDTCN(train_x=train_x, train_y=train_y, validation_x=val_x, validation_y=val_y, test_x=test_x, test_y=test_y)
    net.create(max_len=15)
    assert net.max_len == 12

def test_predict_labels_with_default_EDTCN_network(data_for_TCN):
    train_x, train_y, val_x, val_y, test_x, test_y = data_for_TCN
    net = EDTCN(train_x=train_x, train_y=train_y, validation_x=val_x, validation_y=val_y, test_x=test_x, test_y=test_y)
    net.create()
    net.train()
    N = 2
    p = net.get_predictions(x=train_x[0:N])
    assert len(p) == N
    assert p[0] == train_y[0]
    assert p[1] == train_y[1]

