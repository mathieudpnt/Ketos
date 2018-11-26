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
from tensorflow import reset_default_graph


@pytest.mark.test_EDTCN
def test_initialize_EDTCN_with_default_constructor_and_default_args(database_prepared_for_NN):
    d = database_prepared_for_NN
    x = d["train_x"]
    y = d["train_y"]
    _ = EDTCN(train_x=x, train_y=y, verbosity=0)
    reset_default_graph()