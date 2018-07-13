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


@pytest.mark.test_CNNWhale
def test_initialize_CNNWhale_with_default_args(database_prepared_for_NN):
    network = nn.CNNWhale.from_prepared_data(database_prepared_for_NN, batch_size=1, num_channels=1, num_labels=1)

