""" Unit tests for the the 'training_data_sampler' module in the 'sound_classification' package

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
from sound_classification.training_data_sampler import TrainingDataSampler


@pytest.mark.test_TrainingDataSampler
def test_get_samples(data_classified_by_nn):
    x, y, w = data_classified_by_nn
    sampler = TrainingDataSampler(x=x, y=y, randomize=False)
    x1, y1 = sampler.get_samples(num_samples=2)
    print('\n\n  xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  ', x1, y1)