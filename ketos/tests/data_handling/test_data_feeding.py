""" Unit tests for the the 'training_data_provider' module in the 'sound_classification' package

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
from ketos.data_handling.data_feeding import TrainingDataProvider
from ketos.neural_networks.neural_networks import class_confidences, predictions


@pytest.mark.test_TrainingDataSampler
def test_get_samples(data_classified_by_nn):
    x, y, w = data_classified_by_nn
    p = predictions(w)
    c = class_confidences(w)

    sampler = TrainingDataProvider(x=x, y=y, randomize=False, max_keep=0.5, conf_cut=0.5, seed=1, equal_rep=False)

    x1, y1, _ = sampler.get_samples(num_samples=2) #0,1
    assert np.all(x1 == x[0:2])    
    sampler.update_prediction_confidence(pred=p[:2], conf=c[:2])

    x1, y1, _ = sampler.get_samples(num_samples=2) #2,3
    assert np.all(x1 == x[2:4])    
    sampler.update_prediction_confidence(pred=p[2:4], conf=c[2:4])

    x1, y1, _ = sampler.get_samples(num_samples=2) #3,4 (keeps one from previous iteration)
    assert np.all(x1 == x[3:5])    
    sampler.update_prediction_confidence(pred=[y1[0],p[4]], conf=[1.,c[4]])  # pretend network has learned #3

    x1, y1, _ = sampler.get_samples(num_samples=2) #4,5 (keeps one from previous iteration)
    assert np.all(x1 == x[4:6])    
    sampler.update_prediction_confidence(pred=[y1[0],p[5]], conf=[1.,c[5]])  # pretend network has learned #4

    x1, y1, _ = sampler.get_samples(num_samples=2) #0,5 (keeps one from previous iteration)
    assert np.all(x1 == [x[0],x[5]])    
    sampler.update_prediction_confidence(pred=[p[0],y1[1]], conf=[c[0],1.])  # pretend network has learned #5

    x1, y1, _ = sampler.get_samples(num_samples=2) #1,2
    assert np.all(x1 == [x[1],x[2]])    
    sampler.update_prediction_confidence(pred=p[1:3], conf=c[1:3])
