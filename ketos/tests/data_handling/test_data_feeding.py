""" Unit tests for the data feeding module within the ketos library

    
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



                >>> from tables import open_file
                >>> from ketos.data_handling.database_interface import open
                  
                >>> h5 = open_file("../tests/assets/15x_same_spec.h5", 'r') # create the database handle  
                >>> train_data = open(h5, "/train/species1")

                
                >>> train_generator = BatchGenerator(hdf5_table=train_data, batch_size=3, return_batch_ids=True) #create a batch generator 
                >>> n_epochs = 7    
                >>> for e in range(n_epochs):
                ...    ids, batch_X, batch_Y = next(train_generator)
                ...    print("epoch:{0} | instance ids:{1}, X batch shape: {2}, Y batch shape: {3}".format(e, ids, batch_X.shape, batch_Y.shape))
                epoch:0 | instance ids:[0, 1, 2], X batch shape: (3, 2413, 201), Y batch shape: (3,)
                epoch:1 | instance ids:[3, 4, 5], X batch shape: (3, 2413, 201), Y batch shape: (3,)
                epoch:2 | instance ids:[6, 7, 8], X batch shape: (3, 2413, 201), Y batch shape: (3,)
                epoch:3 | instance ids:[9, 10, 11], X batch shape: (3, 2413, 201), Y batch shape: (3,)
                epoch:4 | instance ids:[12, 13, 14], X batch shape: (3, 2413, 201), Y batch shape: (3,)
                epoch:5 | instance ids:[0, 1, 2], X batch shape: (3, 2413, 201), Y batch shape: (3,)
                epoch:6 | instance ids:[3, 4, 5], X batch shape: (3, 2413, 201), Y batch shape: (3,)

                >>> h5.close()