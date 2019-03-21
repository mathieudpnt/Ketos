""" Unit tests for the data_feeding module within the ketos library

    
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

import os
import pytest
import numpy as np
import pandas as pd
from tables import open_file
from ketos.data_handling.database_interface import open_table
from ketos.data_handling.data_feeding import ActiveLearningBatchGenerator, BatchGenerator
from ketos.neural_networks.neural_networks import class_confidences, predictions

current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')


@pytest.mark.test_ActiveLearningBatchGenerator
def test_get_samples(data_classified_by_nn):
    x, y, w = data_classified_by_nn
    p = predictions(w)
    c = class_confidences(w)

    sampler = ActiveLearningBatchGenerator(x=x, y=y, randomize=False, max_keep=0.5, conf_cut=0.5, seed=1, equal_rep=False)

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



@pytest.mark.test_ActiveLearningBatchGenerator
def test_get_samples_randomize(data_classified_by_nn):
    x, y, w = data_classified_by_nn
    p = predictions(w)
    c = class_confidences(w)

    
    sampler = ActiveLearningBatchGenerator(x=x, y=y, randomize=True, max_keep=0.5, conf_cut=0.5, seed=1, equal_rep=False)

    x1, y1, _ = sampler.get_samples(num_samples=2) #3,2
    assert np.all(x1 == np.array([3,2]))    
    sampler.update_prediction_confidence(pred=p[[3,2]], conf=c[[3,2]])

    x1, y1, _ = sampler.get_samples(num_samples=2) #1,2
    assert np.all(x1 == np.array([1,2]))    
    sampler.update_prediction_confidence(pred=p[[3,2]], conf=c[[3,2]])

    x1, y1, _ = sampler.get_samples(num_samples=2) #2,5 (keeps one from previous iteration)
    assert np.all(x1 ==np.array([2,5]))    
    sampler.update_prediction_confidence(pred=[y1[0],p[4]], conf=[1.,c[4]])  # pretend network has learned #2

    x1, y1, _ = sampler.get_samples(num_samples=2) #4,5 (keeps one from previous iteration)
    assert np.all(x1 ==np.array([4,5]))    
    sampler.update_prediction_confidence(pred=[y1[0],p[5]], conf=[1.,c[5]])  # pretend network has learned #4

    
@pytest.mark.test_ActiveLearningBatchGenerator
def test_get_samples_equal_rep(data_classified_by_nn):
    x, y, w = data_classified_by_nn
    p = predictions(w)
    c = class_confidences(w)


    #Without equal representation: 1 positive, 3 negatives
    sampler = ActiveLearningBatchGenerator(x=x, y=y, randomize=True, max_keep=0.5, conf_cut=0.5, seed=1, equal_rep=False)
    _, y1, _ = sampler.get_samples(num_samples=4) #3,2
    assert np.all(y1 == np.array([0,1,0,0]))

    #With equal representation: 2 positives, 2 negatives
    sampler = ActiveLearningBatchGenerator(x=x, y=y, randomize=True, max_keep=0.5, conf_cut=0.5, seed=1, equal_rep=True)
    _, y1, _ = sampler.get_samples(num_samples=4) #3,2
    assert np.all(y1 == np.array([1,1,0,0]))


@pytest.mark.test_BatchGenerator
def test_one_batch():
    """ Test if one batch has the expected shape and contents
    """
    h5 = open_file(os.path.join(path_to_assets, "15x_same_spec.h5"), 'r') # create the database handle  
    train_data = open_table(h5, "/train/species1")

    five_specs = train_data[:5]['data']
    five_boxes = train_data[:5]['boxes']

    train_generator = BatchGenerator(hdf5_table=train_data, batch_size=5, return_batch_ids=True) #create a batch generator 
    ids, X, Y = next(train_generator)
    assert ids == [0,1,2,3,4]
    assert X.shape == (5, 2413, 201)
    np.testing.assert_array_equal(X, five_specs)
    assert Y.shape == (5,)
    np.testing.assert_array_equal(Y, five_boxes)

    h5.close()


@pytest.mark.test_BatchGenerator
def test_labels_as_Y():
    """ Test if batch generator returns labels instead of boxes
    """
    h5 = open_file(os.path.join(path_to_assets, "15x_same_spec.h5"), 'r') # create the database handle  
    train_data = open_table(h5, "/train/species1")
    
    five_labels = train_data[:5]['labels']

    train_generator = BatchGenerator(hdf5_table=train_data, y_field='labels', batch_size=5, return_batch_ids=False) #create a batch generator 
    _, Y = next(train_generator)
    np.testing.assert_array_equal(Y, five_labels)

    h5.close()
    


@pytest.mark.test_BatchGenerator
def test_batch_sequence_same_as_db():
    """ Test if batches are generated with instances in the same order as they appear in the database
    """
    h5 = open_file(os.path.join(path_to_assets, "15x_same_spec.h5"), 'r') #create the database handle  
    train_data = open_table(h5, "/train/species1")


    ids_in_db = train_data[:]['id']
    train_generator = BatchGenerator(hdf5_table=train_data, x_field='id', batch_size=3, return_batch_ids=True) #create a batch generator 

    for i in range(3):
        ids, X, _ = next(train_generator)
        np.testing.assert_array_equal(X, ids_in_db[i*3: i*3+3])
        assert ids == list(range(i*3, i*3+3))
    
    h5.close()


@pytest.mark.test_BatchGenerator
def test_last_batch():
    """ Test if last batch has the expected number of instances
    """
    h5 = open_file(os.path.join(path_to_assets, "15x_same_spec.h5"), 'r') #create the database handle  
    train_data = open_table(h5, "/train/species1")


    ids_in_db = train_data[:]['id']
    train_generator = BatchGenerator(hdf5_table=train_data, batch_size=6, return_batch_ids=True) #create a batch generator 
    #First batch
    ids, X, _ = next(train_generator)
    assert ids == [0,1,2,3,4,5]
    assert X.shape == (6, 2413, 201)
    #Second batch
    ids, X, _ = next(train_generator)
    assert ids == [6,7,8,9,10,11]
    assert X.shape == (6, 2413, 201)
    #last batch
    ids, X, _ = next(train_generator)
    assert ids == [12,13,14]
    assert X.shape == (3, 2413, 201)

    h5.close()

@pytest.mark.test_BatchGenerator
def test_multiple_epochs():
    """ Test if batches are as expected after the first epoch
    """
    h5 = open_file(os.path.join(path_to_assets, "15x_same_spec.h5"), 'r') #create the database handle  
    train_data = open_table(h5, "/train/species1")


    ids_in_db = train_data[:]['id']
    train_generator = BatchGenerator(hdf5_table=train_data, batch_size=6, return_batch_ids=True) #create a batch generator 
    #Epoch 0, batch 0
    ids, X, _ = next(train_generator)
    assert ids == [0,1,2,3,4,5]
    assert X.shape == (6, 2413, 201)
    #Epoch 0, batch 1
    ids, X, _ = next(train_generator)
    assert ids == [6,7,8,9,10,11]
    assert X.shape == (6, 2413, 201)
    #Epoch 0 batch2
    ids, X, _ = next(train_generator)
    assert ids == [12,13,14]
    assert X.shape == (3, 2413, 201)

    #Epoch 1, batch 0
    ids, X, _ = next(train_generator)
    assert ids == [0,1,2,3,4,5]
    assert X.shape == (6, 2413, 201)

    h5.close()

@pytest.mark.test_BatchGenerator
def test_shuffle():
    """Test shuffle argument.
        Instances should be shuffled before divided into batches, but the order should be consistent across epochs if
        'refresh_on_epoch_end' is False.
    """
    h5 = open_file(os.path.join(path_to_assets, "15x_same_spec.h5"), 'r') #create the database handle  
    train_data = open_table(h5, "/train/species1")


    ids_in_db = train_data[:]['id']
    np.random.seed(100)
    
    train_generator = BatchGenerator(hdf5_table=train_data, batch_size=6, return_batch_ids=True, shuffle=True) #create a batch generator 
    

    for epoch in range(5):
        #batch 0
        ids, X, _ = next(train_generator)
        assert ids == [9, 1, 12, 13, 6, 10]
        assert X.shape == (6, 2413, 201)
        #batch 1
        ids, X, _ = next(train_generator)
        assert ids == [5, 2, 4, 0, 11, 7]
        assert X.shape == (6, 2413, 201)
        #batch 2
        ids, X, _ = next(train_generator)
        assert ids ==  [3, 14, 8]
        assert X.shape == (3, 2413, 201)
        
    
    h5.close()


@pytest.mark.test_BatchGenerator
def test_refresh_on_epoch_end():
    """ Test if batches are generated with randomly selected instances for each epoch
    """
    h5 = open_file(os.path.join(path_to_assets, "15x_same_spec.h5"), 'r') #create the database handle  
    train_data = open_table(h5, "/train/species1")


    ids_in_db = train_data[:]['id']
    np.random.seed(100)
    
    train_generator = BatchGenerator(hdf5_table=train_data, batch_size=6, return_batch_ids=True, shuffle=True, refresh_on_epoch_end=True) #create a batch generator 

    expected_ids = {'epoch_1':([9, 1, 12, 13, 6, 10],[5, 2, 4, 0, 11, 7],[3, 14, 8]),
                     'epoch_2': ( [9, 7, 1, 13, 5, 12],[3, 2, 6, 14, 10, 11],[4, 8, 0]),    
                     'epoch_3': ([11, 6, 2, 0, 10, 14],[8, 9, 1, 7, 13, 12],[4, 3, 5])}
                     

    for epoch in ['epoch_1', 'epoch_2', 'epoch_3']:
        #batch 0
        ids, X, _ = next(train_generator)
        print(epoch)
        assert ids == expected_ids[epoch][0]
        #batch 1
        ids, X, _ = next(train_generator)
        assert ids == expected_ids[epoch][1]
        #batch 2
        ids, X, _ = next(train_generator)
        assert ids == expected_ids[epoch][2]
    
    h5.close()

@pytest.mark.test_BatchGenerator
def test_instance_function():
    """ Test if the function passed as 'instance_function' is applied to the batch
    """
    h5 = open_file(os.path.join(path_to_assets, "15x_same_spec.h5"), 'r') # create the database handle  
    train_data = open_table(h5, "/train/species1")

    def apply_to_batch(X,Y):
        X = np.mean(X, axis=(1,2))
        return (X, Y)

    train_generator = BatchGenerator(hdf5_table=train_data, batch_size=5, return_batch_ids=True, instance_function=apply_to_batch) #create a batch generator 
    ids, X, Y = next(train_generator)
    assert X.shape == (5,)
    assert X[0] == pytest.approx(7694.1147, 0.1)
    assert Y.shape == (5,)
    

    h5.close()
