# ================================================================================ #
#   Authors: Fabio Frazao and Oliver Kirsebom                                      #
#   Contact: fsfrazao@dal.ca, oliver.kirsebom@dal.ca                               #
#   Organization: MERIDIAN (https://meridian.cs.dal.ca/)                           #
#   Team: Data Analytics                                                           #
#   Project: ketos                                                                 #
#   Project goal: The ketos library provides functionalities for handling          #
#   and processing acoustic data and applying deep neural networks to sound        #
#   detection and classification tasks.                                            #
#                                                                                  #
#   License: GNU GPLv3                                                             #
#                                                                                  #
#       This program is free software: you can redistribute it and/or modify       #
#       it under the terms of the GNU General Public License as published by       #
#       the Free Software Foundation, either version 3 of the License, or          #
#       (at your option) any later version.                                        #
#                                                                                  #
#       This program is distributed in the hope that it will be useful,            #
#       but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              #
#       GNU General Public License for more details.                               # 
#                                                                                  #
#       You should have received a copy of the GNU General Public License          #
#       along with this program.  If not, see <https://www.gnu.org/licenses/>.     #
# ================================================================================ #

""" Unit tests for the data_feeding module within the ketos library
"""

import os
import pytest
import numpy as np
import pandas as pd
from tables import open_file
from ketos.data_handling.database_interface import open_table
from ketos.data_handling.data_feeding import BatchGenerator
from ketos.neural_networks.neural_networks import class_confidences, predictions

current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')


def test_one_batch():
    """ Test if one batch has the expected shape and contents
    """
    h5 = open_file(os.path.join(path_to_assets, "mini_narw.h5"), 'r') # create the database handle  
    train_data = open_table(h5, "/train/data")

    five_specs = train_data[:5]['data']
    five_labels = train_data[:5]['label']
    
    five_labels = [np.array(l) for l in five_labels]

    train_generator = BatchGenerator(data_table=train_data,batch_size=5, return_batch_ids=True) #create a batch generator 
    ids, X, Y = next(train_generator)
    
    np.testing.assert_array_equal(ids,[0,1,2,3,4])
    assert X.shape == (5, 94, 129)
    np.testing.assert_array_equal(X, five_specs)
    assert Y.shape == (5,)
    np.testing.assert_array_equal(Y['label'], five_labels)

    h5.close()

#TODO: create a test database with strong annotations and test if the batch generator returns start, end, min_freq, max_freq instead of label
def test_output_for_strong_annotations():
    """ Test if batch generator returns multiple labels for strongly annotated instances
    """
    h5 = open_file(os.path.join(path_to_assets, "11x_same_spec.h5"), 'r') # create the database handle  
    data = open_table(h5, "/group_1/table_data")
    annot = open_table(h5, "/group_1/table_annot")
    

    
    expected_y = np.array([[annot[0]['label'],annot[1]['label']],
                            [annot[2]['label'],annot[3]['label']],
                            [annot[4]['label'],annot[5]['label']],
                            [annot[6]['label'],annot[7]['label']],
                            [annot[8]['label'],annot[9]['label']]])


    train_generator = BatchGenerator(batch_size=5, data_table=data, annot_in_data_table=False, annot_table=annot, y_field=['label'], shuffle=False, refresh_on_epoch_end=False)
    
    _, Y = next(train_generator)
    np.testing.assert_array_equal(Y['label'], expected_y)

    h5.close()
    

def test_batch_sequence_same_as_db():
    """ Test if batches are generated with instances in the same order as they appear in the database
    """
    h5 = open_file(os.path.join(path_to_assets, "mini_narw.h5"), 'r') # create the database handle  
    train_data = open_table(h5, "/train/data")
    train_annot = open_table(h5, "/train/data_annot")

    ids_in_db = train_data[:]['id']
    train_generator = BatchGenerator(data_table=train_data, annot_table=train_annot, batch_size=3, return_batch_ids=True) #create a batch generator 

    for i in range(3):
        ids, X, _ = next(train_generator)
        np.testing.assert_array_equal(X, train_data[ids_in_db[i*3: i*3+3]]['data'])
        np.testing.assert_array_equal(ids,list(range(i*3, i*3+3)))
    
    h5.close()


def test_last_batch():
    """ Test if last batch has the expected number of instances
    """
    h5 = open_file(os.path.join(path_to_assets, "mini_narw.h5"), 'r') # create the database handle  
    train_data = open_table(h5, "/train/data")
    train_annot = open_table(h5, "/train/data_annot")

    ids_in_db = train_data[:]['id']
    train_generator = BatchGenerator(data_table=train_data, annot_table=train_annot, batch_size=6, return_batch_ids=True) #create a batch generator 
    #First batch
    ids, X, _ = next(train_generator)
    np.testing.assert_array_equal(ids,[0,1,2,3,4,5])
    assert X.shape == (6, 94, 129)
    #Second batch
    ids, X, _ = next(train_generator)
    np.testing.assert_array_equal(ids,[6,7,8,9,10,11])
    assert X.shape == (6, 94, 129)

    #Third batch; Last batch ( will have the remaining instances)
    ids, X, _ = next(train_generator)
    np.testing.assert_array_equal(ids,[12, 13, 14, 15, 16, 17, 18, 19])
    assert X.shape == (8, 94, 129)
    
    h5.close()

def test_use_only_subset_of_data():
    """ Test that only the indices specified are used
    """
    h5 = open_file(os.path.join(path_to_assets, "mini_narw.h5"), 'r') # create the database handle  
    train_data = open_table(h5, "/train/data")
    train_annot = open_table(h5, "/train/data_annot")
    
    train_generator = BatchGenerator(data_table=train_data, annot_table=train_annot, batch_size=4, select_indicess=[1,3,5,7,9,11,13,14], return_batch_ids=True) #create a batch generator 
    #First batch
    ids, X, _ = next(train_generator)
    np.testing.assert_array_equal(ids,[1,3,5,7])
    #Second batch
    ids, X, _ = next(train_generator)
    np.testing.assert_array_equal(ids,[9,11,13,14])

    h5.close()

def test_multiple_epochs():
    """ Test if batches are as expected after the first epoch
    """
    h5 = open_file(os.path.join(path_to_assets, "mini_narw.h5"), 'r') # create the database handle  
    train_data = open_table(h5, "/train/data")
    train_annot = open_table(h5, "/train/data_annot")

    ids_in_db = train_data[:]['id']
    train_generator = BatchGenerator(data_table=train_data, annot_table=train_annot, batch_size=6, return_batch_ids=True) #create a batch generator 
    #Epoch 0, batch 0
    ids, X, _ = next(train_generator)
    np.testing.assert_array_equal(ids,[0,1,2,3,4,5])
    assert X.shape == (6, 94, 129)
    #Epoch 0, batch 1
    ids, X, _ = next(train_generator)
    np.testing.assert_array_equal(ids,[6,7,8,9,10,11])
    assert X.shape == (6, 94, 129)

    ##Epoch 0, batch 2 Last batch ( will have the remaining instances)
    ids, X, _ = next(train_generator)
    np.testing.assert_array_equal(ids,[12, 13, 14, 15, 16, 17, 18, 19])
    assert X.shape == (8, 94, 129)
    
    #Epoch 1, batch 0
    ids, X, _ = next(train_generator)
    np.testing.assert_array_equal(ids,[0,1,2,3,4,5])
    assert X.shape == (6, 94, 129)

    h5.close()

def test_load_from_memory():
    """ Test if batch generator can work with data loaded from memory
    """
    x = np.ones(shape=(15,32,16))
    y = np.zeros(shape=(15))

    generator = BatchGenerator(x=x, y=y, batch_size=6, return_batch_ids=True) #create a batch generator 

    #Epoch 0, batch 0
    ids, X, _ = next(generator)
    assert ids == [0,1,2,3,4,5]
    assert X.shape == (6, 32, 16)
    #Epoch 0, batch 1
    ids, X, _ = next(generator)
    assert ids == [6,7,8,9,10,11,12,13,14]
    assert X.shape == (9, 32, 16)
    
    
    #Epoch 1, batch 0
    ids, X, _ = next(generator)
    assert ids == [0,1,2,3,4,5]
    assert X.shape == (6, 32, 16)

def test_shuffle():
    """Test shuffle argument.
        Instances should be shuffled before divided into batches, but the order should be consistent across epochs if
        'refresh_on_epoch_end' is False.
    """
    h5 = open_file(os.path.join(path_to_assets, "mini_narw.h5"), 'r') # create the database handle  
    train_data = open_table(h5, "/train/data")
    train_annot = open_table(h5, "/train/data_annot")

    np.random.seed(100)

    ids_in_db = train_data[:]['id']
    train_generator = BatchGenerator(data_table=train_data, annot_table=train_annot, batch_size=6, return_batch_ids=True, shuffle=True) #create a batch generator 

    
    for epoch in range(5):
        #batch 0
        ids, X, _ = next(train_generator)
        np.testing.assert_array_equal(ids,[17, 19, 11, 18, 13,  6])
        assert X.shape == (6,94,129)
        #batch 1
        ids, X, _ = next(train_generator)
        np.testing.assert_array_equal(ids, [16, 1, 9, 14, 12, 5])
        assert X.shape == (6, 94, 129)
        #batch 2
        ids, X, _ = next(train_generator)
        np.testing.assert_array_equal(ids,[2, 4, 10, 0, 15, 7, 3, 8])
        assert X.shape == (8, 94, 129)

       
    h5.close()


def test_refresh_on_epoch_end():
    """ Test if batches are generated with randomly selected instances for each epoch
    """
    h5 = open_file(os.path.join(path_to_assets, "mini_narw.h5"), 'r') # create the database handle  
    train_data = open_table(h5, "/train/data")
    train_annot = open_table(h5, "/train/data_annot")

    np.random.seed(100)

    ids_in_db = train_data[:]['id']
    train_generator = BatchGenerator(data_table=train_data, annot_table=train_annot, batch_size=6, return_batch_ids=True, shuffle=True, refresh_on_epoch_end=True) #create a batch generator 

    expected_ids = {'epoch_1':  ([17, 19, 11, 18, 13, 6], [16, 1, 9, 14, 12, 5], [2, 4, 10, 0, 15, 7, 3, 8]),
                     'epoch_2': ([18, 19, 17, 0, 8, 6], [14, 7, 11, 10, 15, 3], [5, 13, 1, 4, 12, 2, 9, 16]),    
                     'epoch_3': ([3, 4, 12, 17, 10, 1], [19, 5, 11, 8, 0, 18], [6, 13, 7, 15, 16, 14, 2, 9])}
                     
    for epoch in ['epoch_1', 'epoch_2', 'epoch_3']:
        print(train_generator.batch_indices_data)
        #batch 0
        ids, X, _ = next(train_generator)
        print(epoch)
        np.testing.assert_array_equal(ids,expected_ids[epoch][0])
        #batch 1
        ids, X, _ = next(train_generator)
        np.testing.assert_array_equal(ids,expected_ids[epoch][1])
        #batch 2
        ids, X, _ = next(train_generator)
        np.testing.assert_array_equal(ids,expected_ids[epoch][2])
       
    
    h5.close()

def test_refresh_on_epoch_end_annot():
    """ Test if the correct annotation labels are when the batches are refreshed
    """
    h5 = open_file(os.path.join(path_to_assets, "mini_narw.h5"), 'r') # create the database handle  
    train_data = open_table(h5, "/train/data")
    train_annot = open_table(h5, "/train/data_annot")

    np.random.seed(100)

    def transform_output(x,y):
        X = x
        print(y)
        Y = np.array([(value[0], value[1]) for value in y])
       

        return X,Y


    ids_in_db = train_data[:]['id']
    train_generator = BatchGenerator(data_table=train_data, annot_table=train_annot, batch_size=6,
                                     y_field=['data_index','label'], return_batch_ids=True, shuffle=True,
                                     refresh_on_epoch_end=True, output_transform_func=transform_output) #create a batch generator 

    expected_ids = {'epoch_1':  ([17, 19, 11, 18, 13, 6], [16, 1, 9, 14, 12, 5], [2, 4, 10, 0, 15, 7, 3, 8]),
                     'epoch_2': ([18, 19, 17, 0, 8, 6], [14, 7, 11, 10, 15, 3], [5, 13, 1, 4, 12, 2, 9, 16]),    
                     'epoch_3': ([3, 4, 12, 17, 10, 1], [19, 5, 11, 8, 0, 18], [6, 13, 7, 15, 16, 14, 2, 9])}

    expected_labels = {'epoch_1':  ([0, 0, 0, 0, 0, 1], [0, 1, 1, 0, 0, 1], [1, 1, 0, 1, 0, 1, 1, 1]),
                     'epoch_2': ([0, 0, 0, 1, 1, 1], [0, 1, 0, 0, 0, 1], [1, 0, 1, 1, 0, 1, 1, 0]),    
                     'epoch_3': ([1, 1, 0, 0, 0, 1], [0, 1, 0, 1, 1, 0], [1, 0, 1, 0, 0, 0, 1, 1])}
                     
    for epoch in ['epoch_1', 'epoch_2', 'epoch_3']:
        #batch 0
        ids, X, Y = next(train_generator)
     
        #print(Y)

        
        np.testing.assert_array_equal(Y[:,0],expected_ids[epoch][0])
        np.testing.assert_array_equal(Y[:,1],expected_labels[epoch][0])
        #batch 1
        ids, X, Y = next(train_generator)
     
        np.testing.assert_array_equal(Y[:,0],expected_ids[epoch][1])
        np.testing.assert_array_equal(Y[:,1],expected_labels[epoch][1])
        #batch 2
        ids, X, Y = next(train_generator)
     
        np.testing.assert_array_equal(Y[:,0],expected_ids[epoch][2])
        np.testing.assert_array_equal(Y[:,1],expected_labels[epoch][2])
        
       
    
    h5.close()

def test_instance_function():
    """ Test if the function passed as 'instance_function' is applied to the batch
    """
    h5 = open_file(os.path.join(path_to_assets, "mini_narw.h5"), 'r') # create the database handle  
    train_data = open_table(h5, "/train/data")
    train_annot = open_table(h5, "/train/data_annot")

    def apply_to_batch(X,Y):
        X = np.mean(X, axis=(1,2))
        return (X, Y)

    train_generator = BatchGenerator(data_table=train_data, annot_table=train_annot, batch_size=6, return_batch_ids=True, output_transform_func=apply_to_batch) #create a batch generator 
    
    _, X, Y = next(train_generator)
    assert X.shape == (6,)
    assert X[0] == pytest.approx(-37.345703, 0.1)
    assert Y.shape == (6,)
    
    h5.close()


