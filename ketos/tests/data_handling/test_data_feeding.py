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
import warnings
import numpy as np
import pandas as pd
from tables import open_file
from ketos.data_handling.database_interface import open_table
from ketos.data_handling.data_feeding import BatchGenerator, JointBatchGen, MultiModalBatchGen
from ketos.neural_networks.resnet import ResNetInterface


current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')


def test_get_unique_labels_and_mapping_db_without_attr():
    """ Check that we can retrive a list of unique labels from a table
        that does not contain a unique_labels attribute.
        Also check that the correct label mapping is automatically created. 
    """
    f = open_file(os.path.join(path_to_assets, "db.h5"), 'r') 
    tbl = open_table(f, "/audio")
    g = BatchGenerator(data_table=tbl, batch_size=2)
    ul = g.get_unique_labels()
    assert isinstance(ul, np.ndarray)
    assert np.all(ul == [1,3,5]) 
    ul = g.get_unique_labels() #get them a second time
    assert isinstance(ul, np.ndarray)
    assert np.all(ul == [1,3,5]) 
    assert g.mapper == {1:0, 3:1, 5:2}
    f.close()  

def test_get_unique_labels_and_mapping_db_with_attr():
    """ Check that we can retrive a list of unique labels from a table
        that does contain a unique_labels attribute.
        Also check that the correct label mapping is automatically created. 
        Finally, check that attribute is ignored if indices are specified.
    """
    f = open_file(os.path.join(path_to_assets, "db-ul.h5"), 'r') 
    tbl = open_table(f, "/audio")
    # create batch generator with all instances
    g = BatchGenerator(data_table=tbl, batch_size=2)
    ul = g.get_unique_labels()
    assert isinstance(ul, np.ndarray)
    assert np.all(ul == [2,4]) 
    assert g.mapper == {2:0, 4:1}
    # create batch generator with specific indices
    g = BatchGenerator(data_table=tbl, batch_size=2, select_indices=[0,1,2])
    ul = g.get_unique_labels()
    assert isinstance(ul, np.ndarray)
    assert np.all(ul == [1,3,5]) 
    assert g.mapper == {1:0, 3:1, 5:2}
    f.close()  

def test_get_unique_labels_and_mapping_joint_batch_generator():
    """ Check that we can retrive a list of unique labels from a joint batch generator.
        Also check that the correct label mapping is automatically created. 
    """
    f1 = open_file(os.path.join(path_to_assets, "db.h5"), 'r') 
    tbl1 = open_table(f1, "/audio")
    g1 = BatchGenerator(data_table=tbl1, batch_size=2)
    f2 = open_file(os.path.join(path_to_assets, "db-ul.h5"), 'r') 
    tbl2 = open_table(f2, "/audio")
    g2 = BatchGenerator(data_table=tbl2, batch_size=2)
    g = JointBatchGen([g1, g2])
    ul = g.get_unique_labels()
    assert isinstance(ul, np.ndarray)
    assert np.all(ul == [1,2,3,4,5]) 
    assert g.mapper == {1:0, 2:1, 3:2, 4:3, 5:4}
    f1.close()  
    f2.close()  

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
    np.testing.assert_array_equal(Y, five_labels)

    h5.close()

def test_multiple_data_fields():
    """ Test if one batch has the expected shape and contents when loading multiple data fields 
        from the same table
    """
    h5 = open_file(os.path.join(path_to_assets, "mini_narw_mult.h5"), 'r') # create the database handle  
    train_data = open_table(h5, "/train/data")

    five_specs = train_data[:5]['spec']
    five_gammas = train_data[:5]['gamma']
    five_labels = [0,0,0,0,0]

    five_labels = [np.array(l) for l in five_labels]

    train_generator = BatchGenerator(data_table=train_data, batch_size=5, x_field=['spec','gamma'], return_batch_ids=True) #create a batch generator 
    ids, X, Y = next(train_generator)
    
    np.testing.assert_array_equal(ids,[0,1,2,3,4])
    assert len(X) == 5
    assert len(X[0]) == 2
    assert X[0]['spec'].shape == (94, 129)
    assert X[0]['gamma'].shape == (3000, 20)
    specs = [x[0] for x in X]
    gammas = [x[1] for x in X]
    np.testing.assert_array_equal(specs, five_specs)
    np.testing.assert_array_equal(gammas, five_gammas)
    assert Y.shape == (5,)
    np.testing.assert_array_equal(Y, five_labels)

    h5.close()

def test_multiple_data_fields_map_labels_false():
    """ Test if one batch has the expected shape and contents when loading multiple data fields 
        from the same table when the map labels parameter of the batch generator is false
    """
    h5 = open_file(os.path.join(path_to_assets, "mini_narw_mult.h5"), 'r') # create the database handle  
    train_data = open_table(h5, "/train/data")

    five_specs = train_data[:5]['spec']
    five_gammas = train_data[:5]['gamma']
    five_labels = train_data[:5]['label']

    five_labels = [np.array(l) for l in five_labels]

    train_generator = BatchGenerator(data_table=train_data, batch_size=5, map_labels=False, x_field=['spec','gamma'], return_batch_ids=True) #create a batch generator 
    ids, X, Y = next(train_generator)
    
    np.testing.assert_array_equal(ids,[0,1,2,3,4])
    assert len(X) == 5
    assert len(X[0]) == 2
    assert X[0]['spec'].shape == (94, 129)
    assert X[0]['gamma'].shape == (3000, 20)
    specs = [x[0] for x in X]
    gammas = [x[1] for x in X]
    np.testing.assert_array_equal(specs, five_specs)
    np.testing.assert_array_equal(gammas, five_gammas)
    assert Y.shape == (5,)
    np.testing.assert_array_equal(Y, five_labels)

    h5.close()

def test_output_for_strong_annotations():
    """ Test if batch generator returns multiple labels for strongly annotated instances
    """
    h5 = open_file(os.path.join(path_to_assets, "11x_same_spec.h5"), 'r') # create the database handle  
    data = open_table(h5, "/group_1/table_data")
    annot = open_table(h5, "/group_1/table_annot")

    expected_y = np.array([[0,1], [0,1], [0,1], [0,1], [0,1]])
                            
    train_generator = BatchGenerator(batch_size=5, data_table=data, annot_in_data_table=False, 
        annot_table=annot, y_field=['label'], shuffle=False, refresh_on_epoch_end=False)
    
    _, Y = next(train_generator)
    np.testing.assert_array_equal(Y, expected_y)

    h5.close()


def test_output_for_strong_annotations_not_all_samples_have_annotations():
    """ Test if batch generator returns multiple labels for strongly annotated instances, when 
        some samples have no annotations
    """
    h5 = open_file(os.path.join(path_to_assets, "11x_wf_annot_tbl.h5"), 'r') # create the database handle  
    data = open_table(h5, "/audio")
    annot = open_table(h5, "/audio_annot")

    # first, using only one y-field
    expected_y = np.array([[0, 1], [2], [], [0, 1]])                            
    train_generator = BatchGenerator(batch_size=4, data_table=data, annot_in_data_table=False, 
        annot_table=annot, y_field=['label'], shuffle=False, refresh_on_epoch_end=False)
    _, Y = next(train_generator)
    np.testing.assert_array_equal(Y, expected_y)

    # first, now let's try with two y-fields
    dtype = {'names':['label','end'], 'formats':['u1','<f8'], 'offsets':[20,4], 'itemsize':29}
    expected_y = [np.array([(0, 2.2), (1, 4.2)], dtype=dtype), 
                  np.array([(2, 6.2)], dtype=dtype), 
                  np.array([], dtype=dtype), 
                  np.array([(0, 2.2), (1, 4.2)], dtype=dtype)]                            
    train_generator = BatchGenerator(batch_size=4, data_table=data, annot_in_data_table=False, 
        annot_table=annot, y_field=['label','end'], shuffle=False, refresh_on_epoch_end=False)
    _, Y = next(train_generator)
    for _Y,_y in zip(Y, expected_y):
        np.testing.assert_array_equal(_Y, _y)

    h5.close()


def test_batch_sequence_same_as_db():
    """ Test if batches are generated with instances in the same order as they appear in the database
    """
    h5 = open_file(os.path.join(path_to_assets, "mini_narw.h5"), 'r') # create the database handle  
    train_data = open_table(h5, "/train/data")

    ids_in_db = train_data[:]['id']
    train_generator = BatchGenerator(data_table=train_data, batch_size=3, return_batch_ids=True) #create a batch generator 

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

    ids_in_db = train_data[:]['id']
    train_generator = BatchGenerator(data_table=train_data, batch_size=6, return_batch_ids=True) #create a batch generator 
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
    
    train_generator = BatchGenerator(data_table=train_data, batch_size=4, select_indices=[1,3,5,7,9,11,13,14], return_batch_ids=True) #create a batch generator 
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

    ids_in_db = train_data[:]['id']
    train_generator = BatchGenerator(data_table=train_data, batch_size=6, return_batch_ids=True) #create a batch generator 
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

    np.random.seed(100)

    ids_in_db = train_data[:]['id']
    train_generator = BatchGenerator(data_table=train_data, batch_size=6, return_batch_ids=True, shuffle=True) #create a batch generator 

    
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

    np.random.seed(100)

    ids_in_db = train_data[:]['id']
    train_generator = BatchGenerator(data_table=train_data, batch_size=6, return_batch_ids=True, shuffle=True, refresh_on_epoch_end=True) #create a batch generator 

    expected_ids = {'epoch_1': ([17, 19, 11, 18, 13,  6], [16,  1,  9, 14, 12,  5], [2,  4, 10,  0, 15, 7,  3,  8]),    
                     'epoch_2':  ([18, 19, 17, 0, 8, 6], [14, 7, 11, 10, 15, 3], [5, 13, 1, 4, 12, 2, 9, 16]),
                     'epoch_3': ([3, 4, 12, 17, 10, 1], [19, 5, 11, 8, 0, 18], [6, 13, 7, 15, 16, 14, 2, 9])}
                     
    for epoch in ['epoch_1', 'epoch_2', 'epoch_3']:
        #batch 0
        ids, X, _ = next(train_generator)
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
    
    np.random.seed(100)

    def transform_output(x,y):
        X = x
        print(y)
        Y = np.array([(value[0], value[1]) for value in y])
        return X,Y

    ids_in_db = train_data[:]['id']
    train_generator = BatchGenerator(data_table=train_data, batch_size=6,
                                     y_field=['label'], return_batch_ids=True, shuffle=True,
                                     refresh_on_epoch_end=True, output_transform_func=None) #create a batch generator 

    expected_ids = {'epoch_1': ([17, 19, 11, 18, 13,  6], [16,  1,  9, 14, 12,  5], [2,  4, 10,  0, 15, 7,  3,  8]),    
                     'epoch_2':  ([18, 19, 17, 0, 8, 6], [14, 7, 11, 10, 15, 3], [5, 13, 1, 4, 12, 2, 9, 16]),
                     'epoch_3': ([3, 4, 12, 17, 10, 1], [19, 5, 11, 8, 0, 18], [6, 13, 7, 15, 16, 14, 2, 9])}
                     
    expected_labels = {'epoch_1':  ([0, 0, 0, 0, 0, 1], [0, 1, 1, 0, 0, 1], [1, 1, 0, 1, 0, 1, 1, 1]),
                     'epoch_2': ([0, 0, 0, 1, 1, 1], [0, 1, 0, 0, 0, 1], [1, 0, 1, 1, 0, 1, 1, 0]),    
                     'epoch_3': ([1, 1, 0, 0, 0, 1], [0, 1, 0, 1, 1, 0], [1, 0, 1, 0, 0, 0, 1, 1])}
                     
    for epoch in ['epoch_1', 'epoch_2', 'epoch_3']:
        #batch 0
        ids, X, Y = next(train_generator)
        np.testing.assert_array_equal(ids,expected_ids[epoch][0])
        np.testing.assert_array_equal(Y,expected_labels[epoch][0])
        #batch 1
        ids, X, Y = next(train_generator)    
        np.testing.assert_array_equal(ids,expected_ids[epoch][1])
        np.testing.assert_array_equal(Y,expected_labels[epoch][1])
        #batch 2
        ids, X, Y = next(train_generator)
        np.testing.assert_array_equal(ids,expected_ids[epoch][2])
        np.testing.assert_array_equal(Y,expected_labels[epoch][2])
    
    h5.close()


def test_output_transform_function():
    """ Test if the function passed as 'instance_function' is applied to the batch
    """
    h5 = open_file(os.path.join(path_to_assets, "mini_narw.h5"), 'r') # create the database handle  
    train_data = open_table(h5, "/train/data")

    def apply_to_batch(X,Y):
        X = np.mean(X, axis=(1,2))
        return (X, Y)

    train_generator = BatchGenerator(data_table=train_data,  batch_size=6, return_batch_ids=True, output_transform_func=apply_to_batch) #create a batch generator 
    
    _, X, Y = next(train_generator)
    assert X.shape == (6,)
    assert X[0] == pytest.approx(-37.345703, 0.1)
    assert Y.shape == (6,)
    
    h5.close()

def test_extended_batches():
    """ Test that batches can be extended to include last/first samples from previous/next batch
    """
    h5 = open_file(os.path.join(path_to_assets, "mini_narw.h5"), 'r') # create the database handle  
    train_data = open_table(h5, "/train/data")

    ids_in_db = train_data[:]['id']
    train_generator = BatchGenerator(data_table=train_data, batch_size=6, return_batch_ids=True, n_extend=2) #create a batch generator 
    
    #First batch
    ids, X, _ = next(train_generator)
    np.testing.assert_array_equal(ids,[0,1,2,3,4,5,6,7])
    assert X.shape == (8, 94, 129)

    #Second batch
    ids, X, _ = next(train_generator)
    np.testing.assert_array_equal(ids,[4,5,6,7,8,9,10,11,12,13])
    assert X.shape == (10, 94, 129)

    #Third batch; Last batch ( will have the remaining instances)
    ids, X, _ = next(train_generator)
    np.testing.assert_array_equal(ids,[10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    assert X.shape == (10, 94, 129)
    
    h5.close()

def test_batch_size_larger_than_dataset_size():
    """ Test that batch size can exceed dataset size
    """
    h5 = open_file(os.path.join(path_to_assets, "mini_narw.h5"), 'r') # create the database handle  
    train_data = open_table(h5, "/train/data")

    ids_in_db = train_data[:]['id']
    with warnings.catch_warnings(record=True) as w:
        train_generator = BatchGenerator(data_table=train_data, batch_size=99, return_batch_ids=True) #create a batch generator 

        assert train_generator.batch_size == 20
        assert len(w) == 1
        assert "The batch size is greater than the number of instances available. Setting batch_size to n_instances." in str(w[-1].message)
    
    #First batch
    ids, X, _ = next(train_generator)
    np.testing.assert_array_equal(ids,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
    assert X.shape == (20, 94, 129)
    
    h5.close()


def test_joint_batch_gen():
    """ Test the a joint batch generator can be used to load from tables with a single data column
    """
    h51 = open_file(os.path.join(path_to_assets, "mini_narw.h5"), 'r') # create the database handle  
    h52 = open_file(os.path.join(path_to_assets, "mini_narw.h5"), 'r') # create the database handle  
    tbl1 = open_table(h51, "/train/data")
    tbl2 = open_table(h52, "/train/data")

    three_specs = tbl1.col('data')[:3]
    three_labels = tbl1.col('label')[:3]

    three_labels = [np.array(l) for l in three_labels]

    gen1 = BatchGenerator(data_table=tbl1, batch_size=3, return_batch_ids=True)  
    gen2 = BatchGenerator(data_table=tbl2, batch_size=2, return_batch_ids=False)  

    gen = JointBatchGen([gen1, gen2], n_batches="min") 
    X, Y = next(gen)
    
    assert len(X) == 5
    assert X[0].shape == (94, 129)
    np.testing.assert_array_equal(X[:3], three_specs)
    np.testing.assert_array_equal(X[3:], three_specs[:2])

    assert len(Y) == 5
    for i in range(3):
        np.testing.assert_array_equal(Y[i], three_labels[i])
    for i in range(2):
        np.testing.assert_array_equal(Y[3+i], three_labels[i])

    h51.close()
    h52.close()

def test_joint_batch_gen_ids():
    """ Test the a joint batch generator can return ids
    """
    h51 = open_file(os.path.join(path_to_assets, "mini_narw.h5"), 'r') # create the database handle  
    h52 = open_file(os.path.join(path_to_assets, "mini_narw.h5"), 'r') # create the database handle  
    tbl1 = open_table(h51, "/train/data")
    tbl2 = open_table(h52, "/train/data")

    three_specs = tbl1.col('data')[:3]
    three_labels = tbl1.col('label')[:3]

    three_labels = [np.array(l) for l in three_labels]

    gen1 = BatchGenerator(data_table=tbl1, batch_size=3, return_batch_ids=True)  
    gen2 = BatchGenerator(data_table=tbl2, batch_size=2, return_batch_ids=False)  

    gen = JointBatchGen([gen1, gen2], n_batches="min", return_batch_ids=True) 
    ids, X, Y = next(gen)
    
    assert len(ids) == 5
    assert len(ids[0]) == 2
    assert np.all(ids[0] == [0, 0])
    assert np.all(ids[1] == [0, 1])
    assert np.all(ids[2] == [0, 2])
    assert np.all(ids[3] == [1, 0])
    assert np.all(ids[4] == [1, 1])

    assert len(X) == 5
    assert X[0].shape == (94, 129)
    np.testing.assert_array_equal(X[:3], three_specs)
    np.testing.assert_array_equal(X[3:], three_specs[:2])

    assert len(Y) == 5
    for i in range(3):
        np.testing.assert_array_equal(Y[i], three_labels[i])
    for i in range(2):
        np.testing.assert_array_equal(Y[3+i], three_labels[i])

    h51.close()
    h52.close()

def test_joint_batch_gen_output_transform():
    """ Test the a joint batch generator can be used to load from tables with a single data column
        while applying the ResNet output transform
    """
    h51 = open_file(os.path.join(path_to_assets, "mini_narw.h5"), 'r') # create the database handle  
    h52 = open_file(os.path.join(path_to_assets, "mini_narw.h5"), 'r') # create the database handle  
    tbl1 = open_table(h51, "/train/data")
    tbl2 = open_table(h52, "/train/data")

    three_specs = tbl1.col('data')[:3]
    three_labels = tbl1.col('label')[:3]

    three_labels = [np.array(l) for l in three_labels]

    gen1 = BatchGenerator(data_table=tbl1, batch_size=3, return_batch_ids=True, output_transform_func=ResNetInterface.transform_batch)  
    gen2 = BatchGenerator(data_table=tbl2, batch_size=2, return_batch_ids=False, output_transform_func=ResNetInterface.transform_batch)  

    gen = JointBatchGen([gen1, gen2], n_batches="min") 
    X, Y = next(gen)
    
    assert len(X) == 5
    assert X[0].shape == (94, 129, 1)
    np.testing.assert_array_equal(X[:3,:,:,0], three_specs)
    np.testing.assert_array_equal(X[3:,:,:,0], three_specs[:2])

    assert len(Y) == 5
    np.testing.assert_array_equal(Y[:3], np.array([[0,1],[0,1],[0,1]]))
    np.testing.assert_array_equal(Y[3:], np.array([[0,1],[0,1]]))

    h51.close()
    h52.close()

def test_joint_batch_gen_multi_modal():
    """ Test the a joint batch generator can be used to load multi-modal data
    """
    h51 = open_file(os.path.join(path_to_assets, "mini_narw_mult.h5"), 'r') # create the database handle  
    h52 = open_file(os.path.join(path_to_assets, "mini_narw_mult.h5"), 'r') # create the database handle  
    tbl1 = open_table(h51, "/train/data")
    tbl2 = open_table(h52, "/train/data")

    three_specs = tbl1.col('spec')[:3]
    three_gammas = tbl1.col('gamma')[:3]
    # three_labels = tbl1.col('label')[:3]
    three_labels = [0, 0, 0]
    three_labels = [np.array(l) for l in three_labels]

    gen1 = BatchGenerator(data_table=tbl1, batch_size=3, x_field=['spec','gamma'])  
    gen2 = BatchGenerator(data_table=tbl2, batch_size=2, x_field=['spec','gamma'])  

    gen = JointBatchGen([gen1, gen2], n_batches="min", return_batch_ids=True) 
    ids, X, Y = next(gen)

    assert len(ids) == 5
    assert len(ids[0]) == 2
    assert np.all(ids[0] == [0, 0])
    assert np.all(ids[1] == [0, 1])
    assert np.all(ids[2] == [0, 2])
    assert np.all(ids[3] == [1, 0])
    assert np.all(ids[4] == [1, 1])

    assert len(X) == 5
    assert len(X[0]) == 2
    assert X[0]['spec'].shape == (94, 129)
    assert X[0]['gamma'].shape == (3000, 20)
    specs = [x[0] for x in X]
    gammas = [x[1] for x in X]
    np.testing.assert_array_equal(specs[:3], three_specs)
    np.testing.assert_array_equal(gammas[:3], three_gammas)
    np.testing.assert_array_equal(specs[3:], three_specs[:2])
    np.testing.assert_array_equal(gammas[3:], three_gammas[:2])

    assert len(Y) == 5
    labels = [y for y in Y]
    np.testing.assert_array_equal(labels[:3], three_labels)
    np.testing.assert_array_equal(labels[3:], three_labels[:2])

    h51.close()
    h52.close()

def test_joint_batch_gen_multi_modal_transform():
    """ Test the a joint batch generator can be used to load multi-modal data
        while applying output transform """
    h51 = open_file(os.path.join(path_to_assets, "mini_narw_mult.h5"), 'r') # create the database handle  
    h52 = open_file(os.path.join(path_to_assets, "mini_narw_mult.h5"), 'r') # create the database handle  
    tbl1 = open_table(h51, "/train/data")
    tbl2 = open_table(h52, "/train/data")

    three_specs = tbl1.col('spec')[:3]
    three_gammas = tbl1.col('gamma')[:3]
    three_labels = [0, 0, 0]

    three_labels = [np.array(l) for l in three_labels]

    def transform_batch(X, Y):
        X = [[x['spec'][:,:,np.newaxis], x['gamma'][:,:,np.newaxis]] for x in X]
        Y = np.array([label for label in Y])        
        return (X,Y)

    gen1 = BatchGenerator(data_table=tbl1, batch_size=3, x_field=['spec','gamma'], output_transform_func=transform_batch)  
    gen2 = BatchGenerator(data_table=tbl2, batch_size=2, x_field=['spec','gamma'], output_transform_func=transform_batch)  

    gen = JointBatchGen([gen1, gen2], n_batches="min") 
    X, Y = next(gen)
    
    assert len(X) == 5
    assert len(X[0]) == 2
    assert X[0][0].shape == (94, 129, 1)
    assert X[0][1].shape == (3000, 20, 1)
    specs = [x[0][:,:,0] for x in X]
    gammas = [x[1][:,:,0] for x in X]
    np.testing.assert_array_equal(specs[:3], three_specs)
    np.testing.assert_array_equal(gammas[:3], three_gammas)
    np.testing.assert_array_equal(specs[3:], three_specs[:2])
    np.testing.assert_array_equal(gammas[3:], three_gammas[:2])

    assert len(Y) == 5
    np.testing.assert_array_equal(Y[:3], three_labels)
    np.testing.assert_array_equal(Y[3:], three_labels[:2])

    h51.close()
    h52.close()


def test_multi_modal_batch_generator():
    """ Test that the multi-modal batch generator behaves as expected
    """
    h51 = open_file(os.path.join(path_to_assets, "mini_narw_mult.h5"), 'r') # create the database handle  
    h52 = open_file(os.path.join(path_to_assets, "mini_narw_mult.h5"), 'r') # create the database handle  
    tbl1 = open_table(h51, "/train/data")
    tbl2 = open_table(h52, "/train/data")

    three_specs = tbl1.col('spec')[:3]
    three_gammas = tbl1.col('gamma')[:3]
    three_labels = [0, 0, 0]

    def transform_batch(X, Y):
        # invert ordering: batch,mode -> mode,batch
        # also, stack the inner (batch) dimension to create numpy arrays
        X = [np.stack([x[i] for x in X]) for i in range(len(X[0]))]
        Y = [np.stack([y[i] for y in Y]) for i in range(len(Y[0]))]
        # only pick the labels from the first mode:
        Y = Y[0]
        return (X,Y)

    gen1 = BatchGenerator(data_table=tbl1, batch_size=3, x_field='spec')  
    gen2 = BatchGenerator(data_table=tbl2, batch_size=3, x_field='gamma')  
    gen = MultiModalBatchGen([gen1, gen2], output_transform_func=transform_batch) 
    X, Y = next(gen)
    assert np.all(X[0] == three_specs)
    assert np.all(X[1] == three_gammas)
    assert np.all(Y == three_labels)

    gen1 = BatchGenerator(data_table=tbl1, batch_size=3, x_field='spec')  
    gen2 = BatchGenerator(data_table=tbl2, batch_size=3, x_field='spec')  
    gen = MultiModalBatchGen([gen1, gen2], shuffle=True, output_transform_func=transform_batch) 
    X, Y = next(gen)
    assert np.all(X[0] == X[1])

    h51.close()
    h52.close()