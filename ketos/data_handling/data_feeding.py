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

""" Data feeding module within the ketos library

    This module provides utilities to load data and feed it to models.

    Contents:
        BatchGenerator class
        TrainiDataProvider class
"""

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from ketos.data_handling.data_handling import check_data_sanity, to1hot



class BatchGenerator():
    """ Creates batches to be fed to a model

        Instances of this class are python generators. They will load one batch at 
        a time from a HDF5 database, which is particularly useful when working with 
        larger than memory datasets. Yield (X,Y) or (ids,X,Y) if 'return_batch_ids' 
        is True. X is a batch of data as a np.array of shape (batch_size,mx,nx) where 
        mx,nx are the shape of on instance of X in the database. Similarly, Y is an 
        np.array of shape[0]=batch_size with the corresponding labels.

        It is also possible to load the entire data set into memory and provide it 
        to the BatchGenerator via the arguments x and y. This can be convenient when 
        working with smaller data sets.

        Args:
            batch_size: int
                The number of instances in each batch. The last batch of an epoch might 
                have fewer examples, depending on the number of instances in the hdf5_table.
            hdf5_table: pytables table (instance of table.Table()) 
                The HDF5 table containing the data
            x: numpy array
                Array containing the data images.
            y: numpy array
                Array containing the data labels.
            indices: list of ints
                Indices of those instances that will retrieved from the HDF5 table by the 
                BatchGenerator. By default all instances are retrieved.
            output_transform_func: function
                A function to be applied to the batch, transforming the instances. Must accept 
                'X' and 'Y' and, after processing, also return  'X' and 'Y' in a tuple.
            x_field: str
                The name of the column containing the X data in the hdf5_table
            y_field: str
                The name of the column containing the Y labels in the hdf5_table
            shuffle: bool
                If True, instances are selected randomly (without replacement). If False, 
                instances are selected in the order the appear in the database
            refresh_on_epoch: bool
                If True, and shuffle is also True, resampling is performed at the end of 
                each epoch resulting in different batches for every epoch. If False, the 
                same batches are used in all epochs.
                Has no effect if shuffle is False.
            return_batch_ids: bool
                If False, each batch will consist of X and Y. If True, the instance indices 
                (as they are in the hdf5_table) will be included ((ids, X, Y)).
            filter: str
                A valid PyTables query. If provided, the Batch Generator will query the hdf5
                database before defining the batches and only the matching records will be used.
                Only relevant when data is passed through the hdf5_table argument. If both 'filter'
                and 'indices' are passed, 'indices' is ignored.

        Attr:
            data: pytables table (instance of table.Table()) 
                The HDF5 table containing the data
            n_instances: int
                The number of intances (rows) in the hdf5_table
            n_batches: int
                The number of batches of size 'batch_size' for each epoch
            entry_indices:list of ints
                A list of all intance indices, in the order used to generate batches for this epoch
            batch_indices: list of tuples (int,int)
                A list of (start,end) indices for each batch. These indices refer to the 'entry_indices' attribute.
            batch_count: int
                The current batch within the epoch. This will be the batch yielded on the next call to 'next()'.
            from_memory: bool
                True if the data are loaded from memory rather than an HDF5 table.

        Examples:
            >>> from tables import open_file
            >>> from ketos.data_handling.database_interface import open_table
            >>> h5 = open_file("ketos/tests/assets/15x_same_spec.h5", 'r') # create the database handle  
            >>> train_data = open_table(h5, "/train/species1")
            >>> train_generator = BatchGenerator(hdf5_table=train_data, batch_size=3, return_batch_ids=True) #create a batch generator 
            >>> #Run 2 epochs. 
            >>> n_epochs = 2    
            >>> for e in range(n_epochs):
            ...    for batch_num in range(train_generator.n_batches):
            ...        ids, batch_X, batch_Y = next(train_generator)   
            ...        print("epoch:{0}, batch {1} | instance ids:{2}, X batch shape: {3}, Y batch shape: {4}".format(e, batch_num, ids, batch_X.shape, batch_Y.shape))
            epoch:0, batch 0 | instance ids:[0, 1, 2], X batch shape: (3, 2413, 201), Y batch shape: (3,)
            epoch:0, batch 1 | instance ids:[3, 4, 5], X batch shape: (3, 2413, 201), Y batch shape: (3,)
            epoch:0, batch 2 | instance ids:[6, 7, 8], X batch shape: (3, 2413, 201), Y batch shape: (3,)
            epoch:0, batch 3 | instance ids:[9, 10, 11], X batch shape: (3, 2413, 201), Y batch shape: (3,)
            epoch:0, batch 4 | instance ids:[12, 13, 14], X batch shape: (3, 2413, 201), Y batch shape: (3,)
            epoch:1, batch 0 | instance ids:[0, 1, 2], X batch shape: (3, 2413, 201), Y batch shape: (3,)
            epoch:1, batch 1 | instance ids:[3, 4, 5], X batch shape: (3, 2413, 201), Y batch shape: (3,)
            epoch:1, batch 2 | instance ids:[6, 7, 8], X batch shape: (3, 2413, 201), Y batch shape: (3,)
            epoch:1, batch 3 | instance ids:[9, 10, 11], X batch shape: (3, 2413, 201), Y batch shape: (3,)
            epoch:1, batch 4 | instance ids:[12, 13, 14], X batch shape: (3, 2413, 201), Y batch shape: (3,)
            >>> #Applying a custom function to the batch
            >>> #Takes the mean of each instance in X; leaves Y untouched
            >>> def apply_to_batch(X,Y):
            ...    X = np.mean(X, axis=(1,2)) #since X is a 3d array
            ...    return (X,Y)
            >>> train_generator = BatchGenerator(hdf5_table=train_data, batch_size=3, return_batch_ids=False, output_transform_func=apply_to_batch) 
            >>> X,Y = next(train_generator)                
            >>> #Now each X instance is one single number, instead of a (2413,201) matrix
            >>> #A batch of size 3 is an array of the 3 means
            >>> X.shape
            (3,)
            >>> #Here is how one X instance looks like
            >>> X[0]
            7694.1147
            >>> #Y is the same as before 
            >>> Y.shape
            (3,)
            >>> h5.close()
    """
    def __init__(self, batch_size, data_table=None, annot_in_data_table=True, annot_table=None, x=None, y=None, select_indices=None, output_transform_func=None, x_field='data', y_field='label',\
                    shuffle=False, refresh_on_epoch_end=False, return_batch_ids=False, filter=None):

        self.from_memory = x is not None and y is not None
        self.annot_in_data_table = annot_in_data_table
        

        if self.from_memory:
            check_data_sanity(x, y)
            self.x = x
            self.y = y

            if select_indices is None:
                self.select_indices = np.arange(len(self.x), dtype=int) 
            else:
                self.select_indices = select_indices
            self.n_instances = len(self.select_indices)
        else:
            assert (data_table is not None) and (annot_table is not None), 'data_table + annot_table or x + y must be specified'
            self.data = data_table
            self.annot = annot_table
            self.x_field = x_field
            self.y_field = y_field
            if type(self.y_field) is not list:
                self.y_field = [self.y_field]
            if select_indices is None:
                self.n_instances = self.data.nrows
                self.select_indices = self.data[:]['id']
            else:
                self.n_instances = len(select_indices)
                self.select_indices = select_indices

            if filter is not None:
                self.id_row_index = self.data.get_where_list(self.filter)
                self.select_indices = self.data[self.id_row_index]['id']
                self.n_instances = len(self.select_indices)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.output_transform_func = output_transform_func
        self.batch_count = 0
        self.refresh_on_epoch_end = refresh_on_epoch_end
        self.return_batch_ids = return_batch_ids


        self.n_batches = int(self.n_instances // self.batch_size)

        self.entry_indices = self.__update_indices__()

        self.batch_indices_data, self.batch_indices_annot = self.__get_batch_indices__()

    
    def __update_indices__(self):
        """Updates the indices used to divide the instances into batches.

            A list of indices is kept in the self.entry_indices attribute.
            The order of the indices determines which instances will be placed in each batch.
            If the self.shuffle is True, the indices are randomly reorganized, resulting in batches with randomly selected instances.

            Returns
                indices: list of ints
                    The list of instance indices
        """

        if self.from_memory:
            row_index = self.select_indices
            
            if self.shuffle:
                np.random.shuffle(self.select_indices)
            return row_index
        else:
            
            
            # row_index = np.array([(row_idx, row['id']) for row_idx, row in enumerate(self.data.iterrows()) if row['id'] in self.select_indices])
            # annot_row_index = np.array([(row['data_index'], row_index[row_index[:,1] == row['data_index'],0][0], annot_idx) for annot_idx,row in enumerate(self.annot.iterrows()) if row['data_index'] in row_index[:,1]])
            annot_row_index = np.array([(row['data_index'], annot_idx) for annot_idx,row in enumerate(self.annot.iterrows()) if row['data_index'] in self.select_indices])
            
        
            if self.shuffle:
                np.random.shuffle(annot_row_index)
              
        #[data_id, data_idx, annot_idx]
        return annot_row_index

    def __get_batch_indices__(self):
        """Selects the indices for each batch

            Divides the instances into batchs of self.batch_size, based on the list generated by __update_indices__()

            Returns:
                list_of_indices: list of tuples
                    A list of tuple, each containing two integer values: the start and end of the batch. These positions refer to the list stored in self.entry_indices.                
        
        """
        if self.from_memory:
            ids = self.entry_indices
        else:
            ids = np.unique(self.entry_indices[:,0])#data_index


        n_complete_batches = int( self.n_instances // self.batch_size) # number of batches that can accomodate self.batch_size intances
        extra_instances = self.n_instances % self.batch_size
    
        list_of_indices = [list(ids[(i*self.batch_size):(i*self.batch_size)+self.batch_size]) for i in range(n_complete_batches)]
        if extra_instances > 0:
            extra_instance_ids = list(ids[-extra_instances:])
            list_of_indices[-1]=list_of_indices[-1] + extra_instance_ids
        if self.from_memory:
            batch_indices = list_of_indices
            annot_indices = list_of_indices
        else:
                        
            data_indices = list_of_indices
            if self.annot_in_data_table:
                annot_indices = [[self.entry_indices[self.entry_indices[:,0]==data_idx,1] for data_idx in batch] for batch in list_of_indices] 
                annot_indices = [np.vstack(batch).flatten() for batch in annot_indices]
            else:
                annot_indices = None
        

        return data_indices, annot_indices

    def __iter__(self):
        return self

    def __next__(self):
        """         
            Return: tuple
            A batch of instances (X,Y) or, if 'returns_batch_ids" is True, a batch of instances accompanied by their indices (ids, X, Y) 
        """

        batch_data_row_index = self.batch_indices_data[self.batch_count]
        if self.annot_in_data_table:
            batch_annot_row_index = self.batch_indices_annot[self.batch_count]
        
        if self.from_memory:
            batch_ids = batch_data_row_index
            
        else:
            batch_ids = self.entry_indices[np.isin(self.entry_indices[:,0], batch_data_row_index),1]

        if self.from_memory:
            X = np.take(self.x, batch_ids, axis=0)
            Y = np.take(self.y, batch_ids, axis=0)
        else:
            X = self.data[batch_data_row_index][self.x_field]

            if self.annot_in_data_table:            
                Y = self.annot[batch_annot_row_index][['data_index'] + self.y_field]
                Y = np.split(Y[self.y_field], np.cumsum(np.unique(Y['data_index'], return_counts=True)[1])[:-1])
            else:
                Y = self.data[batch_data_row_index][self.y_field]
            

        self.batch_count += 1
        if self.batch_count > (self.n_batches - 1):
            self.batch_count = 0
            if self.refresh_on_epoch_end:
                self.entry_indices = self.__update_indices__()
                self.batch_indices_data, self.batch_indices_annot = self.__get_batch_indices__()

        if self.output_transform_func is not None:
            X,Y = self.output_transform_func(X,Y)

        if self.return_batch_ids:
            return (batch_ids,X,Y)
        else:
            return (X, Y)

