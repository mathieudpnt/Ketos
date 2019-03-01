""" Data feeding module within the ketos library

    This module provides utilities to load data and feed it to models.

    Contents:
        BatchGenerator class: 
        TrainiDataProvider class

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

import numpy as np
import pandas as pd
from sklearn.utils import shuffle



class BatchGenerator():
    """ Creates batches to be fed to a model

        Instances of this class are python generators. They will load one batch at a time from a HDF5 database, which is particularly useful when working with larger than memory datasets.
        Yield (X,Y) or (ids,X,Y) if 'return_batch_ids' is True. X is a batch of data as a np.array of shape batch_size,mx,nx where mx,nx are the shape of on instance of X in the database. Similarly, Y is batch_size,my,ny with the corresponding labels.


        Args:
            hdf5_table: pytables table (instance of table.Table()) 
                The HDF5 table  containing the data
            batch_size: int
                The number of instances in each batch. The last batch of an epoch might have fewer examples, depending on the number of instances in the hdf5_table.
            instance_function: function
                A function to be applied to the batch. Must accept 'X' and 'Y' and, after processing, also return  'X' and 'Y' in a tuple.
            x_field:str
                The name of the column containing the X data in the hdf5_table
            y_field: str
                The name of the column containing the Y labels in the hdf5_table
            shuffle: bool
                If True, the batches will contain batch_size randomly picked instances. If false batches maintain the order of instances in the database
            refresh_on_epoch: bool
                If True and shuffle is also True, will resample the instance at the end of each epoch, resulting in different batches for each epoch. If false will reuse the same batches for each epoch (i.e: each batch will have the same random selection of samples across batch)
                Has no effect if shuffle is False.
            return_batch_ids: bool
                If False, each batch will consist of X and Y. If True, the instance ids (as they are in the hdf5_table) will be included ((ids, X, Y)).


            Examples:

                >>> import tables
                >>> import os
                >>> import numpy as np
                >>> import ketos.data_handling.database_interface as h5
                >>> from ketos.audio_processing.audio import AudioSignal
                >>> from ketos.audio_processing.spectrogram import MagSpectrogram
                >>> from ketos.data_handling.data_handling import read_wave
           
                >>> rate, sig = read_wave(file='../tests/assets/2min_wav )  
                >>> audio = AudioSignal(rate=rate, data=sig)
                >>> spec = MagSpectrogram(audio, 0.2, 0.05)
                >>> spec.boxes = [[10, 15, 200, 400]
                >>> spec.labels = [[1]]

                >>> f = tables.open_file(tmp_db.h5, 'w')
                    # create table
                >>> tbl = h5.create(h5file=f, path='/train/', name='species1', shape=spec.image.shape)
                    # write 15 instances to table
                >>> for i in range(15):
                ...     h5.write(table=tbl, x=spec)
             


                    
                

    """

    def __init__(self, hdf5_table, batch_size, instance_function=None,x_field='data', y_field='boxes', shuffle=False, refresh_on_epoch_end=False, return_batch_ids=False):
        self.data = hdf5_table
        self.batch_size = batch_size
        self.n_instances = self.data.nrows
        self.n_batches = int(np.ceil(self.n_instances / self.batch_size))
        self.shuffle = shuffle
        self.instance_function = instance_function
        self.entry_indices = self.__update_indices__()
        self.batch_indices = self.__get_batch_indices__()
        self.batch_count = 0
        self.refresh_on_epoch_end = refresh_on_epoch_end
        self.return_batch_ids = return_batch_ids

    
    def __update_indices__(self):
        """Updates the indices used to divide the instances into batches.

            A list of indices is kept in the self.entry_indices attribute.
            The order of the indices determines which instances will be placed in each batch.
            If the self.shuffle is True, the indices are randomly reorganized, resulting in batches with randomly picked instances.

            Returns
                indices: list of ints
                    The list of instance indices
        """
        indices = np.arange(self.n_instances)
        if self.shuffle:
            np.random.shuffle(indices)
        return indices

    def __get_batch_indices__(self):
        """Selects the indices for each batch

            Divides the instances into batchs of self.batch_size, based on the list generated by __update_indices__()

            Returns:
                list_of_indices: list of tuples
                    A list of tuple, each containing two integer values: the start and end of the batch. These positions refer to the list stored in self.entry_indices.                
        
        """
        ids = self.entry_indices
        n_complete_batches = int( self.n_instances // self.batch_size) # number of batches that can accomodate self.batch_size intances
        last_batch_size = self.n_instances % n_complete_batches
    
        list_of_indices = [list(ids[(i*self.batch_size):(i*self.batch_size)+self.batch_size]) for i in range(self.n_batches)]
        if last_batch_size > 0:
            last_batch_ids = list(ids[-last_batch_size:])
            list_of_indices.append(last_batch_ids)

        return list_of_indices

    def __iter__(self):
        return self

    def __next__(self):
        batch_ids = self.batch_indices[self.batch_count]
        X = self.data[batch_ids]['data']
        Y = self.data[batch_ids]['boxes']

        self.batch_count += 1
        if self.batch_count > (self.n_batches - 1):
            self.batch_count = 0
            if self.refresh_on_epoch_end:
                self.entry_indices = self.__update_indices__()
                self.batch_indices = self.__get_batch_indices__()

        if self.instance_function is not None:
            X,Y = self.instance_function(X,Y)

        if self.return_batch_ids:
            return (batch_ids,X,Y)
        else:
            return (X, Y)





class TrainingDataProvider():
    """ Training data provider.

        Args:
            x: pandas DataFrame
                Training data
            y: pandas DataFrame
                Labels for training data
            randomize: bool
                Randomize order of training data
            num_samples: int
                Number of new samples that will be drawn for each iteration
            max_keep: float
                Maximum fraction of samples from previous iteration that will be kept for next iteration
            conf_cut: float
                Correct predictions with confidence below conf_cut will be kept for next iteration
            seed: int
                Seed for random number generator
            equal_rep: bool
                Ensure that new samples drawn at each iteration have equal representation of 0s and 1s
    """
    def __init__(self, x, y, randomize=False, num_samples=100, max_keep=0, conf_cut=0, seed=None, equal_rep=True):

        N = x.shape[0]
        self.x = x
        self.df = pd.DataFrame({'y':y, 'pred':y, 'conf':np.ones(N), 'prev':np.zeros(N, dtype=bool)})
        self.randomize = randomize
        self.num_samples = num_samples
        self.max_keep = max_keep
        self.conf_cut = conf_cut
        self.it = 0
        self.current_pos = 0
        self.equal_rep = equal_rep
        self.seed = seed
        if seed is not None:
            np.random.seed(seed) 

        print('positives: ',  len(self.df[self.df.y == 1]))
        print('negatives: ',  len(self.df[self.df.y == 0]))

        self.posfrac = float(len(self.df[self.df.y == 1])) / float(len(self.df))

    def get_samples(self, num_samples=None, max_keep=None, conf_cut=None):

        # use default value if none provided
        if num_samples is None:
            num_samples = self.num_samples
        if max_keep is None:
            max_keep = self.max_keep
        if conf_cut is None:
            conf_cut = self.conf_cut

        x, y = None, None

        # get poorly performing samples from previous iteration
        num_poor_max = int(np.ceil(num_samples * max_keep))
        idx_poor = self._get_poor(num=num_poor_max, conf_cut=conf_cut)
        num_poor = len(idx_poor)

        # get new samples
        idx = self._get_new(num_samples=num_samples - num_poor, randomize=self.randomize, equal_rep=self.equal_rep)
        if num_poor > 0:
            idx = idx.union(idx_poor)

        # combine poorly performing and new samples
        df = self.df.loc[idx]
        x = self.x[idx]
        y = df.y.values

        # internal book keeping
        self.df.prev = False
        self.df.loc[idx,'prev'] = True

        keep_frac = num_poor / len(idx)
        return x, y, keep_frac

    def update_prediction_confidence(self, pred, conf):
        assert len(pred) == len(conf),'length of prediction and confidence arrays do not match'
        idx = self.df[self.df.prev == True].index
        assert len(pred) == len(idx),'length of prediction and confidence arrays do not match the number of samples drawn in the last iteration'
        self.df.loc[idx,'pred'] = pred
        self.df.loc[idx,'conf'] = conf

    def _get_poor(self, num, conf_cut):
        df_prev = self.df[self.df.prev == True]
        N = df_prev.shape[0]
        if N == 0:
            return list()
        df_poor = df_prev[(df_prev.pred != df_prev.y) | (df_prev.conf < self.conf_cut)]
        if df_poor.shape[0] == 0:
            return list()
        M = min(df_poor.shape[0], num)
        idx = np.random.choice(df_poor.index, M, replace=False)
        idx = pd.Index(idx)
        return idx

    def _get_new(self, num_samples, randomize, equal_rep):

        num_0 = int(num_samples / 2)
        num_1 = num_samples - num_0

        if self.randomize:
            df_new = self.df[self.df.prev == False]
            if self.equal_rep:
                df_new_0 = df_new[df_new.y == 0]
                df_new_1 = df_new[df_new.y == 1]
                idx_0 = np.random.choice(df_new_0.index, num_0, replace=False)
                idx_1 = np.random.choice(df_new_1.index, num_1, replace=False)
                idx = np.concatenate((idx_0, idx_1), axis=0)
                idx = shuffle(idx, random_state=self.seed)
            else:
                idx = np.random.choice(df_new.index, num_samples, replace=False)

            idx = pd.Index(idx)

        else:
            start = self.current_pos
            stop = min(start + num_samples, self.df.shape[0])
            idx = self.df.index[start:stop]
            dn = num_samples - len(idx)
            dn_0 = 0
            dn_1 = 0
            if self.equal_rep:
                dfi = self.df.loc[idx]
                n_0 = len(dfi[dfi.y == 0])
                n_1 = len(dfi[dfi.y == 1])
                dn_0 = num_0 - n_0
                dn_1 = num_1 - n_1

            while (dn > 0) or (dn_0 > 0) or (dn_1 > 0):

                start = stop % self.df.shape[0]

                if self.equal_rep:
                    if self.posfrac < 1.:
                        dn_0_norm = int(float(dn_0) / (1. - self.posfrac))
                    else:
                        dn_0_norm = 0

                    if self.posfrac > 0.:
                        dn_1_norm = int(float(dn_1) / self.posfrac)
                    else:
                        dn_1_norm = 0

                    stop = max(dn, max(dn_0_norm, dn_1_norm))
                else:
                    stop = dn

                stop += start
                stop = min(stop, self.df.shape[0])

                idx_add = self.df.index[start:stop].values
                idx = np.concatenate((idx.values, idx_add), axis=0)
                idx = pd.Index(idx)
                dn = num_samples - len(idx)

                if self.equal_rep:
                    dfi = self.df.loc[idx]
                    n_0 = len(dfi[dfi.y == 0])
                    n_1 = len(dfi[dfi.y == 1])
                    dn_0 = num_0 - n_0
                    dn_1 = num_1 - n_1

            if self.equal_rep:
                dfi = self.df.loc[idx]
                idx_0 = dfi[dfi.y == 0].index
                idx_1 = dfi[dfi.y == 1].index
                idx_0 = np.random.choice(idx_0, num_0, replace=False)
                idx_1 = np.random.choice(idx_1, num_1, replace=False)
                idx = np.concatenate((idx_0, idx_1), axis=0)
                idx = shuffle(idx, random_state=self.seed)
                idx = pd.Index(idx)

            idx = idx.drop_duplicates()

            self.current_pos = stop

        return idx
