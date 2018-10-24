""" Training data sampler module within the sound_classification package

Authors: Fabio Frazao and Oliver Kirsebom
    contact: fsfrazao@dal.ca, oliver.kirsebom@dal.ca
    Organization: MERIDIAN
    Team: Acoustic data Analytics, Dalhousie University
    Project: packages/sound_classification
             Project goal: To package code useful for handling data, deriving features and 
             creating Deep Neural Networks for sound classification projects.
     
    License:

"""

import numpy as np
import pandas as pd


class TrainingDataSampler():
    """ Training data sampler.

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
    """
    def __init__(self, x, y, randomize=False, num_samples=100, max_keep=0, conf_cut=0):

        N = x.shape[0]
        self.df = pd.DataFrame({'x':x, 'y':y, 'pred':np.empty(N), 'conf':np.empty(N), 'prev':np.zeros(N, dtype=bool)})
        self.randomize = randomize
        self.num_samples = num_samples
        self.max_keep = max_keep
        self.conf_cut = conf_cut
        self.it = 0

    def get_samples(self, num_samples=None, max_keep=None, conf_cut=None):

        # use default value if none provided
        if num_samples is None:
            num_samples = self.num_samples
        if max_keep is None:
            max_keep = self.max_keep
        if conf_cut is None:
            conf_cut = self.conf_cut

        # get new samples
        idx_new = self.get_new(num_samples=num_samples, randomize=self.randomize)

        # get all poorly performing samples from previous iteration
        idx_poor = self.get_poor(max_keep=max_keep, conf_cut=conf_cut)
        x_poor, y_poor = list(), list()
        if len(idx_poor) > 0:
            x_poor = self.df[idx_poor]['x']
            y_poor = self.df[idx_poor]['y']

        x = x_poor
        y = y_poor 
        return x, y


    def get_poor(self, max_keep, conf_cut):
        df_prev = self.df[self.df.prev == True]
        if df_prev.shape[0] == 0:
            return list()
        df_poor = df_prev[(df_prev.pred != df_prev.y) | (df_prev.conf < self.conf_cut)]
        idx = np.random.choice(df_poor.index, np.ceil(N/2.), replace=False)
        return idx

    def get_new(self, num_samples, randomize):
        df_new = self.df[self.df.prev == False]
        if self.randomize:
            idx = np.random.choice(df_new.index, num_samples, replace=False)
        else:
            start = self.it
            stop = min(self.it + num_samples, df_new.shape[0])
            idx = np.arange(start=start, stop=stop)
        x = np.take(x_all, indices=idx, axis=0)
        y = np.take(y_all, indices=idx, axis=0)
