""" Training data provider module within the sound_classification package

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
    """
    def __init__(self, x, y, randomize=False, num_samples=100, max_keep=0, conf_cut=0):

        N = x.shape[0]
        self.df = pd.DataFrame({'x':x, 'y':y, 'pred':y, 'conf':np.ones(N), 'prev':np.zeros(N, dtype=bool)})
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

        x, y = None, None

        # get all poorly performing samples from previous iteration
        idx_poor = self._get_poor(num=int(np.ceil(num_samples * max_keep)), conf_cut=conf_cut)
        num_poor = len(idx_poor)

        # get new samples
        idx = self._get_new(num_samples=num_samples - num_poor, randomize=self.randomize)
        if num_poor > 0:
            idx = idx.union(idx_poor)

        # combine poorly performing and new samples
        df = self.df.loc[idx]
        x = df.x.values
        y = df.y.values

        # internal book keeping
        self.df.prev = False
        self.df.loc[idx,'prev'] = True

        return x, y

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

    def _get_new(self, num_samples, randomize):
        if self.randomize:
            df_new = self.df[self.df.prev == False]
            idx = np.random.choice(df_new.index, num_samples, replace=False)
        else:
            df_prev = self.df[self.df.prev == True]
            if len(df_prev) == 0:
                start = 0
            else:
                start = df_prev.index[-1] + 1
            if start >= self.df.shape[0]:
                start = 0
            stop = min(start + num_samples, self.df.shape[0])
            idx = self.df.index[start:stop]
        return idx