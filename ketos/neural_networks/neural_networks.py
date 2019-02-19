""" Neural Networks module within the sound_classification package

This module includes classes and functions useful for creating 
Deep Neural Networks applied to sound classification within MERIDIAN.

Authors: Fabio Frazao and Oliver Kirsebom
    contact: fsfrazao@dal.ca, oliver.kirsebom@dal.ca
    Organization: MERIDIAN
    Team: Acoustic data Analytics, Dalhousie University
    Project: packages/sound_classification
             Project goal: To package code useful for handling data, deriving features and 
             creating Deep Neural Networks for sound classification projects.
     
    License:

"""
import tensorflow as tf
import numpy as np
import pandas as pd
from enum import Enum
from ketos.data_handling.data_handling import check_data_sanity, to1hot


def class_confidences(class_weights):
    """Compute the classification confidence from classification weights.

        Confidence is computed as the difference between the largest class weight 
        and the second largest class weight. E.g. if there are three classes and 
        the neural network assigns weights 0.2, 0.55, 0.25, the confidence is 0.55-0.25=0.25.

        Args:
            class_weights: numpy array
                Classification weights

        Returns:
            conf: numpy array
                Confidence level
    """
    idx = np.argsort(class_weights, axis=1)
    w0 = np.choose(idx[:,-1], class_weights.T) # max weights
    w1 = np.choose(idx[:,-2], class_weights.T) # second largest weights
    conf = w0 - w1 # classification confidence
    return conf


def predictions(class_weights):
    """Compute predicted labels from classification weights.

        Args:
            class_weights: numpy array
                Classification weights

        Returns:
            p: numpy array
                Predicted labels
    """
    p = np.argmax(class_weights, axis=1)
    return p     
    
    
class DataUse(Enum):
    TRAINING = 1
    VALIDATION = 2
    TEST = 3


class DataHandler():
    """ Parent class for all MERIDIAN machine-learning models.

        Args:
            train_x: pandas DataFrame
                Data Frame in which each row holds one image.
            train_y: pandas DataFrame
                Data Frame in which each row contains the one hot encoded label
            validation_x: pandas DataFrame
                Data Frame in which each row holds one image
            validation_y: pandas DataFrame
                Data Frame in which each row contains the one hot encoded label
            test_x: pandas DataFrame
                Data Frame in which each row holds one image
            test_y: pandas DataFrame
                Data Frame in which each row contains the one hot encoded label
    """
    def __init__(self, train_x, train_y, validation_x=None, validation_y=None,
                 test_x=None, test_y=None, num_labels=None):

        self.num_labels = num_labels    

        self.images = {DataUse.TRAINING: None, DataUse.VALIDATION: None, DataUse.TEST: None}
        self.labels = {DataUse.TRAINING: None, DataUse.VALIDATION: None, DataUse.TEST: None}

        self._set_data(train_x, train_y, use=DataUse.TRAINING)        
        self._set_data(validation_x, validation_y, use=DataUse.VALIDATION)        
        self._set_data(test_x, test_y, use=DataUse.TEST)    

    def _set_data(self, x, y, use):
        """ Set data for specified use (training, validation, or test). 
            Replaces any existing data for that use type.

            Args:
                x: pandas DataFrame
                    Data Frame in which each row holds one image. 
                y: pandas DataFrame
                    Data Frame in which each row contains the one hot encoded label
                use: DataUse
                    Data use. Possible options are TRAINING, VALIDATION and TEST
        """
        check_data_sanity(x, y)

        if np.ndim(x) == 3:
            x = x[:,:,:,np.newaxis]

        if np.ndim(y) == 1:
            if self.num_labels is None:
                depth = np.max(y) + 1 # figure it out from the data
            else:
                depth = self.num_labels

            y = to1hot(y, depth=depth) # use one-hot encoding

        self.images[use] = x
        self.labels[use] = y

    def _add_data(self, x, y, use):
        """ Add data for specified use (training, validation, or test). 
            Will be appended to any existing data for that use type.

            Args:
                x: pandas DataFrame
                    Data Frame in which each row holds one image. 
                y: pandas DataFrame
                    Data Frame in which each row contains the one hot encoded label
                use: DataUse
                    Data use. Possible options are TRAINING, VALIDATION and TEST
        """
        x0 = self.images[use]
        y0 = self.labels[use]
        if x0 is not None:
            x = np.append(x0, x, axis=0)
        if y0 is not None:
            y = np.append(y0, y, axis=0)
        self._set_data(x=x, y=y, use=use)

    def _get_data(self, use):
        return self.images[use], self.labels[use]

    def set_training_data(self, x, y):
        """ Set training data. Replaces any existing training data.

            Args:
                x: pandas DataFrame
                    Data Frame in which each row holds one image. 
                y: pandas DataFrame
                    Data Frame in which each row contains the one hot encoded label
        """
        self._set_data(x=x, y=y, use=DataUse.TRAINING)

    def add_training_data(self, x, y):
        """ Add training data. Will be appended to any existing training data.

            Args:
                x: pandas DataFrame
                    Data Frame in which each row holds one image. 
                y: pandas DataFrame
                    Data Frame in which each row contains the one hot encoded label
        """
        self._add_data(x=x, y=y, use=DataUse.TRAINING)

    def get_training_data(self):
        """ Get training data.
        """
        return self._get_data(use=DataUse.TRAINING)

    def set_validation_data(self, x, y):
        """ Set validation data. Replaces any existing validation data.

            Args:
                x: pandas DataFrame
                    Data Frame in which each row holds one image. 
                y: pandas DataFrame
                    Data Frame in which each row contains the one hot encoded label
        """
        self._set_data(x=x, y=y, use=DataUse.VALIDATION)

    def add_validation_data(self, x, y):
        """ Add validation data. Will be appended to any existing validation data.

            Args:
                x: pandas DataFrame
                    Data Frame in which each row holds one image. 
                y: pandas DataFrame
                    Data Frame in which each row contains the one hot encoded label
        """
        self._add_data(x=x, y=y, use=DataUse.VALIDATION)

    def get_validation_data(self):
        """ Get validation data.
        """
        return self._get_data(use=DataUse.VALIDATION)

    def set_test_data(self, x, y):
        """ Set test data. Replaces any existing test data.

            Args:
                x: pandas DataFrame
                    Data Frame in which each row holds one image. 
                y: pandas DataFrame
                    Data Frame in which each row contains the one hot encoded label
        """
        self._set_data(x=x, y=y, use=DataUse.TEST)

    def add_test_data(self, x, y):
        """ Add test data. Will be appended to any existing test data.

            Args:
                x: pandas DataFrame
                    Data Frame in which each row holds one image. 
                y: pandas DataFrame
                    Data Frame in which each row contains the one hot encoded label
        """
        self._add_data(x=x, y=y, use=DataUse.TEST)

    def get_test_data(self):
        """ Get test data.
        """
        return self._get_data(use=DataUse.TEST)

