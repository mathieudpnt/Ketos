""" Neural networks module within the ketos library

    This module provides utilities to work with Neural Networks.

    Contents:
        DataHandler class:
        DataUse class:

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
from enum import Enum
from ketos.data_handling.data_handling import check_data_sanity, to1hot


def class_confidences(class_weights):
    """ Compute the classification confidence from classification weights.

        Confidence is computed as the difference between the largest class weight 
        and the second largest class weight.

        Args:
            class_weights: numpy array
                Classification weights

        Returns:
            conf: numpy array
                Confidence level

        Example:

            >>> from ketos.neural_networks.neural_networks import class_confidences
            >>> weights = [0.2, 0.55, 0.25]
            >>> conf = class_confidences(weights)
            >>> print('{:.2f}'.format(conf))
            0.30
    """
    w = class_weights

    if type(w) is not np.ndarray:
        w = np.array(class_weights)
        w = np.squeeze(w)

    if np.ndim(w) == 1:
        w = w[np.newaxis, :]

    idx = np.argsort(w, axis=1)
    w0 = np.choose(idx[:,-1], w.T) # max weights
    w1 = np.choose(idx[:,-2], w.T) # second largest weights
    conf = w0 - w1 # classification confidence

    if len(conf) == 1:
        conf = conf[0]

    return conf


def predictions(class_weights):
    """ Compute predicted labels from classification weights.

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

