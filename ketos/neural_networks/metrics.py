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

""" metrics sub-module within the ketos.neural_networks module

    This module provides convenience wrappers around metrics that are commonly used as secondary metrics by ketos' NNInterface. 

    Contents:
        FScore class
        Accuracy class
        Precision class
        Recall class
"""



from sklearn.metrics import  accuracy_score, recall_score, precision_score, average_precision_score, fbeta_score
import numpy as np


class FScore():
    """ FScore metric.

        When instantiated, the resulting metric function expects the predictions in the 'y_pred' argument and the true labels in the
        'y_true' argument.

        Args:
            beta:float
                The relative weight of recall in relation to precision.
                 Examples:
                    If beta = 1.0, recall has same weight as precision
                    If beta = 0.5, recall has half the weight of precision
                    If beta = 2.0, recall has twice the weight of precision
            onehot:bool
                Whether or not y_pred and y_true are one-hot encoded.
    """
    def __init__(self, onehot=True, beta=1.0):
        self.onehot = onehot
        self.beta = beta
    def __call__(self, y_true, y_pred):
        if self.onehot:
            y_true = np.argmax(y_true, axis=1)
            y_pred = np.argmax(y_pred, axis=1)

        epsilon = 0.000001
    
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)

        f_score = (1.0 + self.beta**2)*p*r / ((self.beta**2*p)+r+epsilon)
        
        return f_score 



class Accuracy():
    """ Accuracy metric.

        When instantiated, the resulting metric function expects the predictions in the 'y_pred' argument and the true labels in the
        'y_true' argument.

        Args:
            onehot:bool
                Whether or not y_pred and y_true are one-hot encoded.
    """
    def __init__(self, onehot=True):
        self.func = accuracy_score
        self.onehot = onehot
    def __call__(self, y_true, y_pred):
        if self.onehot:
            y_true = np.argmax(y_true, axis=1)
            y_pred = np.argmax(y_pred, axis=1)
        return self.func(y_true, y_pred)
      


class Precision():
    """ Precision metric.

        When instantiated, the resulting metric function expects the predictions in the 'y_pred' argument and the true labels in the
        'y_true' argument.

        Args:
            onehot:bool
                Whether or not y_pred and y_true are one-hot encoded.
    """
    def __init__(self, onehot=True):
        self.func = precision_score
        self.onehot = onehot
    def __call__(self, y_true, y_pred):
        if self.onehot:
            y_true = np.argmax(y_true, axis=1)
            y_pred = np.argmax(y_pred, axis=1)
        return self.func(y_true, y_pred, zero_division=0)
       


class Recall():
    """ Recall metric.

        When instantiated, the resulting metric function expects the predictions in the 'y_pred' argument and the true labels in the
        'y_true' argument.

        Args:
            onehot:bool
                Whether or not y_pred and y_true are one-hot encoded.
    """

    def __init__(self, onehot=True):
        self.func = recall_score
        self.onehot = onehot
    def __call__(self, y_true, y_pred):
        if self.onehot:
            y_true = np.argmax(y_true, axis=1)
            y_pred = np.argmax(y_pred, axis=1)
        return self.func(y_true, y_pred, zero_division=0)




