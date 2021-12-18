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

""" losses sub-module within the ketos.neural_networks module

    This module provides loss functions

    Contents:
        FScoreLoss class:
"""

import tensorflow as tf
import numpy as np

class FScoreLoss(tf.keras.losses.Loss):
    """ Loss function based on the inverse of F-Score.

        When instantiated, the resulting loss function expects the predictions in the 'y_pred' argument and the true labels in the
        'y_true' argument.

        Args:
            beta:float
                The relative weight of recall in relation to precision.

                Examples:
                    If beta = 1.0, recall has same weight as precision
                    If beta = 0.5, recall has half the weight of precision
                    If beta = 2.0, recall has twice the weight of precision 
    """

    def __init__(self, beta=1.0, **kwargs):
        super(FScoreLoss, self).__init__(**kwargs)
        self.beta = beta

    def call(self, y_true, y_pred):
        """ Invokes the Loss instance.

        Args:
            y_true: Ground truth values. 
                shape = [batch_size, d0, .. dN], except sparse loss functions such as sparse categorical crossentropy where shape = [batch_size, d0, .. dN-1]

            y_pred: The predicted values. 
                shape = [batch_size, d0, .. dN]

        Returns:
            Loss values with the shape [batch_size, d0, .. dN-1].
        """
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.dtypes.cast(y_true, y_pred.dtype)

        epsilon = 0.000001

        tp = tf.math.reduce_sum(y_true*y_pred)
        tn = tf.math.reduce_sum((1-y_true)*(1-y_pred))
        fp = tf.math.reduce_sum((1-y_true)*y_pred)
        fn = tf.math.reduce_sum(y_true*(1-y_pred))

        p = tp / (tp + fp + epsilon)
        r = tp / (tp + fn + epsilon)

        f = (1.0 + self.beta**2)*p*r / ((self.beta**2*p)+r+epsilon)
        f = tf.where(tf.math.is_nan(f), tf.zeros_like(f), f)
        return 1 - tf.math.reduce_mean(f)
