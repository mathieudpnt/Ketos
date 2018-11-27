""" Encoder-Decoder Temporal Convolutional Network (ED-TDCN).

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
import sound_classification.data_handling as dh
from sound_classification.neural_networks import MNet


def channel_normalization(x):
    # Normalize by the highest activation
    max_values = tf.keras.backend.max(tf.keras.backend.abs(x), 2, keepdims=True) + 1e-5
    out = x / max_values
    return out


class EDTCN(MNet):
    """ Create an Encoder-Decoder Temporal Convolutional Network (ED-TCN) for classification tasks.
    """

    def __init__(self, train_x, train_y, validation_x=None, validation_y=None,
                 test_x=None, test_y=None, num_labels=2, batch_size=4, 
                 num_epochs=100, learning_rate=0.01, keep_prob=0.7, seed=42, verbosity=2):

        super(EDTCN, self).__init__(train_x=train_x, train_y=train_y, 
                validation_x=validation_x, validation_y=validation_y,
                test_x=test_x, test_y=test_y, num_labels=num_labels, 
                batch_size=batch_size, num_epochs=num_epochs, 
                learning_rate=learning_rate, keep_prob=keep_prob, 
                seed=seed, verbosity=verbosity)
        
    def _create_net_structure(self, input_shape, num_labels, **kwargs):
        """Create the Neural Network structure.

            Args:
                n_nodes: int tuple
                    bla ...
                conv_len: int
                    bla ...
                max_len: int
                    bla ...

            Returns:
                tf_nodes: dict
                    A dictionary with the tensorflow objects necessary
                    to train and run the model.
                    sess, x, y, cost_function, optimizer, predict, correct_prediction,
                    accuracy,init_op, merged, writer, saver
                    These objects are stored as
                    instance attributes when the class is instantiated.

        """
        dropout_rate = 1.0 - self.keep_prob

        # default configuration:
        n_nodes = [16, 32]
        conv_len = 16
        max_len = 100

        for a in kwargs['kwargs']:
            if a == 'n_nodes': n_nodes = kwargs['kwargs'][a]
            elif a == 'conv_len': conv_len = kwargs['kwargs'][a]
            elif a == 'max_len': max_len = kwargs['kwargs'][a]

        n_feat = input_shape.shape[0]
        n_layers = len(n_nodes)
        n_classes = num_labels

        # --- Input layer ---
        inputs = tf.keras.layers.Input(shape=(max_len, n_feat))
        model = inputs

        # ---- Encoder ----
        for i in range(n_layers):
            model = tf.keras.layers.Convolution1D(n_nodes[i], conv_len, padding='same')(model) # 1D convolutional layer
            model = tf.keras.layers.SpatialDropout1D(rate=dropout_rate)(model) # Spatial 1D version of Dropout
            model = tf.keras.layers.Activation('relu')(model) # apply activation
            model = tf.keras.layers.Lambda(channel_normalization, name="encoder_norm_{}".format(i))(model)
            model = tf.keras.layers.MaxPooling1D(2)(model) # max pooling

        # ---- Decoder ----
        for i in range(n_layers):
            model = tf.keras.layers.UpSampling1D(2)(model)
            model = tf.keras.layers.Convolution1D(n_nodes[-i-1], conv_len, padding='same')(model)
            model = tf.keras.layers.SpatialDropout1D(0.3)(model)
            model = tf.keras.layers.Activation('relu')(model)
            model = tf.keras.layers.Lambda(channel_normalization, name="decoder_norm_{}".format(i))(model)

        # Output fully connected layer
        model = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=n_classes, activation="softmax" ))(model)

        # create and compile model
        model = tf.keras.models.Model(inputs=inputs, outputs=model)
        model.compile(loss='categorical_crossentropy', optimizer="rmsprop", sample_weight_mode="temporal", metrics=['accuracy'])

        tf_nodes = {'x': model.layers[0].output
            }

        return tf_nodes