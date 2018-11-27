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
from sound_classification.neural_networks import DataHandler


def channel_normalization(x):
    # Normalize by the highest activation
    max_values = tf.keras.backend.max(tf.keras.backend.abs(x), 2, keepdims=True) + 1e-5
    out = x / max_values
    return out


class EDTCN(DataHandler):
    """ Create an Encoder-Decoder Temporal Convolutional Network (ED-TCN) for classification tasks.

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
            num_labels: int
                Number of labels
            batch_size: int
                The number of examples in each batch
            num_epochs: int
                The number of epochs
    """

    def __init__(self, train_x, train_y, validation_x=None, validation_y=None,
                 test_x=None, test_y=None, num_labels=2, batch_size=4, 
                 num_epochs=100):

        self.num_labels = num_labels
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        super(EDTCN, self).__init__(train_x=train_x, train_y=train_y, 
                validation_x=validation_x, validation_y=validation_y,
                test_x=test_x, test_y=test_y)
        
    def create(self, n_nodes=[16, 32], conv_len=16, max_len=100):
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

        n_feat = self.get_training_data().shape[1]
        n_layers = len(n_nodes)
        n_classes = self.num_labels

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
            model = tf.keras.layers.SpatialDropout1D(rate=dropout_rate)(model)
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