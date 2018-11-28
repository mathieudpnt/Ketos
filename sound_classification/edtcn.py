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
from sound_classification.data_handling import to1hot
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
                 num_epochs=100, keep_prob=0.7, verbosity=0):

        self.num_labels = num_labels
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.keep_prob = keep_prob
        self.verbosity = verbosity
        self.max_len = 500

        self.model = None

        super(EDTCN, self).__init__(train_x=train_x, train_y=train_y, 
                validation_x=validation_x, validation_y=validation_y,
                test_x=test_x, test_y=test_y)

    def set_verbosity(self, verbosity):
        """Set verbosity level.
            0: no messages
            1: warnings only
            2: warnings and diagnostics

            Args:
                verbosity: int
                    Verbosity level.        
        """
        self.verbosity = verbosity

    def create(self, n_nodes=[16, 32], conv_len=16, max_len=500):
        """Create the Neural Network structure.

            Args:
                n_nodes: tuple(int)
                    Number of output filters in each 1D convolutional layer.
                    (The number of convolutional layers is given by the length of n_nodes)
                conv_len: int
                    Length of the 1D convolution window.
                max_len: int
                    The input data will be split into chuncks of size max_len. 
                    Thus, max_len effectively limits the extent of the memory of the network. 
        """
        dropout_rate = 1.0 - self.keep_prob

        self.max_len = max_len

        if self.max_len % 4 > 0:
            self.max_len -= self.max_len % 4
            if self.verbosity >= 1: 
                print(' Warning: max_len must be divisible by 4; max_len has been adjust to {0}'.format(self.max_len))

        x, _ = self.get_training_data()
        n_feat = x.shape[1]
        n_layers = len(n_nodes)
        n_classes = self.num_labels

        # --- Input layer ---
        inputs = tf.keras.layers.Input(shape=(self.max_len, n_feat))
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

        self.model = model

    def train(self, batch_size=None, num_epochs=None):

        if batch_size is None:
            batch_size = self.batch_size
        if num_epochs is None:
            num_epochs = self.num_epochs

        x, y = self.get_training_data()
        x_val, y_val = self.get_validation_data()

        y = to1hot(y, self.num_labels)
        y_val = to1hot(y_val, self.num_labels)

        x = split(x, self.max_len)
        y = split(y, self.max_len)
        x_val = split(x_val, self.max_len)
        y_val = split(y_val, self.max_len)

        history = self.model.fit(x=x, y=y, batch_size=batch_size, epochs=num_epochs, verbose=self.verbosity, validation_data=(x_val, y_val))        

def split(a, n):
    """Split the data into chunks with certain size.
        
        Args:
            a: numpy array
                Array containing the data to be split.
            
            n: int
                Chunk size.
    """
    orig_len = a.shape[0]
    nsegs = int(np.ceil(orig_len / n))
    new_len = nsegs * n
    pad_shape = np.array([new_len - orig_len], dtype=np.int32)
    pad_shape = np.append(pad_shape, a.shape[1:])
    a = np.append(a, np.zeros(shape=pad_shape), axis=0)
    new_shape = np.array([nsegs, n], dtype=np.int32)
    new_shape = np.append(new_shape, a.shape[1:])
    a = np.reshape(a=a, newshape=new_shape)
    return a