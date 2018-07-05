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


class CNNWhale():

    def __init__(self, train_x, train_y, validation_x, validation_y,
                 test_x, test_y, batch_size, num_channels, num_labels,
                 input_shape, learning_rate=0.01, num_epochs=10, seed=42):
        self.train_x = train_x
        self.train_y = train_y
        self.validation_x = validation_x
        self.validation_y = validation_y
        self.test_x = test_x
        self.test_y = test_y
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.num_labels = num_labels
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.seed = seed
        self.train_size = self.train_y.shape[0]



def create_net_structure(self):
        train_data_node = tf.placeholder(tf.float32, shape=(self.batch_size,
                                         self.input_shape[0], self.input_shape[1], 1))
        train_labels_node = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_labels))
        validation_data_node = tf.constant(self.validation_x)
        test_data_node = tf.constant(self.test_x)

        x = tf.placeholder(tf.float32, [None, self.input_shape[0] * self.input_shape[1]])
        x_shaped = tf.reshape(x, [-1, self.input_shape[0], self.input_shape[1], 1])
        y = tf.placeholder(tf.float32, [None, self.num_labels])


        layer1 = self.create_new_conv_layer(x_shaped, 1, 32, [2, 8], [2, 2], name='layer1')
        layer2 = self.create_new_conv_layer(layer1, 32, 64, [30, 8], [2, 2], name='layer2')

        flattened = tf.reshape(layer2, [-1, 5 * 15 * 64])

        # setup some weights and bias values for this layer, then activate with ReLU
        wd1 = tf.Variable(tf.truncated_normal([5* 15 * 64, 512], stddev=0.03), name='wd1')
        bd1 = tf.Variable(tf.truncated_normal([512], stddev=0.01), name='bd1')
        dense_layer1 = tf.matmul(flattened, wd1) + bd1
        dense_layer1 = tf.nn.relu(dense_layer1)

        # another layer with softmax activations
        wd2 = tf.Variable(tf.truncated_normal([512,self.num_labels], stddev=0.03), name='wd2')
        bd2 = tf.Variable(tf.truncated_normal([self.num_labels], stddev=0.01), name='bd2')
        dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
        y_ = tf.nn.softmax(dense_layer2)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))

        # add an optimiser
        optimiser = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cross_entropy)

        # define an accuracy assessment operation
        predict = tf.argmax(y_, 1, name="predict")
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1), name="correct_pred")
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="accuracy")

        # setup the initialisation operator
        init_op = tf.global_variables_initializer()

        # setup recording variables
        # add a summary to store the accuracy
        tf.summary.scalar('accuracy', accuracy)

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('summaries')
        saver = tf.train.Saver()

        return {'x': x,
                'y':y,            
                'cost_function': cross_entropy,
                'optimiser': optimiser,
                'predict': predict,
                'correct_prediction': correct_prediction,
                'accuracy': accuracy,
                'init_op': init_op,
                'merged':  merged,
                'writer': writer,
                'saver': saver,
                }
