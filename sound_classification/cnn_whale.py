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
from collections import namedtuple
import sound_classification.data_handling as dh
from sound_classification.neural_networks import MNet


ConvParams = namedtuple('ConvParams', 'name n_filters filter_shape')
ConvParams.__doc__ = '''\
Name and dimensions of convolutional layer in neural network

name - Name of convolutional layer, e.g. "conv_layer"
n_filters - Number of filters, e.g. 16
filter_shape - Filter shape, e.g. [4,4]'''



class CNNWhale(MNet):
    """ Create a Convolutional Neural Network for classification tasks.
    """

    def __init__(self, train_x, train_y, validation_x=None, validation_y=None,
                 test_x=None, test_y=None, num_labels=2, batch_size=128, 
                 num_epochs=10, learning_rate=0.01, keep_prob=1.0, seed=42, verbosity=2):

        super(CNNWhale, self).__init__(train_x=train_x, train_y=train_y, 
                validation_x=validation_x, validation_y=validation_y,
                test_x=test_x, test_y=test_y, num_labels=num_labels, 
                batch_size=batch_size, num_epochs=num_epochs, 
                learning_rate=learning_rate, keep_prob=keep_prob, 
                seed=seed, verbosity=verbosity)
        
    @classmethod
    def from_prepared_data(cls, prepared_data, num_labels=2, batch_size=128,
                num_epochs=10, learning_rate=0.01, keep_prob=1.0, seed=42, verbosity=2):

        train_x = prepared_data["train_x"]
        train_y = prepared_data["train_y"]
        validation_x = prepared_data["validation_x"]
        validation_y = prepared_data["validation_y"]
        test_x = prepared_data["test_x"]
        test_y = prepared_data["test_y"]

        return cls(train_x=train_x, train_y=train_y, 
                validation_x=validation_x, validation_y=validation_y,
                test_x=test_x, test_y=test_y, num_labels=num_labels, 
                batch_size=batch_size, num_epochs=num_epochs, 
                learning_rate=learning_rate, keep_prob=keep_prob, 
                seed=seed, verbosity=verbosity)

    def _create_net_structure(self, input_shape, num_labels, **kwargs):
        """Create the Neural Network structure.

            The Network has a number of convolutional layers followed by a number 
            of fully connected layers with ReLU activation functions and a final 
            output layer with softmax activation.
            
            The default network structure has two convolutional layers 
            and one fully connected layers with ReLU activation.

            Args:
                conv_params: list(ConvParams)
                    Configuration parameters for the convolutional layers.
                    Each item in the list represents a convolutional layer.
                dense_size: list(int)
                    Sizes of the fully connected layers preceeding the output layer.

            Returns:
                tf_nodes: dict
                    A dictionary with the tensorflow objects necessary
                    to train and run the model.
                    sess, x, y, cost_function, optimizer, predict, correct_prediction,
                    accuracy,init_op, merged, writer, saver
                    These objects are stored as
                    instance attributes when the class is instantiated.

        """
        # default configuration:
        layer1 = ConvParams(name='conv_1',n_filters=32,filter_shape=[2,8])
        layer2 = ConvParams(name='conv_2',n_filters=64,filter_shape=[30,8])
        conv_params = [layer1, layer2] 
        dense_size = [512]

        for a in kwargs['kwargs']:
            if a == 'conv_params':
                conv_params = kwargs['kwargs'][a]
            elif a == 'dense_size':
                dense_size = kwargs['kwargs'][a]

        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        x = tf.placeholder(tf.float32, [None, input_shape[0] * input_shape[1]], name="x")
        x_shaped = tf.reshape(x, [-1, input_shape[0], input_shape[1], 1])
        y = tf.placeholder(tf.float32, [None, num_labels], name="y")

        pool_shape=[2,2]

        # input and convolutional layers
        params = [ConvParams(name='input', n_filters=1, filter_shape=[1,1])] # input layer with dimension 1
        params.extend(conv_params)
            
        # dense layers including output layer
        dense_size.append(num_labels)

        # create convolutional layers
        conv_layers = [x_shaped]
        N = len(params)
        conv_summary = list()
        for i in range(1,N):
            # previous layer
            l_prev = conv_layers[len(conv_layers)-1]
            # layer parameters
            n_input = params[i-1].n_filters
            n_filters = params[i].n_filters
            filter_shape = params[i].filter_shape
            name = params[i].name
            # create new layer
            l = self.create_new_conv_layer(l_prev, n_input, n_filters, filter_shape, pool_shape, name=name)
            conv_layers.append(l)
            # collect info
            dim = l.shape[1] * l.shape[2] * l.shape[3]
            conv_summary.append("  {0}       {1} x {2}          [{3},{4}]         {5}".format(name, n_input, n_filters, filter_shape[0], filter_shape[1], dim))
            # apply DropOut 
            drop_out = tf.nn.dropout(l, keep_prob)  # DROP-OUT here
            conv_layers.append(drop_out)

        # last layer
        last = conv_layers[-1]

        # flatten
        dim = last.shape[1] * last.shape[2] * last.shape[3]
        flattened = tf.reshape(last, [-1, dim])

        # fully-connected layers with ReLu activation
        dense_layers = [flattened]
        dense_summary = list()
        for i in range(len(dense_size)):
            size = dense_size[i] 
            l_prev = dense_layers[i]
            w_name = 'w_{0}'.format(i+1)
            w = tf.Variable(tf.truncated_normal([int(l_prev.shape[1]), size], stddev=0.03), name=w_name)
            b_name = 'b_{0}'.format(i+1)
            b = tf.Variable(tf.truncated_normal([size], stddev=0.01), name=b_name)
            l = tf.matmul(l_prev, w) + b
            if i < len(dense_size) - 1:
                n = 'dense_{0}'.format(i+1)
                l = tf.nn.relu(l, name=n) # ReLu activation
            else: # output layer
                cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=l, labels=y),name="cost_function")
                n = 'class_weights'
                l = tf.nn.softmax(l, name=n) # softmax                    

            dense_layers.append(l)
            dense_summary.append("  {0}    {1}".format(n, size))

        if self.verbosity >= 2:
            print('\n')
            print('======================================================')
            print('                   Convolutional layers               ')
            print('------------------------------------------------------')
            print('  Name   Input x Filters   Filter Shape   Output dim. ')
            print('------------------------------------------------------')
            for line in conv_summary:
                print(line)
            print('======================================================')
            print('                  Fully connected layers              ')
            print('------------------------------------------------------')
            print('  Name       Size                                      ')
            print('------------------------------------------------------')
            for line in dense_summary:
                print(line)
            print('======================================================')

        # output layer
        y_ = dense_layers[-1]

        # add an optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,name = "optimizer").minimize(cross_entropy)

        # define an accuracy assessment operation
        predict = tf.argmax(y_, 1, name="predict")
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1), name="correct_prediction")
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="accuracy")

        # setup the initialisation operator
        init_op = tf.global_variables_initializer()

        # setup recording variables
        # add a summary to store the accuracy
        tf.summary.scalar('accuracy', accuracy)
        merged = tf.summary.merge_all()

        writer = tf.summary.FileWriter('summaries')
        saver = tf.train.Saver()

        tf_nodes = {'x': x,
                'y':y,            
                'cost_function': cross_entropy,
                'optimizer': optimizer,
                'predict': predict,
                'correct_prediction': correct_prediction,
                'accuracy': accuracy,
                'init_op': init_op,
                'merged': merged,
                'writer': writer,
                'saver': saver,
                'keep_prob': keep_prob,
                'learning_rate': learning_rate,
                'class_weights': y_,
                }

        return tf_nodes
        
    def create_new_conv_layer(self, input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
        """Create a convolutional layer.

            Args:
                input_data: tensorflow tensor
                    The input nodes for the convolutional layer
                num_input_channels: int
                    The number of input channels in input image
                num_filters: int
                    Number of filters to be used in the convolution
                filter_shape: list (int)
                    List of integers defining the shape of the filters.
                    Example: [2,8]
                pool_shape: list (int)
                    List of integers defining the shape of the pooling window.
                    Example: [2,8]
                name: str
                    Name by which the layer will be identified in the graph.

            Returns:
                out_layer: tensorflow layer
                    The convolutional layer.

        """
        # setup the filter input shape for tf.nn.conv_2d
        conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

        # initialise weights and bias for the filter
        weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name+'_W')
        bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

        # setup the convolutional layer operation
        out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

        # add the bias
        out_layer += bias

        # apply a ReLU non-linear activation
        out_layer = tf.nn.relu(out_layer)

        # now perform max pooling
        # ksize is the argument which defines the size of the max pooling window (i.e. the area over which the maximum is
        # calculated).  It must be 4D to match the convolution - in this case, for each image we want to use a 2 x 2 area
        # applied to each channel
        ksize = [1, pool_shape[0], pool_shape[1], 1]
        # strides defines how the max pooling area moves through the image - a stride of 2 in the x direction will lead to
        # max pooling areas starting at x=0, x=2, x=4 etc. through your image.  If the stride is 1, we will get max pooling
        # overlapping previous max pooling areas (and no reduction in the number of parameters).  In this case, we want
        # to do strides of 2 in the x and y directions.
        strides = [1, 2, 2, 1]
        out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')
###oli        out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='VALID')

        return out_layer