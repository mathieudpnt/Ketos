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

        tf_operations = self.create_net_structure()

        self.sess = tf.Session()
        self.x = tf_operations['x']
        self.y = tf_operations['y']
        self.cost_function = tf_operations['cost_function']
        self.optimiser = tf_operations['optimiser']
        self.predict = tf_operations['predict']
        self.correct_prediction = tf_operations['correct_prediction']
        self.accuracy = tf_operations['accuracy']
        self.init_op = tf_operations['init_op']
        self.merged = tf_operations['merged']
        self.writer = tf_operations['writer']
        self.saver = tf_operations['saver']



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

    def train(self):
        print("=============================================")
        print("Training  started")
        sess = self.sess
        # initialise the variables
        sess.run(self.init_op)
        total_batch = int(self.train_size / self.batch_size)
        for epoch in range(self.num_epochs):
            avg_cost = 0
            for i in range(total_batch):
                offset = i*self.batch_size
                batch_x = self.train_x[offset:(offset + self.batch_size), :, :, :]
                batch_x_reshaped = self.reshape_x(batch_x)
                batch_y = self.train_y[offset:(offset + self.batch_size)]
                _, c = sess.run([self.optimiser, self.cost_function], feed_dict={self.x: batch_x_reshaped, self.y: batch_y})
                avg_cost += c / total_batch
            
            validation_x_reshaped = self.reshape_x(self.validation_x)
            train_acc = self.train_accuracy()
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), "train accuracy: {:.3f}".format(train_acc))
            summary = sess.run(self.merged, feed_dict={self.x: validation_x_reshaped, self.y: self.validation_y})
            self.writer.add_summary(summary, epoch)


        print("\nTraining complete!")
        self.writer.add_graph(sess.graph)

    def create_new_conv_layer(self, input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
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

        return out_layer
    
    def save_model(self, destination):
        self.saver.save(self.sess, destination)

    def to1hot(self,row):
        one_hot = np.zeros(2)
        one_hot[row] = 1.0
        return one_hot

    def from1hot(self,row):
        value=0.0
        if row[1] == 1.0:
            value = 1.0
        return value

    def check_accuracy(self, x, y):
        results = self.sess.run(self.accuracy, feed_dict={self.x:x, self.y:y})
        return results

    def _get_predictions(self, x ,y):
        results = self.sess.run(self.predict, feed_dict={self.x:x, self.y:y})
        return results