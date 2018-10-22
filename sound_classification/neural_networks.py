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
from sound_classification.data_handling import check_data_sanity, from1hot, get_image_size


class DataUse(Enum):
    TRAINING = 1
    EVALUATION = 2
    TEST = 3


class MNet():
    """ MERIDIAN Neural Network.

        Parent class for all MERIDIAN neural network implementations.
    """
    def __init__(self, seed=42, verbosity=2):

        self.verbosity = verbosity
        self.set_seed(seed)
        self.images = {DataUse.TRAINING: None, DataUse.VALIDATION: None, DataUse.TEST: None}
        self.labels = {DataUse.TRAINING: None, DataUse.VALIDATION: None, DataUse.TEST: None}
        self.sess = tf.Session()

    def _set_data(self, x, y, use):
        check_data_sanity(x, y)
        self.images[use] = x
        self.labels[use] = y

    def _add_data(self, x, y, use):
        x0 = self.images[use]
        y0 = self.labels[use]
        if x0 is not None:
            x = np.append(x0, x, axis=0)
        if y0 is not None:
            y = np.append(y0, y, axis=0)
        self._set_data(x=x, y=y, use=use)

    def set_training_data(self, x, y):
        self._set_data(x=x, y=y, use=DataUse.TRAINING)

    def add_training_data(self, x, y):
        self._add_data(x=x, y=y, use=DataUse.TRAINING)

    def set_evaluation_data(self, x, y):
        self._set_data(x=x, y=y, use=DataUse.EVALUATION)

    def add_evaluation_data(self, x, y):
        self._add_data(x=x, y=y, use=DataUse.EVALUATION)

    def set_test_data(self, x, y):
        self._set_data(x=x, y=y, use=DataUse.TEST)

    def add_test_data(self, x, y):
        self._add_data(x=x, y=y, use=DataUse.TEST)

    def set_tf_nodes(self, tf_nodes):
        """ Set the nodes of the tensorflow graph as instance attributes, so that other methods can access them

            Args:
                tf_nodes:dict
                A dictionary with the tensorflow objects necessary
                to train and run the model.
                sess, x, y, cost_function, optimizer, predict, correct_prediction,
                accuracy,init_op, merged, writer, saver
                These objects are stored as
                instance attributes when the class is instantiated.

            Returns:
                None
        """
        self.x = tf_nodes['x']
        self.y = tf_nodes['y']
        self.cost_function = tf_nodes['cost_function']
        self.optimizer = tf_nodes['optimizer']
        self.predict = tf_nodes['predict']
        self.correct_prediction = tf_nodes['correct_prediction']
        self.accuracy = tf_nodes['accuracy']
        self.init_op = tf_nodes['init_op']
        self.merged = tf_nodes['merged']
        self.writer = tf_nodes['writer']
        self.saver = tf_nodes['saver']
        self.keep_prob = tf_nodes['keep_prob']

    def set_seed(self,seed):
        """Set the random seed.

            Useful to generate reproducible results.

            Args:
                seed: int
                    Seed to be used by both tensorflow and numpy when generating random numbers
            Returns:
                None        
        """
        np.random.seed(seed)
        tf.set_random_seed(seed)
             
    def set_verbosity(self, verbosity):
        self.verbosity = verbosity

    def load_net_structure(self, saved_meta, checkpoint):
        """Load the Neural Network structure from a saved model.

        See the save_model() method. 

        Args:
            saved_meta: str
                Path to the saved .meta file.

            checkpoint: str
                Path to the checkpoint to be used when loading the saved model
            

        Returns:
            tf_nodes: dict
                A dictionary with the tensorflow objects necessary
                to train and run the model.
                sess, x, y, cost_function, optimizer, predict, correct_prediction,
                accuracy,init_op, merged, writer, saver
                These objects are stored as
                instance attributes when the class is instantiated.

        """
        sess = self.sess
        restorer = tf.train.import_meta_graph(saved_meta)
        restorer.restore(sess, tf.train.latest_checkpoint(checkpoint))

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        cost_function = graph.get_tensor_by_name("cost_function:0")
        optimizer = graph.get_operation_by_name("optimizer")
        predict = graph.get_tensor_by_name("predict:0")
        correct_prediction = graph.get_tensor_by_name("correct_prediction:0")
        accuracy = graph.get_tensor_by_name("accuracy:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")        

        init_op = tf.global_variables_initializer()
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('summaries')
        saver = tf.train.Saver()

        tf_nodes = {'x': x,
                'y':y,            
                'cost_function': cost_function,
                'optimizer': optimizer,
                'predict': predict,
                'correct_prediction':correct_prediction,
                'accuracy': accuracy,
                'init_op': init_op,
                'merged': merged,
                'writer': writer,
                'saver': saver,
                'keep_prob': keep_prob,
                }

        return tf_nodes

    def create_net_structure(self):
        """Create the neural network structure.

            Returns:
                tf_nodes: dict
                    A dictionary with the tensorflow objects necessary
                    to train and run the model.
                    sess, x, y, cost_function, optimizer, predict, correct_prediction,
                    accuracy,init_op, merged, writer, saver
                    These objects are stored as
                    instance attributes when the class is instantiated.

        """
        img_shape = self._image_shape()
        tf_nodes = _create_net_structure(input_shape=img_shape)
        return tf_nodes

    def _image_shape(self):
        assert self.images[DataUse.TRAIN] is not None, "Training data must be provided before the neural network structure can be created."

        img_shape = get_image_size(self.images[DataUse.TRAIN]) 

        if self.images[DataUse.VALIDATION] is not None:
            assert img_shape == get_image_size(self.images[DataUse.VALIDATION]), "Training and validation images must have same shape."

        if self.images[DataUse.TEST] is not None:
            assert img_shape == get_image_size(self.images[DataUse.TEST]), "Training and test images must have same shape."

        return img_shape

        tf_nodes = self._create_net_structure(input_shape=img_shape)
        return tf_nodes

    def _create_net_structure(self, input_shape):
        return None

    def train(self, batch_size=0, epochs=0, dropout=1.0, feature_layer_name=None):
        """Train the neural network. on the training set.

           Devide the training set in batches in orther to train.
           Once training is done, check the accuracy on the validation set.
           Record summary statics during training. 

        Args:
            batch_size: int
                Batch size. Overwrites batch size specified at initialization.
            epochs: int
                Number of epochs: Overwrites number of epochs specified at initialization.
            dropout: float
                Float in the range [0,1] specifying the probability of keeping the weights, 
                i.e., drop out will only be effectuated if dropout < 1.
            feature_layer_name: str
                Name of 'feature' layer.

        Returns:
            avg_cost: float
                Average cost of last completed training epoch.
        """
        if batch_size == 0:
            batch_size = self.batch_size
        if epochs == 0:
            epochs = self.num_epochs

        sess = self.sess
        x = self.train_x
        y = self.train_y
        x_val = self.validation_x
        y_val = self.validation_y

        self.writer.add_graph(sess.graph)

        features = None
        if feature_layer_name is not None:
            features = sess.graph.get_tensor_by_name(feature_layer_name)

        if self.verbosity >= 2:
            print("\nTraining  started")
            header = '\nEpoch  Cost  Test acc.  Val acc.'
            line   = '----------------------------------'
            if features is not None:
                header += '   No. Feat. used'
                line   += '-----------------'
            print(header)
            print(line)

        # initialise the variables
        sess.run(self.init_op)

        batches = int(y.shape[0] / batch_size)
        for epoch in range(epochs):
            avg_cost = 0
            avg_acc = 0
            feat_used = 0
            for i in range(batches):
                offset = i * batch_size
                x_i = x[offset:(offset + batch_size), :, :, :]
                x_i = self.reshape_x(x_i)
                y_i = y[offset:(offset + batch_size)]
                fetch = [self.optimizer, self.cost_function, self.accuracy]

                if features is not None:
                    fetch.append(features)
                    _, c, a, f = sess.run(fetches=fetch, feed_dict={self.x: x_i, self.y: y_i, self.keep_prob: dropout})
                    f = np.sum(f, axis=0)
                    feat_used += np.sum(f > 0) / batches
                else:
                    _, c, a = sess.run(fetches=fetch, feed_dict={self.x: x_i, self.y: y_i, self.keep_prob: dropout})
                
                avg_cost += c / batches
                avg_acc += a / batches
            
            if self.verbosity >= 2:
                val_acc = self.accuracy_on_validation()
                s = ' {0}  {1:.3f}  {2:.3f}  {3:.3f}'.format(epoch + 1, avg_cost, avg_acc, val_acc)
                if features is not None:
                    s += '  {4:.1f}'.format(feat_used)
                print(s)

            x_val = self.reshape_x(x_val)
            summary = sess.run(self.merged, feed_dict={self.x: x_val, self.y: y_val, self.keep_prob: 1.0})
            self.writer.add_summary(summary, epoch)

        if self.verbosity >= 2:
            print(line)
            print("\nTraining completed!")

        return avg_cost

        
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
    
    def save_model(self, destination):
        """ Save the model to destination

            Args:
                destination: str
                    Path to the file in which the model will be saved. 

            Returns:
                None.
        
        """
        self.saver.save(self.sess, destination)

    def _check_accuracy(self, x, y):
        """ Check accuracy of the model by checking how close
         to y the models predictions are when fed x

            Based on the accuracy operation stored in the attribute 'self.accuracy'),
            which is defined by the 'create_net_structure()' method.

        Args:
            x:tensor
                Tensor containing the input data
            y: tensor
                Tensor containing the one hot encoded labels
        Returns:
            results: float
                The accuracy value
        """
        results = self.sess.run(self.accuracy, feed_dict={self.x:x, self.y:y, self.keep_prob:1.0})
        return results

    def get_predictions(self, x):
        """ Predict labels by running the model on x

        Args:
            x:tensor
                Tensor containing the input data.
            
        Returns:
            results: vector
                A vector containing the predicted labels.                
        """
        results = self.sess.run(self.predict, feed_dict={self.x:x, self.keep_prob:1.0})
        return results

    def reshape_x(self, x):
        """ Reshape input data from a 2s matrix to a 1d vector.

        Args:
            x: numpy array
                2d array containing the input data.
        Returns:
            results: vector
                A vector containing the flattened inputs.                
        """
        reshaped_x = np.reshape(x, (-1, self.input_shape[0] * self.input_shape[1]))
        return reshaped_x

    def predict_on_validation(self):
        """ Predict labels by running the model on the validation set.
        
        Returns:
            results: vector
                A vector containing the predicted labels.                
        """
        validation_x_reshaped = self.reshape_x(self.validation_x)
        results = self.get_predictions(validation_x_reshaped)
        return results

    def predict_on_test(self):
        """ Predict labels by running the model on the test set.
        
        Returns:
            results: vector
                A vector containing the predicted labels.                
        """
        test_x_reshaped = self.reshape_x(self.test_x)
        results = self.get_predictions(test_x_reshaped)
        return results
    
    def _get_mislabelled(self, x, y, print_report=False):
        """ Report the number of examples mislabelled by the model.

            Args:
                x:tensor
                    Tensor containing the input data.
                y: tensor
                    Tensor containing the one hot encoded labels.
                print_report:bool
                    If True, prints the percentage of correct and incorrect
                    and the index of examples misclassified examples with the
                    correct and predicted labels.
            Returns:
                results: tuple (pandas DataFrames)
                Tuple with two  DataFrames (report, incorrect). The first contains
                number and percentage of correct/incorrect classification. The second,
                the incorrect examples indices with incorrect and correct labels. 
        
        """

        x_reshaped = self.reshape_x(x)
        predicted = self.get_predictions(x_reshaped)
        pred_df = pd.DataFrame({"label":np.array(list(map(from1hot,y))), "pred": predicted})
       
        n_predictions = len(pred_df)
        n_correct = sum(pred_df.label == pred_df.pred)
        perc_correct = round(n_correct/n_predictions * 100, 2)
        incorrect = pred_df[pred_df.label != pred_df.pred]
        n_incorrect = len(incorrect)  
        perc_incorrect = round(n_incorrect/n_predictions * 100, 2)
        
        #pred_df.to_csv("predictions.csv")
        report = pd.DataFrame({"correct":[n_correct], "incorrect":[n_incorrect],
                            "%correct":[perc_correct],"%incorrect":[perc_incorrect],
                            "total":[n_predictions]})

        if print_report:
            print("=============================================")
            print("Correct classifications: {0} of {1} ({2}%)".format(n_correct, n_predictions, perc_correct))
            print("Incorrect classifications: {0} of {1} ({2})%".format(n_incorrect, n_predictions, perc_incorrect))
            print("These were the incorrect classifications:")
            print(incorrect)
            print("=============================================") 
        
        results =(report,incorrect)    
        return results

    def mislabelled_on_validation(self, print_report=False):
        """ Report the number of examples mislabelled by the trained model on
            the validation set.

            This method wraps around the '_get_mislabelled()' method in the same class.

            Args:
                print_report:bool
                    If True, prints the percentage of correct and incorrect
                    and the index of examples misclassified examples with the
                    correct and predicted labels.
            Returns:
                results: tuple (pandas DataFrames)
                Tuple with two  DataFrames. The first contains
                number and percentage of correct/incorrect classification. The second,
                the incorrect examples indices with incorrect and correct labels. 
        """

        results = self._get_mislabelled(x=self.validation_x,y=self.validation_y, print_report=print_report)
        return results

    def mislabelled_on_test(self, print_report=False):
        """ Report the number of examples mislabelled by the trained model on
            the test set.

            This method wraps around the '_get_mislabelled()' method in the same class.

            Args:
                print_report:bool
                    If True, prints the percentage of correct and incorrect
                    and the index of examples misclassified examples with the
                    correct and predicted labels.
            Returns:
                results: tuple (pandas DataFrames)
                Tuple with two  DataFrames. The first contains
                number and percentage of correct/incorrect classification. The second,
                the incorrect examples indices with incorrect and correct labels. 
        """
        report = self._get_mislabelled(x=self.test_x,y=self.test_y, print_report=print_report)
        return report

    def accuracy_on_train(self):
        """ Report the model accuracy on the training set

            This method wraps around 'check_accuracy()' in the same class.

            Returns:
                results: float
                    The accuracy on the training set.
        """
        train_x_reshaped = self.reshape_x(self.train_x)
        results = self._check_accuracy(train_x_reshaped, self.train_y)
        return results

    def accuracy_on_validation(self):
        """Report the model accuracy on the validation set

            This method wraps around 'check_accuracy()' in the same class.

            Returns:
                results: float
                    The accuracy on the validation set.
        """

        validation_x_reshaped = self.reshape_x(self.validation_x)
        results = self._check_accuracy(validation_x_reshaped, self.validation_y)
        return results

    def accuracy_on_test(self):
        """Report the model accuracy on the test set

            This method wraps around 'check_accuracy()' in the same class.

            Returns:
                results: float
                    The accuracy on the test set.
        """
        test_x_reshaped = self.reshape_x(self.test_x)
        results = self._check_accuracy(test_x_reshaped, self.test_y)
        return results
