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
from sound_classification.neural_networks import DataHandler, DataUse, predictions, class_confidences
from sound_classification.data_handling import from1hot, to1hot, get_image_size
from sound_classification.training_data_provider import TrainingDataProvider


ConvParams = namedtuple('ConvParams', 'name n_filters filter_shape')
ConvParams.__doc__ = '''\
Name and dimensions of convolutional layer in neural network

name - Name of convolutional layer, e.g. "conv_layer"
n_filters - Number of filters, e.g. 16
filter_shape - Filter shape, e.g. [4,4]'''


class CNNWhale(DataHandler):
    """ Create a Convolutional Neural Network for classification tasks.

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
            learning_rate: float
                The learning rate to be using by the optimization algorithm
            keep_prob: float
                Probability of keeping weights. If keep_prob < 1.0, drop-out is enabled during training.
            seed: int
                Seed to be used by both tensorflow and numpy when generating random numbers            
            verbosity: int
                Verbosity level (0: no messages, 1: warnings only, 2: warnings and diagnostics)
    """

    def __init__(self, train_x, train_y, validation_x=None, validation_y=None,
                 test_x=None, test_y=None, num_labels=2, batch_size=128, 
                 num_epochs=10, learning_rate=0.01, keep_prob=1.0, seed=42, verbosity=2):

        self.num_labels = num_labels
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate_value = learning_rate
        self.keep_prob_value = keep_prob
        self.set_seed(seed)
        self.verbosity = verbosity
        self.sess = tf.Session()
        self.epoch_counter = 0

        super(CNNWhale, self).__init__(train_x=train_x, train_y=train_y, 
                validation_x=validation_x, validation_y=validation_y,
                test_x=test_x, test_y=test_y)
        
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

    def reset(self):
        self.epoch_counter = 0
        self.sess.run(self.init_op)

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
        self.learning_rate = tf_nodes['learning_rate']
        self.class_weights = tf_nodes['class_weights']
        self.reset()

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
        """Set verbosity level.
            0: no messages
            1: warnings only
            2: warnings and diagnostics

            Args:
                verbosity: int
                    Verbosity level.        
        """
        self.verbosity = verbosity

    def load(self, saved_meta, checkpoint):
        """Load the Neural Network structure from a saved model.

        See the save() method. 

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
        learning_rate = graph.get_tensor_by_name("learning_rate:0")      
        class_weights = graph.get_tensor_by_name("class_weights:0")  

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
                'learning_rate': learning_rate,
                'class_weights': class_weights,
                }

        self.set_tf_nodes(tf_nodes)

        return tf_nodes

    def create(self, conv_params=[ConvParams(name='conv_1',n_filters=32,filter_shape=[2,8]), ConvParams(name='conv_2',n_filters=64,filter_shape=[30,8])], dense_size=[512]):
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
        input_shape = self._image_shape()
        num_labels = self.num_labels

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

        self.set_tf_nodes(tf_nodes)

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

        return out_layer

    def _image_shape(self):
        """Get the image shape.

            Returns:
                img_shape: array
                    Shape of the input data images

        """
        assert self.images[DataUse.TRAINING] is not None, "Training data must be provided before the neural network structure can be created."

        img_shape = get_image_size(self.images[DataUse.TRAINING]) 

        if self.images[DataUse.VALIDATION] is not None:
            assert img_shape == get_image_size(self.images[DataUse.VALIDATION]), "Training and validation images must have same shape."

        if self.images[DataUse.TEST] is not None:
            assert img_shape == get_image_size(self.images[DataUse.TEST]), "Training and test images must have same shape."

        return img_shape

    def train(self, batch_size=None, num_epochs=None, learning_rate=None, keep_prob=None):
        """Train the neural network on the training set.

           Devide the training set in batches in orther to train.
           Once training is done, check the accuracy on the validation set.
           Record summary statics during training. 

        Args:
            batch_size: int
                Batch size. Overwrites batch size specified at initialization.
            num_epochs: int
                Number of epochs: Overwrites number of epochs specified at initialization.
            learning_rate: float
                The learning rate to be using by the optimization algorithm
            keep_prob: float
                Float in the range [0,1] specifying the probability of keeping the weights, 
                i.e., drop-out will be applied if keep_prob < 1.

        Returns:
            avg_cost: float
                Average cost of last completed training epoch.
        """
        if batch_size is None:
            batch_size = self.batch_size
        if num_epochs is None:
            num_epochs = self.num_epochs
        if keep_prob is None:
            keep_prob = self.keep_prob_value
        if learning_rate is None:
            learning_rate = self.learning_rate_value

        sess = self.sess

        x, y = self.get_training_data()
        x_val, y_val = self.get_validation_data()

        self.writer.add_graph(sess.graph)

        if self.verbosity >= 2:
            if self.epoch_counter == 0: 
                print("\nTraining  started")
            header = '\nEpoch  Cost  Test acc.  Val acc.'
            line   = '----------------------------------'
            print(header)
            print(line)

        # initialise the variables
        if self.epoch_counter == 0:
            sess.run(self.init_op)

        batches = int(y.shape[0] / batch_size)
        for epoch in range(num_epochs):
            avg_cost = 0
            avg_acc = 0
            for i in range(batches):
                offset = i * batch_size
                x_i = x[offset:(offset + batch_size), :, :, :]
                x_i = self.reshape_x(x_i)
                y_i = y[offset:(offset + batch_size)]
                fetch = [self.optimizer, self.cost_function, self.accuracy]

                _, c, a = sess.run(fetches=fetch, feed_dict={self.x: x_i, self.y: y_i, self.learning_rate: learning_rate, self.keep_prob: keep_prob})
                
                avg_cost += c / batches
                avg_acc += a / batches
            
            if self.verbosity >= 2:
                val_acc = self.accuracy_on_validation()
                s = ' {0}/{4}  {1:.3f}  {2:.3f}  {3:.3f}'.format(epoch + 1, avg_cost, avg_acc, val_acc, num_epochs)
                print(s)

            if x_val is not None:
                x_val = self.reshape_x(x_val)
                summary = sess.run(fetches=self.merged, feed_dict={self.x: x_val, self.y: y_val, self.learning_rate: learning_rate, self.keep_prob: 1.0})
                self.writer.add_summary(summary, self.epoch_counter)

            self.epoch_counter += 1

        if self.verbosity >= 2:
            print(line)

        return avg_cost

    def train_active(self, provider, iterations=1, batch_size=None, num_epochs=None, learning_rate=None, keep_prob=None):
        """Train the neural network in an active manner using a data provider module.
        
        Args:
            provider: TrainingDataProvider
                Training data provider
            iterations: int
                Number of training iterations.
            batch_size: int
                Batch size. Overwrites batch size specified at initialization.
            num_epochs: int
                Number of epochs: Overwrites number of epochs specified at initialization.
            learning_rate: float
                The learning rate to be using by the optimization algorithm
            keep_prob: float
                Float in the range [0,1] specifying the probability of keeping the weights, 
                i.e., drop-out will be applied if keep_prob < 1.

        Returns:
            avg_cost: float
                Average cost of last completed training epoch.
        """    
        for i in range(iterations):

            # get data from provider
            x_train, y_train, keep_frac = provider.get_samples()    

            if self.verbosity >= 2:
                print('\nIteration: {0}/{1}'.format(i+1, iterations))
                print('Keep frac: {0:.1f}%'.format(100.*keep_frac))

            # feed data to neural net
            self.set_training_data(x=x_train, y=to1hot(y_train,2))

            # train
            self.train(batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate, keep_prob=keep_prob)

            # update predictions and confidences
            w = self.get_class_weights(x_train)
            pred = predictions(w)
            conf = class_confidences(w)
            provider.update_prediction_confidence(pred=pred, conf=conf)


    def save(self, destination):
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
        if x is None:
            return 0
        x = self.reshape_x(x)        
        results = self.sess.run(fetches=self.accuracy, feed_dict={self.x:x, self.y:y, self.learning_rate: self.learning_rate_value, self.keep_prob:1.0})
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
        x = self.reshape_x(x)
        results = self.sess.run(fetches=self.predict, feed_dict={self.x:x, self.learning_rate: self.learning_rate_value, self.keep_prob:1.0})
        return results

    def get_features(self, x, layer_name):
        """ Compute feature vector by running the model on x

        Args:
            x: tensor
                Tensor containing the input data.
            layer_name: str
                Name of the feature layer.
            
        Returns:
            results: vector
                A vector containing the feature values.                
        """
        x = self.reshape_x(x)
        graph = tf.get_default_graph()
        f = graph.get_tensor_by_name("{0}:0".format(layer_name)) 
        results = self.sess.run(fetches=f, feed_dict={self.x:x, self.learning_rate: self.learning_rate_value, self.keep_prob:1.0})
        return results

    def get_class_weights(self, x):
        """ Compute classification weights by running the model on x.

        Args:
            x:tensor
                Tensor containing the input data.
            
        Returns:
            results: vector
                A vector containing the classification weights. 
        """
        x = self.reshape_x(x)
        fetch = self.class_weights
        feed = {self.x:x, self.learning_rate: self.learning_rate_value, self.keep_prob:1.0}
        results = self.sess.run(fetches=fetch, feed_dict=feed)
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
        img_shape = self._image_shape()
        reshaped_x = np.reshape(x, (-1, img_shape[0] * img_shape[1]))
        return reshaped_x

    def predict_on_validation(self):
        """ Predict labels by running the model on the validation set.
        
        Returns:
            results: vector
                A vector containing the predicted labels.                
        """
        x = self.images[DataUse.VALIDATION]
        results = self.get_predictions(x)
        return results

    def predict_on_test(self):
        """ Predict labels by running the model on the test set.
        
        Returns:
            results: vector
                A vector containing the predicted labels.                
        """
        x = self.images[DataUse.TEST]
        results = self.get_predictions(x)
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
        x = self.images[DataUse.VALIDATION]
        y = self.labels[DataUse.VALIDATION]
        results = self._get_mislabelled(x=x,y=y, print_report=print_report)
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
        x = self.images[DataUse.TEST]
        y = self.labels[DataUse.TEST]
        results = self._get_mislabelled(x=x,y=y, print_report=print_report)
        return results

    def accuracy_on_train(self):
        """ Report the model accuracy on the training set

            This method wraps around 'check_accuracy()' in the same class.

            Returns:
                results: float
                    The accuracy on the training set.
        """
        x = self.images[DataUse.TRAINING]
        y = self.labels[DataUse.TRAINING]
        results = self._check_accuracy(x,y)
        return results

    def accuracy_on_validation(self):
        """Report the model accuracy on the validation set

            This method wraps around 'check_accuracy()' in the same class.

            Returns:
                results: float
                    The accuracy on the validation set.
        """
        x = self.images[DataUse.VALIDATION]
        y = self.labels[DataUse.VALIDATION]
        results = self._check_accuracy(x,y)
        return results

    def accuracy_on_test(self):
        """Report the model accuracy on the test set

            This method wraps around 'check_accuracy()' in the same class.

            Returns:
                results: float
                    The accuracy on the test set.
        """
        x = self.images[DataUse.TEST]
        y = self.labels[DataUse.TEST]
        results = self._check_accuracy(x,y)
        return results


