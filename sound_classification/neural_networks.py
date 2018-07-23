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
import sound_classification.data_handling as dh


class CNNWhale():
    """ Create a Convolutional Neural Network.

        The Network has two convolutional layers
        and two fully connected layers with ReLU activation functions.

        Args:
            train_x: pandas DataFrame
                Data Frame in which each row hold one flatten (as a vector, not matrix) image. 
            train_y: pandas DataFrame
                Data Frame in which each row contains the one hot encoded label
            validation_x: pandas DataFrame
                Data Frame in which each row hold one flatten (as a vector, not matrix) image.

            validation_y: pandas DataFrame
                Data Frame in which each row contains the one hot encoded label
            test_x: pandas DataFrame
                Data Frame in which each row hold one flatten (as a vector, not matrix) image.
            test_y: pandas DataFrame
                Data Frame in which each row contains the one hot encoded label
            batch_size: int
                The number of examples in each batch
            num_channels: int
                ...
            input_shape: tuple (int)
                A tuple of ints specifying the shape of the input images. Example: (60,20)
            learning_rate: float
                The learning rate to be using by the optimization algorithm
            num_epochs: int
                The number of epochs
            
        Attributes:
            sess: tensorflow Session
                The session object that will run operations in the network's graph.
            x: tensorflow tensor
                The input images.
            y: tensorflow tensor
                The labels.
            cost_function: tensorflow operation
                The cost function node in the network's graph.
            optimiser: tensorflow operation
                The optimizer that optimizes the weights.
            predict: tensorflow operation
                The prediction operation. Uses the trained model to predict labels.
            correct_prediction: tensorflow operation
                The operation that verifies if a prediction is correct.
            accuracy: tensorflow operation
                The operation that computes the predictions accuracy.
            init_op: tensorflow operation
                Initializer.
            merged: tensorflow summary
                The merged version of all summary statiscs collected. 
            writer: tensorflow writer
                Writer object that records model information on the saved model file.
            saver: tensorflow saver
                Saver object that save model information to a file.
    """

    def __init__(self, train_x, train_y, validation_x, validation_y,
                 test_x, test_y, batch_size, num_channels, num_labels,
                 learning_rate=0.01, num_epochs=10, seed=42):
        dh.check_data_sanity(train_x, train_y) # check sanity of training data
        dh.check_data_sanity(validation_x, validation_y) # check sanity of validation data
        dh.check_data_sanity(test_x, test_y) # check sanity of test data

        train_img_size = dh.get_image_size(train_x) # automatically determine image size
        val_img_size = dh.get_image_size(validation_x)
        test_img_size = dh.get_image_size(test_x)
        assert train_img_size == val_img_size and val_img_size == test_img_size, "test, validation and train images do not have same size"

        self.train_x = train_x
        self.train_y = train_y
        self.validation_x = validation_x
        self.validation_y = validation_y
        self.test_x = test_x
        self.test_y = test_y
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.num_labels = num_labels
        self.input_shape = train_img_size[0:2]
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.set_seed(seed)
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


    @classmethod
    def from_prepared_data(cls, prepared_data, 
                           batch_size, num_channels, num_labels, 
                           learning_rate=0.01, 
                           num_epochs=10, seed=42):
        train_x = prepared_data["train_x"]
        train_y = prepared_data["train_y"]
        validation_x = prepared_data["validation_x"]
        validation_y = prepared_data["validation_y"]
        test_x = prepared_data["test_x"]
        test_y = prepared_data["test_y"]

        return cls(train_x, train_y, validation_x, validation_y,
                 test_x, test_y, batch_size, num_channels, num_labels,
                 learning_rate, num_epochs, seed)


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


    def create_net_structure(self):
        """Create the Neural Network structure.

            The Network has two convolutional layers
            and two fully connected layers with ReLU activation functions.

            Returns:
                tf_objects: dict
                    A dictionary with the tensorflow objects necessary
                    to train and run the model.
                    sess, x, y, cost_function, optimiser, predict, correct_prediction,
                    accuracy,init_op, merged, writer, saver
                    These objects are stored as
                    instance attributes when the class is instantiated.

        """
        x = tf.placeholder(tf.float32, [None, self.input_shape[0] * self.input_shape[1]])
        x_shaped = tf.reshape(x, [-1, self.input_shape[0], self.input_shape[1], 1])
        y = tf.placeholder(tf.float32, [None, self.num_labels])

        pool_shape=[2,2]

        layer1 = self.create_new_conv_layer(x_shaped, 1, 32, [2, 8], pool_shape, name='layer1')
        layer2 = self.create_new_conv_layer(layer1, 32, 64, [30, 8], pool_shape, name='layer2')

        x_after_pool = int(np.ceil(self.input_shape[0]/(pool_shape[0]*2)))
        y_after_pool = int(np.ceil(self.input_shape[1]/(pool_shape[1]*2)))
        
        flattened = tf.reshape(layer2, [-1, x_after_pool * y_after_pool * 64])

        # setup some weights and bias values for this layer, then activate with ReLU
        wd1 = tf.Variable(tf.truncated_normal([x_after_pool* y_after_pool * 64, 512], stddev=0.03), name='wd1')
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

        tf_objects = {'x': x,
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

        return tf_objects

    def train(self):
        """Train the neural network. on the training set.

           Devide the training set in batches in orther to train.
           Once training is done, check the accuracy on the validation set.
           Record summary statics during training. 


        Returns:
            None
        """
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
            train_acc = self.accuracy_on_train()
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), "train accuracy: {:.3f}".format(train_acc))
            summary = sess.run(self.merged, feed_dict={self.x: validation_x_reshaped, self.y: self.validation_y})
            self.writer.add_summary(summary, epoch)


        print("\nTraining complete!")
        self.writer.add_graph(sess.graph)

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
        results = self.sess.run(self.accuracy, feed_dict={self.x:x, self.y:y})
        return results

    def _get_predictions(self, x ,y):
        """ Predict labels by running the model on x

        Args:
            x:tensor
                Tensor containing the input data.
            y: tensor
                Tensor containing the one hot encoded labels.
        Returns:
            results: vector
                A vector containing the predicted labels.                
        """
        results = self.sess.run(self.predict, feed_dict={self.x:x, self.y:y})
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
        results = self._get_predictions(validation_x_reshaped, self.validation_y)
        return results

    def predict_on_test(self):
        """ Predict labels by running the model on the test set.
        
        Returns:
            results: vector
                A vector containing the predicted labels.                
        """
        test_x_reshaped = self.reshape_x(self.test_x)
        results = self._get_predictions(test_x_reshaped, self.test_y)
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
        predicted = self._get_predictions(x_reshaped,y)
        pred_df = pd.DataFrame({"label":np.array(list(map(dh.from1hot,y))), "pred": predicted})
       
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
        #validation_x_reshaped = reshape_x(x, input_shape)
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