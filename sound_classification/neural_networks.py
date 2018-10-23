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
    VALIDATION = 2
    TEST = 3


class MNet():
    """ MERIDIAN Neural Network.

        Parent class for all MERIDIAN neural network implementations.

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

        self.images = {DataUse.TRAINING: None, DataUse.VALIDATION: None, DataUse.TEST: None}
        self.labels = {DataUse.TRAINING: None, DataUse.VALIDATION: None, DataUse.TEST: None}

        self.num_labels = num_labels
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate_value = learning_rate
        self.keep_prob_value = keep_prob
        self.set_seed(seed)
        self.verbosity = verbosity

        self._set_data(train_x, train_y, use=DataUse.TRAINING)        
        self._set_data(validation_x, validation_y, use=DataUse.VALIDATION)        
        self._set_data(test_x, test_y, use=DataUse.TEST)        

        self.sess = tf.Session()

    def _set_data(self, x, y, use):
        """ Set data for specified use (training, validation, or test). 
            Replaces any existing data for that use type.

            Args:
                x: pandas DataFrame
                    Data Frame in which each row holds one image. 
                y: pandas DataFrame
                    Data Frame in which each row contains the one hot encoded label
                use: DataUse
                    Data use. Possible options are TRAINING, VALIDATION and TEST
        """
        check_data_sanity(x, y)
        self.images[use] = x
        self.labels[use] = y

    def _add_data(self, x, y, use):
        """ Add data for specified use (training, validation, or test). 
            Will be appended to any existing data for that use type.

            Args:
                x: pandas DataFrame
                    Data Frame in which each row holds one image. 
                y: pandas DataFrame
                    Data Frame in which each row contains the one hot encoded label
                use: DataUse
                    Data use. Possible options are TRAINING, VALIDATION and TEST
        """
        x0 = self.images[use]
        y0 = self.labels[use]
        if x0 is not None:
            x = np.append(x0, x, axis=0)
        if y0 is not None:
            y = np.append(y0, y, axis=0)
        self._set_data(x=x, y=y, use=use)

    def set_training_data(self, x, y):
        """ Set training data. Replaces any existing training data.

            Args:
                x: pandas DataFrame
                    Data Frame in which each row holds one image. 
                y: pandas DataFrame
                    Data Frame in which each row contains the one hot encoded label
        """
        self._set_data(x=x, y=y, use=DataUse.TRAINING)

    def add_training_data(self, x, y):
        """ Add training data. Will be appended to any existing training data.

            Args:
                x: pandas DataFrame
                    Data Frame in which each row holds one image. 
                y: pandas DataFrame
                    Data Frame in which each row contains the one hot encoded label
        """
        self._add_data(x=x, y=y, use=DataUse.TRAINING)

    def set_validation_data(self, x, y):
        """ Set validation data. Replaces any existing validation data.

            Args:
                x: pandas DataFrame
                    Data Frame in which each row holds one image. 
                y: pandas DataFrame
                    Data Frame in which each row contains the one hot encoded label
        """
        self._set_data(x=x, y=y, use=DataUse.VALIDATION)

    def add_validation_data(self, x, y):
        """ Add validation data. Will be appended to any existing validation data.

            Args:
                x: pandas DataFrame
                    Data Frame in which each row holds one image. 
                y: pandas DataFrame
                    Data Frame in which each row contains the one hot encoded label
        """
        self._add_data(x=x, y=y, use=DataUse.VALIDATION)

    def set_test_data(self, x, y):
        """ Set test data. Replaces any existing test data.

            Args:
                x: pandas DataFrame
                    Data Frame in which each row holds one image. 
                y: pandas DataFrame
                    Data Frame in which each row contains the one hot encoded label
        """
        self._set_data(x=x, y=y, use=DataUse.TEST)

    def add_test_data(self, x, y):
        """ Add test data. Will be appended to any existing test data.

            Args:
                x: pandas DataFrame
                    Data Frame in which each row holds one image. 
                y: pandas DataFrame
                    Data Frame in which each row contains the one hot encoded label
        """
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
        self.learning_rate = tf_nodes['learning_rate']
        self.class_weights = tf_nodes['class_weights']

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

        return tf_nodes

    def _create_net_structure(self, input_shape, num_labels, **kwargs):
        """Create the neural network structure.

            Implemented in any derived class.
        """
        return None

    def create_net_structure(self, **kwargs):
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
        tf_nodes = self._create_net_structure(input_shape=img_shape, num_labels=self.num_labels, kwargs=kwargs)
        return tf_nodes

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

    def train(self, batch_size=None, num_epochs=None, learning_rate=None, keep_prob=None, feature_layer_name=None):
        """Train the neural network. on the training set.

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
            feature_layer_name: str
                Name of 'feature' layer.

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
        
        x = self.images[DataUse.TRAINING]
        y = self.labels[DataUse.TRAINING]
        x_val = self.images[DataUse.VALIDATION]
        y_val = self.labels[DataUse.VALIDATION]

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
        for epoch in range(num_epochs):
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
                    _, c, a, f = sess.run(fetches=fetch, feed_dict={self.x: x_i, self.y: y_i, self.learning_rate: learning_rate, self.keep_prob: keep_prob})
                    f = np.sum(f, axis=0)
                    feat_used += np.sum(f > 0) / batches
                else:
                    _, c, a = sess.run(fetches=fetch, feed_dict={self.x: x_i, self.y: y_i, self.learning_rate: learning_rate, self.keep_prob: keep_prob})
                
                avg_cost += c / batches
                avg_acc += a / batches
            
            if self.verbosity >= 2:
                val_acc = self.accuracy_on_validation()
                s = ' {0}  {1:.3f}  {2:.3f}  {3:.3f}'.format(epoch + 1, avg_cost, avg_acc, val_acc)
                if features is not None:
                    s += '  {4:.1f}'.format(feat_used)
                print(s)

            if x_val is not None:
                x_val = self.reshape_x(x_val)
                summary = sess.run(fetches=self.merged, feed_dict={self.x: x_val, self.y: y_val, self.learning_rate: learning_rate, self.keep_prob: 1.0})
                self.writer.add_summary(summary, epoch)

        if self.verbosity >= 2:
            print(line)
            print("\nTraining completed!")

        return avg_cost
    
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

def remove_rights(x, y, class_weights, certainty_cut):
    p = np.argmax(class_weights, axis=1) # predictions
    idx = np.argsort(class_weights, axis=1)
    w0 = np.choose(idx[:,-1], class_weights.T) # max weights
    w1 = np.choose(idx[:,-2], class_weights.T) # second largest weights
    cert = w0 - w1
    rights = (p == y) & (cert >= certainty_cut)
    x_trimmed = x[np.logical_not(rights)]
    y_trimmed = y[np.logical_not(rights)]
    return x_trimmed, y_trimmed