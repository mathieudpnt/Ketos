# ================================================================================ #
#   Authors: Fabio Frazao and Oliver Kirsebom                                      #
#   Contact: fsfrazao@dal.ca, oliver.kirsebom@dal.ca                               #
#   Organization: MERIDIAN (https://meridian.cs.dal.ca/)                           #
#   Team: Data Analytics                                                           #
#   Project: ketos                                                                 #
#   Project goal: The ketos library provides functionalities for handling          #
#   and processing acoustic data and applying deep neural networks to sound        #
#   detection and classification tasks.                                            #
#                                                                                  #
#   License: GNU GPLv3                                                             #
#                                                                                  #
#       This program is free software: you can redistribute it and/or modify       #
#       it under the terms of the GNU General Public License as published by       #
#       the Free Software Foundation, either version 3 of the License, or          #
#       (at your option) any later version.                                        #
#                                                                                  #
#       This program is distributed in the hope that it will be useful,            #
#       but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              #
#       GNU General Public License for more details.                               # 
#                                                                                  #
#       You should have received a copy of the GNU General Public License          #
#       along with this program.  If not, see <https://www.gnu.org/licenses/>.     #
# ================================================================================ #

""" resnet sub-module within the ketos.neural_networks module

    This module provides classes that to implement Residual Networks (ResNets).

    Contents:
        ResNetBlock class:
        ResNet class:
"""

import tensorflow as tf
import os
from .losses import FScoreLoss
from .metrics import precision_recall_accuracy_f

class ResNetBlock(tf.keras.Model):
    def __init__(self, channels, strides=1, residual_path=False):
        super(ResNetBlock, self).__init__()

        self.channels = channels
        self.strides = strides
        self.residual_path = residual_path
        self.conv_1 = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=(3,3), strides=self.strides,
                                                padding="same", use_bias=False,
                                                kernel_initializer=tf.random_normal_initializer())
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.conv_2 = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=(3,3), strides=1,
                                                padding="same", use_bias=False,
                                                kernel_initializer=tf.random_normal_initializer())
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()

        if residual_path == True:
            self.conv_down = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=(1,1), strides=self.strides,
                                                padding="same", use_bias=False,
                                                kernel_initializer=tf.random_normal_initializer())
            self.batch_norm_down = tf.keras.layers.BatchNormalization()
        
        self.dropout = tf.keras.layers.Dropout(0.0)

    def call(self,inputs, training=None):
        residual = inputs

        x = self.batch_norm_1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv_1(x)
        x = self.dropout(x)
        x = self.batch_norm_2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = self.dropout(x)

        if self.residual_path:
            residual = self.batch_norm_down(inputs, training=training)
            residual = tf.nn.relu(residual)
            residual = self.conv_down(residual)
            x = self.dropout(x)

        x = x + residual
        return x


class ResNetArch(tf.keras.Model):

    def __init__(self, block_list, n_classes, initial_filters=16, **kwargs):
        super(ResNetArch, self).__init__(**kwargs)

        self.n_blocks = len(block_list)
        self.n_classes = n_classes
        self.block_list = block_list
        self.input_channels = initial_filters
        self.output_channels = initial_filters
        self.conv_initial = tf.keras.layers.Conv2D(filters=self.output_channels, kernel_size=(3,3), strides=1,
                                                padding="same", use_bias=False,
                                                kernel_initializer=tf.random_normal_initializer())

        self.blocks = tf.keras.models.Sequential(name="dynamic_blocks")

        for block_id in range(self.n_blocks):
            for layer_id in range(self.block_list[block_id]):
                #Frst layer of every block except the first
                if block_id != 0 and layer_id == 0:
                    block = ResNetBlock(self.output_channels, strides=2, residual_path=True)
                
                else:
                    if self.input_channels != self.output_channels:
                        residual_path = True
                    else:
                        residual_path = False
                    block = ResNetBlock(self.output_channels, residual_path=residual_path)

                self.input_channels = self.output_channels

                self.blocks.add(block)
            
            self.output_channels *= 2

        self.batch_norm_final = tf.keras.layers.BatchNormalization()
        self.average_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fully_connected = tf.keras.layers.Dense(self.n_classes)
        self.softmax = tf.keras.layers.Softmax()
    
    def call(self, inputs, training=None):

        output = self.conv_initial(inputs)

        output = self.blocks(output, training=training)
        output = self.batch_norm_final(output, training=training)
        output = tf.nn.relu(output)
        output = self.average_pool(output)
        output = self.fully_connected(output)
        output = self.softmax(output)

        return output



class ResNetInterface():

    valid_optimizers = {'Adam':tf.keras.optimizers.Adam}
    valid_losses = {'FScoreLoss':FScoreLoss}
    valid_metrics = {'CategoricalAccuracy':tf.keras.metrics.CategoricalAccuracy}

    @classmethod
    def parse_optimizer(cls, optimizer):
        name = optimizer['name']
        args = optimizer['parameters']

        if name not in cls.valid_optimizers.keys():
            raise ValueError("Invalid optimizer name '{}'".format(name))
        built_optimizer = cls.valid_optimizers[name](**args)

        return built_optimizer

    @classmethod
    def parse_loss_function(cls, loss_function):
        name = loss_function['name']
        args = loss_function['parameters']

        if name not in cls.valid_losses.keys():
            raise ValueError("Invalid loss function name '{}'".format(name))
        built_loss = cls.valid_loss[name](**args)

        return built_loss

    @classmethod
    def parse_metrics(cls, metrics):
        name = loss_function['name']
        args = loss_function['parameters']

        built_metrics = []
        for m in metrics: 
            if m['name'] not in cls.valid_metrics.keys():
                raise ValueError("Invalid metric name '{}'".format(m['name']))
        built_metrics.append(cls.valid_loss[name])

        return built_metrics



    @classmethod
    def read_recipe_file(cls, json_file):
        with open(json_file, 'r') as json_recipe:
            recipe_dict = json.load(json_recipe)

        optimizer = cls.parse_optimizer(recipe['optimizer'])
        loss_function = cls.parse_loss_function(recipe['loss_function'])
        metrics = cls.parse_metrics(recipe['metrics'])

        return recipe_dict

    @classmethod
    def load(cls, recipe, weights_path):
        instance = cls.build_from_recipe(recipe) 
        latest_checkpoint = tf.train.latest_checkpoint(weights_path)
        instance.model.load_weights(latest_checkpoint)

        return instance


    @classmethod
    def build_from_recipe(cls, recipe):
        block_list = recipe['block_list']
        n_classes = recipe['n_classes']
        initial_filters = recipe['initial_filters']
        optimizer = recipe['optimizer']
        loss_function = recipe['loss_function']
        metrics = recipe['metrics']

        instance = cls(block_list=block_list, n_classes=n_classes, initial_filters=initial_filters, optimizer=optimizer, loss_function=loss_function, metrics=metrics)

        return instance

    def __init__(self, block_list, n_classes, initial_filters, optimizer, loss_function, metrics):
        self.block_list = block_list
        self.n_classes = n_classes
        self.initial_filters = initial_filters
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics

        self.model=ResNetArch(block_list=block_list, n_classes=n_classes, initial_filters=initial_filters)
        self.compile_model()
        #self.metrics_names = self.model.metrics_names

        
        self.log_dir = None
        self.checkpoint_dir = None
        self.tensorboard_callback = None
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None

    def compile_model(self):
        self.model.compile(optimizer=self.optimizer,
                            loss = self.loss_function,
                            metrics = self.metrics)
        self.metrics_names = self.model.metrics_names

    def set_train_generator(self, train_generator):
        self.train_generator = train_generator

    def set_val_generator(self, val_generator):
        self.val_generator = val_generator

    def set_test_generator(self, test_generator):
        self.test_generator = test_generator

    def set_log_dir(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def set_tensorboard_callback(self):
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        tensorboard_callback.set_model(self.model)
        
    def print_metrics(self, metric_values):
        message  = [self.metrics_names[i] + ": {} ".format(metric_values[i]) for i in range(len(self.metrics_names))]
        #import pdb; pdb.set_trace()
        print(''.join(message))

    def name_logs(self, logs, prefix="train_"):
        named_logs = {}
        for l in zip(self.metrics_names, logs):
            named_logs[prefix+l[0]] = l[1]
        return named_logs

    def train_loop(self, n_epochs, verbose=True, validate=True, log_tensorboard=False):
        for epoch in range(n_epochs):
            #Reset the metric accumulators
            train_precision = 0    
            train_recall = 0
            train_f_score = 0
            train_accuracy = 0
            val_precision = 0    
            val_recall = 0
            val_f_score = 0
            val_accuracy = 0
            self.model.reset_metrics()
                
            for train_batch_id in range(self.train_generator.n_batches):
                train_X, train_Y = next(self.train_generator)  
                train_result = self.model.train_on_batch(train_X, train_Y)

                train_set_pred = self.model.predict(train_X)
                train_f_p_r = precision_recall_accuracy_f(y_true=train_Y, y_pred=train_set_pred, f_beta=0.5)

                train_f_score += train_f_p_r['f_score']
                train_precision += train_f_p_r['precision']
                train_recall += train_f_p_r['recall']
                train_accuracy += train_f_p_r['accuracy']

                if verbose == True:
                    print("train: ","Epoch:{} Batch:{}".format(epoch, train_batch_id))
                    print("loss:{:.3f} accuracy:{:.3f} precision:{:.3f} recall:{:.3f} f-score:{:.3f}".format(
                        1-train_f_p_r['f_score'], train_f_p_r['accuracy'], train_f_p_r['precision'], train_f_p_r['recall'], train_f_p_r['f_score']) 
                    )
                    print("")


                
                    #self.print_metrics(train_result)
            train_precision = train_precision / self.train_generator.n_batches
            train_recall = train_recall / self.train_generator.n_batches
            train_f_score = train_f_score / self.train_generator.n_batches
            train_accuracy = train_accuracy / self.train_generator.n_batches
            
            if verbose == True:
                    print("train: ","Epoch:{}".format(epoch))
                    print("loss:{:.3f} accuracy:{:.3f} precision:{:.3f} recall:{:.3f} f-score:{:.3f}".format(
                       1 - train_f_score, train_accuracy, train_precision, train_recall, train_f_score) 
                    )
                    

            if log_tensorboard == True:
                self.tensorboard_callback.on_epoch_end(epoch, name_logs(train_result, "train_"))                
            if validate == True:
                for val_batch_id in range(self.val_generator.n_batches):
                    val_X, val_Y = next(self.val_generator)
                    val_result = self.model.test_on_batch(val_X, val_Y, 
                                                # return accumulated metrics
                                                reset_metrics=False)
                    

                    val_set_pred = self.model.predict(val_X)
                    val_f_p_r = precision_recall_accuracy_f(y_true=val_Y, y_pred=val_set_pred, f_beta=0.5)

                    val_f_score += val_f_p_r['f_score']
                    val_precision += val_f_p_r['precision']
                    val_recall += val_f_p_r['recall']
                    val_accuracy += val_f_p_r['accuracy']


                
                    #self.print_metrics(val_result)
                val_precision = val_precision / self.val_generator.n_batches
                val_recall = val_recall / self.val_generator.n_batches
                val_f_score = val_f_score / self.val_generator.n_batches
                val_accuracy = val_accuracy / self.val_generator.n_batches
                
                if verbose == True:
                        print("val: ")
                        print("loss:{:.3f} accuracy:{:.3f} precision:{:.3f} recall:{:.3f} f-score:{:.3f}".format(
                            1 - val_f_score, val_accuracy, val_precision, val_recall, val_f_score) 
                        )
                        


                # if verbose == True:
                #     print("\nval: ")
                #     self.print_metrics(val_result)
                if log_tensorboard == True:
                    self.tensorboard_callback.on_epoch_end(epoch, name_logs(val_result, "val_"))  

            
            # if epoch % 5:
            #     checkpoint_name = "cp-{:04d}.ckpt".format(epoch)
            #     self.model.save_weights(os.path.join(self.checkpoint_dir, checkpoint_name))
        
        if log_tensorboard == True:
            self.tensorboard_callback.on_train_end(None)

    
    