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

import os
import tensorflow as tf
import numpy as np
from .losses import FScoreLoss
from .metrics import precision_recall_accuracy_f
from .nn_interface import RecipeCompat
import json
from zipfile import ZipFile
from glob import glob
from shutil import rmtree

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
    def to1hot(cls, class_label, n_classes=2):
        one_hot = np.zeros(2)
        one_hot[class_label]=1.0
        return one_hot
    
    @classmethod
    def transform_train_batch(cls,x,y):
        X = x.reshape(x.shape[0],x.shape[1], x.shape[2],1)
        Y = np.array([cls.to1hot(sp) for sp in y])
        return (X,Y)

    @classmethod
    def transform_input(cls,input):
        if input.ndim == 2:
            transformed_input = input.reshape(1,input.shape[0], input.shape[1],1)
        elif input.ndim == 3:
            transformed_input = input.reshape(input.shape[0],input.shape[1], input.shape[2],1)
        else:
            raise ValueError("Expected input to have 2 or 3 dimensions, got {}({}) instead".format(input.ndims, input.shape))

        return transformed_input

    @classmethod
    def transform_output(cls,output):
        max_class = np.argmax(output, axis=-1)
        if output.shape[0] == 1:
            max_class_conf = output[0][max_class]
            transformed_output = (max_class[0], max_class_conf[0])
        elif output.shape[0] > 1:
            max_class_conf = np.array([output[i][c] for i, c in enumerate(max_class)])

        transformed_output = (max_class, max_class_conf)
        
        return transformed_output


    @classmethod
    def optimizer_from_recipe(cls, optimizer):
        name = optimizer['name']
        kwargs = optimizer['parameters']

        if name not in cls.valid_optimizers.keys():
            raise ValueError("Invalid optimizer name '{}'".format(name))
        built_optimizer = RecipeCompat(name,cls.valid_optimizers[name],**kwargs)

        return built_optimizer

    @classmethod
    def optimizer_to_recipe(cls, optimizer):
        name = optimizer.name
        kwargs = optimizer.args

        if name not in cls.valid_optimizers.keys():
            raise ValueError("Invalid optimizer name '{}'".format(name))
        recipe_optimizer = {'name':name, 'parameters':kwargs}

        return recipe_optimizer

    @classmethod
    def loss_function_from_recipe(cls, loss_function):
        name = loss_function['name']
        kwargs = loss_function['parameters']

        if name not in cls.valid_losses.keys():
            raise ValueError("Invalid loss function name '{}'".format(name))
        built_loss = RecipeCompat(name, cls.valid_losses[name],**kwargs)

        return built_loss

    @classmethod
    def loss_function_to_recipe(cls, loss_function):
        name = loss_function.name
        kwargs = loss_function.args

        if name not in cls.valid_losses.keys():
            raise ValueError("Invalid loss function name '{}'".format(name))
        recipe_loss = {'name':name, 'parameters':kwargs}

        return recipe_loss


    @classmethod
    def metrics_from_recipe(cls, metrics):
        
        built_metrics = []
        for m in metrics:
            name = m['name']
            kwargs = m['parameters']
             
            if name not in cls.valid_metrics.keys():
                raise ValueError("Invalid metric name '{}'".format(m['name']))
            built_metrics.append(RecipeCompat(name, cls.valid_metrics[name], **kwargs))

        return built_metrics

    @classmethod
    def metrics_to_recipe(cls, metrics):
        # name = metrics.name
        # kwargs = metrics.parameters

        recipe_metrics = []
        for m in metrics: 
            if m.name not in cls.valid_metrics.keys():
                raise ValueError("Invalid metric name '{}'".format(m['name']))
            recipe_metrics.append({'name':m.name, 'parameters':m.args})

        return recipe_metrics



    @classmethod
    def read_recipe_file(cls, json_file):
        with open(json_file, 'r') as json_recipe:
            recipe_dict = json.load(json_recipe)

        optimizer = cls.optimizer_from_recipe(recipe_dict['optimizer'])
        loss_function = cls.loss_function_from_recipe(recipe_dict['loss_function'])
        metrics = cls.metrics_from_recipe(recipe_dict['metrics'])

        recipe_dict['optimizer'] = optimizer
        recipe_dict['loss_function'] = loss_function
        recipe_dict['metrics'] = metrics

        return recipe_dict

    @classmethod
    def write_recipe_file(cls, json_file, recipe):
        with open(json_file, 'w') as json_recipe:
            json.dump(recipe, json_recipe)


    @classmethod
    def load(cls, recipe, weights_path):
        instance = cls.build_from_recipe(recipe) 
        latest_checkpoint = tf.train.latest_checkpoint(weights_path)
        instance.model.load_weights(latest_checkpoint)

        return instance

    @classmethod
    def load_model(cls, model_file):
        tmp_dir = "tmp_model_files"
        try:
            os.makedirs(tmp_dir)
        except FileExistsError:
            rmtree(tmp_dir)
            os.makedirs(tmp_dir)

        with ZipFile(model_file, 'r') as zip:
            zip.extractall(path=tmp_dir)
        recipe = cls.read_recipe_file(os.path.join(tmp_dir,"recipe.json"))
        model_instance = cls.load(recipe,  os.path.join(tmp_dir, "checkpoints"))

        rmtree(tmp_dir)

        return model_instance


        
            

    


    
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

    def write_recipe(self):
        recipe = {}
        recipe['block_list'] = self.block_list
        recipe['n_classes'] = self.n_classes
        recipe['initial_filters'] = self.initial_filters
        recipe['optimizer'] = self.optimizer_to_recipe(self.optimizer)
        recipe['loss_function'] = self.loss_function_to_recipe(self.loss_function)
        recipe['metrics'] = self.metrics_to_recipe(self.metrics)

        return recipe



    def save_recipe(self, recipe_file):
        recipe = self.write_recipe()
        self.write_recipe_file(json_file=recipe_file, recipe=recipe)

    def save_model(self, model_file):
        recipe_path = os.path.join(self.checkpoint_dir, 'recipe.json')
        with ZipFile(model_file, 'w') as zip:
            
            latest = tf.train.latest_checkpoint(self.checkpoint_dir)
            checkpoints = glob(latest + '*')                                                                                                                 
            self.save_recipe(recipe_path)
            zip.write(recipe_path, "recipe.json")
            zip.write(os.path.join(self.checkpoint_dir, "checkpoint"), "checkpoints/checkpoint")
            for c in checkpoints:
                 zip.write(c, os.path.join("checkpoints", os.path.basename(c)))            

        os.remove(recipe_path)



    
    def compile_model(self):
        self.model.compile(optimizer=self.optimizer.func,
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
                    print("====================================================================================")
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
                        print("\nval: ")
                        print("loss:{:.3f} accuracy:{:.3f} precision:{:.3f} recall:{:.3f} f-score:{:.3f}".format(
                            1 - val_f_score, val_accuracy, val_precision, val_recall, val_f_score) 
                        )
                        print("====================================================================================")

                        


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

        
    def run(self, input, return_raw_output=False):
        input = self.transform_input(input)
        output = self.model.predict(input)
        
        if not return_raw_output:
            return self.transform_output(output)
        else:
            return output

    
    def run_on_batch(self, input_batch, return_raw_output=False, transform_input=True):
        if transform_input == True:
            input_batch = self.transform_input(input_batch)
        output = self.model.predict(input_batch)
        
        if not return_raw_output:
            return self.transform_output(output)
        else:
            return output


            

    
    
    