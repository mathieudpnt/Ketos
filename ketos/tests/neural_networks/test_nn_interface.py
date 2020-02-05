import pytest
import numpy as np
import tensorflow as tf
from ketos.neural_networks.nn_interface import RecipeCompat, NNInterface
from ketos.neural_networks.losses import FScoreLoss
from ketos.neural_networks.metrics import Precision, Recall, Accuracy, FScore
from ketos.data_handling.data_feeding import BatchGenerator
import os
import shutil
import tables
import json


current_dir = os.path.dirname(os.path.realpath(__file__))
path_to_assets = os.path.join(os.path.dirname(current_dir),"assets")
path_to_tmp = os.path.join(path_to_assets,'tmp')

@pytest.fixture
def recipe_dict():
    recipe = {'optimizer': {'name':'Adam', 'parameters': {'learning_rate':0.005}},
               'loss_function': {'name':'FScoreLoss', 'parameters':{}},  
               'metrics': [{'name':'CategoricalAccuracy', 'parameters':{}}]
        
    }
    return recipe

@pytest.fixture
def MLPInterface_subclass():
    """ A simple MLP inheriting from NNInterface
    """
    
    class MLP(tf.keras.Model):
        def __init__(self, n_neurons, activation):
            super(MLP, self).__init__()

            self.dense = tf.keras.layers.Dense(n_neurons, activation=activation)
            self.final_node = tf.keras.layers.Dense(2, 'softmax')

        def call(self, inputs):
            print(inputs.shape)
            # output = self.dense(inputs)
            # output = tf.expand_dims(output, -1)
            # print(output.shape)
            output = self.dense(inputs)
            print(output.shape)
            output = self.final_node(output)
            print(output.shape)
            return output

            
    class MLPInterface(NNInterface):

        @classmethod
        def transform_train_batch(cls, x, y, n_classes=2):
            X = x
            Y = np.array([cls.to1hot(class_label=label, n_classes=n_classes) for label in y])
            return (X, Y)

        @classmethod
        def build_from_recipe(cls, recipe):
            n_neurons = recipe['n_neurons']
            activation = recipe['activation']
            optimizer = recipe['optimizer']
            loss_function = recipe['loss_function']
            metrics = recipe['metrics']
            if 'secondary_metrics' in recipe.keys():
                secondary_metrics = recipe['secondary_metrics']
            else:
                 secondary_metrics = None

            instance = cls(n_neurons=n_neurons, activation=activation, optimizer=optimizer, loss_function=loss_function, metrics=metrics, secondary_metrics=secondary_metrics)

            return instance

        @classmethod
        def read_recipe_file(cls, json_file, return_recipe_compat=True):
            
            with open(json_file, 'r') as json_recipe:
                recipe_dict = json.load(json_recipe)
            
            
            optimizer = cls.optimizer_from_recipe(recipe_dict['optimizer'])
            loss_function = cls.loss_function_from_recipe(recipe_dict['loss_function'])
            metrics = cls.metrics_from_recipe(recipe_dict['metrics'])
            if 'secondary_metrics' in recipe_dict.keys():
                secondary_metrics = cls.metrics_from_recipe(recipe_dict['secondary_metrics'])
            else:
                 secondary_metrics = None

            if return_recipe_compat == True:
                recipe_dict['optimizer'] = optimizer
                recipe_dict['loss_function'] = loss_function
                recipe_dict['metrics'] = metrics
                if 'secondary_metrics' in recipe_dict.keys():
                    recipe_dict['secondary_metrics'] = secondary_metrics
                
            else:
                recipe_dict['optimizer'] = cls.optimizer_to_recipe(optimizer)
                recipe_dict['loss_function'] = cls.loss_function_to_recipe(loss_function)
                recipe_dict['metrics'] = cls.metrics_to_recipe(metrics)
                if 'secondary_metrics' in recipe_dict.keys():
                    recipe_dict['secondary_metrics'] = cls.metrics_to_recipe(secondary_metrics)

            recipe_dict['n_neurons'] = recipe_dict['n_neurons']
            recipe_dict['activation'] = recipe_dict['activation']
            return recipe_dict

        def __init__(self, n_neurons, activation, optimizer, loss_function, metrics, secondary_metrics=None):
            #super(MLPInterface, self).__init__(optimizer, loss_function, metrics)
            self.n_neurons = n_neurons
            self.activation = activation

            self.optimizer=optimizer
            self.loss_function=loss_function
            self.metrics=metrics
            self.secondary_metrics=secondary_metrics

            self.model = MLP(n_neurons=n_neurons, activation=activation)
            self.compile_model()
            

        def write_recipe(self):
        
            recipe = {}
            recipe['optimizer'] = self.optimizer_to_recipe(self.optimizer)
            recipe['loss_function'] = self.loss_function_to_recipe(self.loss_function)
            recipe['metrics'] = self.metrics_to_recipe(self.metrics)
            if self.secondary_metrics is not None:
                recipe['secondary_metrics'] = cls.metrics_to_recipe(self.secondary_metrics)
            recipe['n_neurons'] = self.n_neurons
            recipe['activation'] = self.activation

            return recipe

    return MLPInterface


@pytest.fixture
def instance_of_MLPInterface(MLPInterface_subclass):
    path_to_file = os.path.join(path_to_assets, "recipes/basic_recipe.json")
    recipe = NNInterface.read_recipe_file(path_to_file)

    h5 = tables.open_file(os.path.join(path_to_assets, "vectors_1_0.h5"), 'r')
    train_table = h5.get_node("/train")
    val_table = h5.get_node("/val")
    test_table = h5.get_node("/test")

    train_generator = BatchGenerator(batch_size=5, hdf5_table=train_table, instance_function=MLPInterface_subclass.transform_train_batch, x_field='data', y_field='label')
    val_generator = BatchGenerator(batch_size=5, hdf5_table=val_table, instance_function=MLPInterface_subclass.transform_train_batch, x_field='data', y_field='label')
    test_generator = BatchGenerator(batch_size=5, hdf5_table=train_table, instance_function=MLPInterface_subclass.transform_train_batch, x_field='data', y_field='label')
    

    instance = MLPInterface_subclass(activation='relu', n_neurons=64, optimizer=recipe['optimizer'],
                         loss_function=recipe['loss_function'], metrics=recipe['metrics']) 

    instance.set_train_generator(train_generator)
    instance.set_val_generator(val_generator)
    instance.set_test_generator(test_generator)



    return instance


def test_RecipeCompat():
    opt = RecipeCompat("sgd", tf.keras.optimizers.SGD, learning_rate=0.008, momentum=0.1)
    assert opt.name == "sgd"
    assert opt.args ==  {'learning_rate': 0.008, 'momentum': 0.1}
    assert isinstance(opt.func, tf.keras.optimizers.SGD) 

    metric = RecipeCompat("accuracy", tf.keras.metrics.Accuracy)
    assert metric.name == "accuracy"
    assert metric.args == {}
    assert isinstance(metric.func, tf.keras.metrics.Accuracy) 

 

@pytest.mark.parametrize("class_label,n_classes,expected",[
    (0,2,np.array([1.0,0.0])),
    (1,2,np.array([0.0,1.0])),
    (1,3,np.array([0.0,1.0, 0.0])),
    (1,5,np.array([0.0,1.0, 0.0, 0.0, 0.0])),
    ])
def test_to1hot(class_label, n_classes, expected):
    encoded =  NNInterface.to1hot(class_label, n_classes) 
    assert(encoded == expected).all()
    with pytest.raises(IndexError):
        assert NNInterface.to1hot(class_label=1.2, n_classes=2) 
    with pytest.raises(TypeError):
        assert NNInterface.to1hot(class_label=1, n_classes=2.0)



def test_transform_train_batch():
    inputs = np.random.rand(10,5,5)
    labels = np.array([1,0,0,0,1,0,0,1,1,1])

    X,Y = NNInterface.transform_train_batch(inputs, labels)

    
    assert X.shape == (10,5,5,1)
    assert np.array_equal(X[:,:,:,0], inputs)
    assert np.array_equal(Y, np.array([[0., 1.],
                            [1., 0.],
                            [1., 0.],
                            [1., 0.],
                            [0., 1.],
                            [1., 0.],
                            [1., 0.],
                            [0., 1.],
                            [0., 1.],
                            [0., 1.]]))


def test_transform_input():
     input1 = np.random.rand(5,5)        
     input2 = np.random.rand(1,5,5)

     output1 = NNInterface.transform_input(input1)
     output2 = NNInterface.transform_input(input2)

     assert output1.shape == (1,5,5,1)
     assert output2.shape == (1,5,5,1)

     assert np.array_equal(output1[0,:,:,0], input1)
     assert np.array_equal(output2[0,:,:,0], input2[0,:,:])


def test_transform_output():
    output1 = np.array([[0.05,0.05,0.7, 0.1, 0.1]])
    output2 = np.array([[0.05,0.05,0.7, 0.1, 0.1],[0.05,0.15,0.15,0.65,0.1]])

    class1, score1 = NNInterface.transform_output(output1)
    classes2, scores2 = NNInterface.transform_output(output2)

    assert np.array_equal(class1, np.array([2])) 
    assert np.array_equal(score1, np.array([0.7]))
    
    assert np.array_equal(classes2, np.array([2, 3])) 
    assert np.array_equal(scores2, np.array([0.7, 0.65]))
    


def test_optimizer_from_recipe(recipe_dict):
    built_opt = NNInterface.optimizer_from_recipe(recipe_dict['optimizer'])
    assert isinstance(built_opt, RecipeCompat)
    assert built_opt.name == 'Adam'
    assert built_opt.args == {'learning_rate':0.005}
    assert isinstance(built_opt.func, tf.keras.optimizers.Adam)
    
    
def test_loss_function_from_recipe(recipe_dict):
    built_loss = NNInterface.loss_function_from_recipe(recipe_dict['loss_function'])
    assert isinstance(built_loss, RecipeCompat)
    assert built_loss.name == 'FScoreLoss'
    assert built_loss.args == {}
    assert isinstance(built_loss.func, FScoreLoss)
    
def test_metrics_from_recipe(recipe_dict):
    built_metrics = NNInterface.metrics_from_recipe(recipe_dict['metrics'])
    assert isinstance(built_metrics[0], RecipeCompat)
    assert built_metrics[0].name == 'CategoricalAccuracy'
    assert built_metrics[0].args == {}
    assert isinstance(built_metrics[0].func, tf.keras.metrics.CategoricalAccuracy)
    
    
def test_optimizer_to_recipe():
    recipe_compat_opt = RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005)
    optimizer_dict = NNInterface.optimizer_to_recipe(recipe_compat_opt)

    assert optimizer_dict['name'] == 'Adam'
    assert optimizer_dict['parameters'] == {'learning_rate':0.005}


def test_loss_function_to_recipe():
    recipe_compat_loss = RecipeCompat('FScoreLoss', FScoreLoss, beta=0.5)
    loss_dict = NNInterface.loss_function_to_recipe(recipe_compat_loss)

    assert loss_dict['name'] == 'FScoreLoss'
    assert loss_dict['parameters'] == {'beta':0.5}

    
def test_metrics_to_recipe():
    recipe_compat_metric = [RecipeCompat('CategoricalAccuracy', tf.keras.metrics.CategoricalAccuracy)]
    metrics_dicts = NNInterface.metrics_to_recipe(recipe_compat_metric)

    assert isinstance(metrics_dicts, list)
    assert metrics_dicts[0]['name'] == 'CategoricalAccuracy'
    assert metrics_dicts[0]['parameters'] == {}


def test_read_recipe_file():
    path_to_file = os.path.join(path_to_assets, "recipes/basic_recipe.json")
    recipe = NNInterface.read_recipe_file(path_to_file)

    assert isinstance(recipe, dict)
    assert isinstance(recipe['optimizer'], RecipeCompat)
    assert isinstance(recipe['loss_function'], RecipeCompat)
    assert isinstance(recipe['metrics'], list)
    assert isinstance(recipe['metrics'][0], RecipeCompat)

    opt = recipe['optimizer']
    assert opt.name == 'Adam'
    assert isinstance(opt.func, tf.keras.optimizers.Adam)
    assert opt.args == {"learning_rate": 0.001, "beta_1": 0.9, "beta_2": 0.999, "decay": 0.01}

    loss = recipe['loss_function']
    assert loss.name == 'FScoreLoss'
    assert isinstance(loss.func, FScoreLoss)
    assert loss.args == {}

    metric = recipe['metrics'][0]
    assert metric.name == 'CategoricalAccuracy'
    assert isinstance(metric.func, tf.keras.metrics.CategoricalAccuracy)
    assert metric.args == {}


def test_write_recipe_file(recipe_dict):
    destination = os.path.join(path_to_tmp, "test_write_recipe_file.json")
    NNInterface.write_recipe_file(destination, recipe_dict)

    read_recipe =  NNInterface.read_recipe_file(destination, return_recipe_compat=False)
    #If return_recipe_compat is False, the result should be a recipe dictionary just like the recipe_dict used to write the file
    assert read_recipe == recipe_dict
    # assert read_recipe['optimizer'] == recipe_dict['optimizer']
    # assert read_recipe['loss_function'] == recipe_dict['loss_function']
    # assert read_recipe['metrics'] == recipe_dict['metrics']

    read_recipe =  NNInterface.read_recipe_file(destination, return_recipe_compat=True)
    #If return_recipe_compat is True, the result will be a dictionary with RecipeCompat objects for the optimizer, loss_functions and metrics (actually a lis of RecipCompat objects) 

    assert read_recipe['optimizer'].name == NNInterface.optimizer_from_recipe(recipe_dict['optimizer']).name
    assert read_recipe['optimizer'].func.__class__ == NNInterface.optimizer_from_recipe(recipe_dict['optimizer']).func.__class__
    assert read_recipe['optimizer'].args == NNInterface.optimizer_from_recipe(recipe_dict['optimizer']).args

    assert read_recipe['loss_function'].name == NNInterface.loss_function_from_recipe(recipe_dict['loss_function']).name
    assert read_recipe['loss_function'].func.__class__ == NNInterface.loss_function_from_recipe(recipe_dict['loss_function']).func.__class__
    assert read_recipe['loss_function'].args == NNInterface.loss_function_from_recipe(recipe_dict['loss_function']).args
    
    assert read_recipe['metrics'][0].name == NNInterface.metrics_from_recipe(recipe_dict['metrics'])[0].name
    assert read_recipe['metrics'][0].func.__class__ == NNInterface.metrics_from_recipe(recipe_dict['metrics'])[0].func.__class__
    assert read_recipe['metrics'][0].args == NNInterface.metrics_from_recipe(recipe_dict['metrics'])[0].args
    

    
def test_instantiate_nn(MLPInterface_subclass):
    path_to_file = os.path.join(path_to_assets, "recipes/basic_recipe.json")
    recipe = NNInterface.read_recipe_file(path_to_file)
    
    MLPInterface_subclass(activation='relu', n_neurons=64, optimizer=recipe['optimizer'],
                         loss_function=recipe['loss_function'], metrics=recipe['metrics'])    

def test_save_recipe(instance_of_MLPInterface):
    path_to_file = os.path.join(path_to_assets, "recipes/basic_recipe.json")
    recipe = NNInterface.read_recipe_file(path_to_file)

    instance = instance_of_MLPInterface

    path_to_saved_recipe = os.path.join(path_to_tmp, "test_save_recipe.json")
    instance.save_recipe(path_to_saved_recipe)

    read_recipe = instance.read_recipe_file(path_to_saved_recipe)

    #assert read_recipe == recipe

    assert read_recipe['n_neurons'] == 64
    assert read_recipe['activation'] == 'relu'
    assert read_recipe['optimizer'].name ==recipe['optimizer'].name
    assert read_recipe['optimizer'].func.__class__ == recipe['optimizer'].func.__class__
    assert read_recipe['optimizer'].args == recipe['optimizer'].args

    assert read_recipe['loss_function'].name == recipe['loss_function'].name
    assert read_recipe['loss_function'].func.__class__ == recipe['loss_function'].func.__class__
    assert read_recipe['loss_function'].args == recipe['loss_function'].args
    
    assert read_recipe['metrics'][0].name == recipe['metrics'][0].name
    assert read_recipe['metrics'][0].func.__class__ == recipe['metrics'][0].func.__class__
    assert read_recipe['metrics'][0].args == recipe['metrics'][0].args


def test_write_recipe(instance_of_MLPInterface):
    recipe = instance_of_MLPInterface.write_recipe()
    
    assert recipe['n_neurons'] == 64
    assert recipe['activation'] == 'relu'

    assert recipe['optimizer'] == {'name':'Adam', 'parameters': {'beta_1': 0.9, 'beta_2': 0.999, 'decay': 0.01, 'learning_rate': 0.001}}
    assert recipe['loss_function'] == {'name':'FScoreLoss', 'parameters':{}}
    assert recipe['metrics'] == [{'name':'CategoricalAccuracy', 'parameters':{}}]


def test_train_loop(instance_of_MLPInterface):
    instance_of_MLPInterface.train_loop(n_epochs=5)


def test_train_loop_secondary_metrics(MLPInterface_subclass):


    recipe = { 'n_neurons':64,
               'activation':'relu',
               'optimizer': RecipeCompat("Adam",tf.keras.optimizers.Adam,learning_rate=0.005),
               'loss_function': RecipeCompat("FScoreLoss",FScoreLoss),  
               'metrics': [RecipeCompat('CategoricalAccuracy', tf.keras.metrics.CategoricalAccuracy)],
               'secondary_metrics': [RecipeCompat('Precision',tf.keras.metrics.Precision),
                                 RecipeCompat('Recall',tf.keras.metrics.Recall)]
               
    }


    h5 = tables.open_file(os.path.join(path_to_assets, "vectors_1_0.h5"), 'r')
    train_table = h5.get_node("/train")
    val_table = h5.get_node("/val")
    test_table = h5.get_node("/test")

    train_generator = BatchGenerator(batch_size=5, hdf5_table=train_table, instance_function=MLPInterface_subclass.transform_train_batch, x_field='data', y_field='label')
    val_generator = BatchGenerator(batch_size=5, hdf5_table=val_table, instance_function=MLPInterface_subclass.transform_train_batch, x_field='data', y_field='label')
    test_generator = BatchGenerator(batch_size=5, hdf5_table=train_table, instance_function=MLPInterface_subclass.transform_train_batch, x_field='data', y_field='label')
    

    instance = MLPInterface_subclass(activation='relu', n_neurons=64, optimizer=recipe['optimizer'],
                         loss_function=recipe['loss_function'], metrics=recipe['metrics'], secondary_metrics=recipe['secondary_metrics']) 

    instance.set_train_generator(train_generator)
    instance.set_val_generator(val_generator)
    instance.set_test_generator(test_generator)

    instance.train_loop(5)


def test_train_loop_log_csv(MLPInterface_subclass):


    recipe = { 'n_neurons':64,
               'activation':'relu',
               'optimizer': RecipeCompat("Adam",tf.keras.optimizers.Adam,learning_rate=0.005),
               'loss_function': RecipeCompat("FScoreLoss",FScoreLoss),  
               'metrics': [RecipeCompat('CategoricalAccuracy', tf.keras.metrics.CategoricalAccuracy)],
               'secondary_metrics': [RecipeCompat('Precision_Ketos',Precision),
                                     RecipeCompat('Recall_Ketos',Recall),
                                 ]
               
    }


    h5 = tables.open_file(os.path.join(path_to_assets, "vectors_1_0.h5"), 'r')
    train_table = h5.get_node("/train")
    val_table = h5.get_node("/val")

    train_generator = BatchGenerator(batch_size=5, hdf5_table=train_table, instance_function=MLPInterface_subclass.transform_train_batch, x_field='data', y_field='label')
    val_generator = BatchGenerator(batch_size=5, hdf5_table=val_table, instance_function=MLPInterface_subclass.transform_train_batch, x_field='data', y_field='label')

    instance = MLPInterface_subclass(activation='relu', n_neurons=64, optimizer=recipe['optimizer'],
                         loss_function=recipe['loss_function'], metrics=recipe['metrics'], secondary_metrics=recipe['secondary_metrics']) 

    instance.set_train_generator(train_generator)
    instance.set_val_generator(val_generator)
    instance.set_log_dir(os.path.join(path_to_tmp, "test_log_dir"))

    instance.train_loop(15, log_csv=True)

    assert os.path.isfile(os.path.join(instance.log_dir, "log.csv"))
    os.remove(os.path.join(instance.log_dir, "log.csv"))


def test_train_loop_log_tensorboard(MLPInterface_subclass):


    recipe = { 'n_neurons':64,
               'activation':'relu',
               'optimizer': RecipeCompat("Adam",tf.keras.optimizers.Adam,learning_rate=0.005),
               'loss_function': RecipeCompat("FScoreLoss",FScoreLoss),  
               'metrics': [RecipeCompat('CategoricalAccuracy', tf.keras.metrics.CategoricalAccuracy),
                            #RecipeCompat('Precision_Ketos',Precision),
                                #RecipeCompat('Recall_Ketos',Recall),
                                 #RecipeCompat('Precision',tf.keras.metrics.Precision),
                                 RecipeCompat('Recall',tf.keras.metrics.Recall)],
               'secondary_metrics': [RecipeCompat('Precision_Ketos',Precision),
                                 RecipeCompat('Recall_Ketos',Recall),
                                 RecipeCompat('Precision',tf.keras.metrics.Precision),
                                 RecipeCompat('Recall',tf.keras.metrics.Recall)]
               
    }


    h5 = tables.open_file(os.path.join(path_to_assets, "vectors_1_0.h5"), 'r')
    train_table = h5.get_node("/train")
    val_table = h5.get_node("/val")
    test_table = h5.get_node("/test")

    train_generator = BatchGenerator(batch_size=5, hdf5_table=train_table, instance_function=MLPInterface_subclass.transform_train_batch, x_field='data', y_field='label')
    val_generator = BatchGenerator(batch_size=5, hdf5_table=val_table, instance_function=MLPInterface_subclass.transform_train_batch, x_field='data', y_field='label')
    test_generator = BatchGenerator(batch_size=5, hdf5_table=train_table, instance_function=MLPInterface_subclass.transform_train_batch, x_field='data', y_field='label')
    

    instance = MLPInterface_subclass(activation='relu', n_neurons=64, optimizer=recipe['optimizer'],
                         loss_function=recipe['loss_function'], metrics=recipe['metrics'], secondary_metrics=recipe['secondary_metrics']) 

    instance.set_train_generator(train_generator)
    instance.set_val_generator(val_generator)
    instance.set_test_generator(test_generator)
    instance.set_log_dir(os.path.join(path_to_tmp, "test_log_dir"))

    instance.train_loop(15, log_tensorboard=True)

    assert os.path.isdir(os.path.join(instance.log_dir, "tensorboard_metrics"))
    shutil.rmtree(os.path.join(instance.log_dir, "tensorboard_metrics"))
