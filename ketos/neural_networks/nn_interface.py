import tensorflow as tf
from .losses import FScoreLoss
from .metrics import precision_recall_accuracy_f
import numpy as np
import json


class RecipeCompat():
    """ Makes a loss function, metric or optimizer compatible with the Ketos recipe format.


        The resulting object can be included in a ketos recipe and read by the NNInterface (or it's subclasses)

        Args:
            name: str
                The name to be used in the recipe
            func: constructor
                The loss function, metric or optimizer constructor function
            kwargs
                Any keyword arguments to be passed to the constructor (func)

        Returns:
             A RecipeCompat object


        Examples:
          >>> # Example Metric
          >>> p = tf.keras.metrics.Precision
          >>> dec_p = RecipeCompat("precision", p)

          >>> # Example Optimizer
          >>> opt = tf.keras.optimizers.Adam
          >>> dec_opt = RecipeCompat("adam", opt, learning_rate=0.001)

          >>> # Example Loss
          >>> loss = tf.keras.losses.BinaryCrossentropy
          >>> dec_loss = RecipeCompat('binary_crossentropy', loss, from_logits=True)
    
    """
    def __repr__(self):
        return "{0} ketos recipe".format(self.name)

    def __init__(self, name, func, **kwargs):
        self.name = name
        self.func = func(**kwargs)
        self.args = kwargs

    def __call__(self, *args, **kwargs):
        result = self.func(*args, **kwargs)
        return result



class NNInterface():
    """ General interface for neural network architectures in the ketos.neural_networks module.

        This class implements common methods for neural network models and is supposed to be subclassed. 
        When implementing new neural network architectures, the interface implemented in this clas can be inherited.

    Args:

        optimizer: RecipeCompat object
            An instance of the RecipeCompat class wrapping a tensorflow(-compatible) optimizer (e.g.:from tensorflow.keras.optimizers)
                
        loss_function: RecipeCompat object
            An instance of the RecipeCompat class wrappinf a tensorflow(-compatible) loss-function (e.g.:from tensorflow.keras.losses)
        
        metrics: list of RecipeCompat objects
          A list of instances of the RecipeCompat class wrapping a tensorflow(-compatible) metric (e.g.:from tensorflow.keras.metrics)

    Examples:
     
        The following example shows how a newly defined network architecture could use the interface provided by NNInterface.


        First, the new architecture must be defined. Here, a simple multi-layer perceptron is defined in the following class.

        class MLP(tf.keras.Model):
            def __init__(self, n_neurons, activation):
                super(MLP, self).__init__()

                self.dense = tf.keras.layers.Dense(n_neurons, activation=activation)
                self.final_node = tf.keras.layers.Dense(1)

            def call(self, inputs):
                output = self.dense(inputs)
                output = self.dense(output)
                output = self.final_node(output)


        With the architecture, the interface to the MLP can be created by subclassing NNInterface:
        
                class MLPInterface(NNInterface):

            @classmethod
            def build_from_recipe(cls, recipe):
                n_neurons = recipe['n_neurons']
                activation = recipe['activation']
                optimizer = recipe['optimizer']
                loss_function = recipe['loss_function']
                metrics = recipe['metrics']

                instance = cls(n_neurons=n_neurons, activation=activation, optimizer=optimizer, loss_function=loss_function, metrics=metrics)

                return instance

            @classmethod
            def read_recipe_file(cls, json_file, return_recipe_compat=True):
                
                with open(json_file, 'r') as json_recipe:
                    recipe_dict = json.load(json_recipe)
                
               
                optimizer = cls.optimizer_from_recipe(recipe_dict['optimizer'])
                loss_function = cls.loss_function_from_recipe(recipe_dict['loss_function'])
                metrics = cls.metrics_from_recipe(recipe_dict['metrics'])

                if return_recipe_compat == True:
                    recipe_dict['optimizer'] = optimizer
                    recipe_dict['loss_function'] = loss_function
                    recipe_dict['metrics'] = metrics
                else:
                    recipe_dict['optimizer'] = cls.optimizer_to_recipe(optimizer)
                    recipe_dict['loss_function'] = cls.loss_function_to_recipe(loss_function)
                    recipe_dict['metrics'] = cls.metrics_to_recipe(metrics)

                recipe_dict['n_neurons'] = recipe_dict['n_neurons']
                recipe_dict['activation'] = recipe_dict['activation']
                return recipe_dict

            def __init__(self, n_neurons, activation, optimizer, loss_function, metrics):
                #super(MLPInterface, self).__init__(optimizer, loss_function, metrics)
                self.n_neurons = n_neurons
                self.activation = activation
                self.model = MLP(n_neurons=n_neurons, activation=activation)
                self.optimizer=optimizer
                self.loss_function=loss_function
                self.metrics=metrics

            def write_recipe(self):
           
                recipe = {}
                recipe['optimizer'] = self.optimizer_to_recipe(self.optimizer)
                recipe['loss_function'] = self.loss_function_to_recipe(self.loss_function)
                recipe['metrics'] = self.metrics_to_recipe(self.metrics)
                recipe['n_neurons'] = self.n_neurons
                recipe['activation'] = self.activation

                return recipe
            
       
            
    """


    valid_optimizers = {'Adadelta':tf.keras.optimizers.Adadelta,
                        'Adagrad':tf.keras.optimizers.Adagrad,
                        'Adagrad':tf.keras.optimizers.Adagrad,
                        'Adam':tf.keras.optimizers.Adam,
                        'Adamax':tf.keras.optimizers.Adamax,
                        'Nadam':tf.keras.optimizers.Nadam,
                        'RMSprop':tf.keras.optimizers.RMSprop,
                        'SGD':tf.keras.optimizers.SGD,
                        }

    valid_losses = {'FScoreLoss':FScoreLoss,
                    'BinaryCrossentropy':tf.keras.losses.BinaryCrossentropy,
                    'CategoricalCrossentropy':tf.keras.losses.CategoricalCrossentropy,
                    'CategoricalHinge':tf.keras.losses.CategoricalHinge,
                    'CosineSimilarity':tf.keras.losses.CosineSimilarity,
                    'Hinge':tf.keras.losses.Hinge,
                    'Huber':tf.keras.losses.Huber,
                    'KLD':tf.keras.losses.KLD,
                    'LogCosh':tf.keras.losses.LogCosh,
                    'MAE':tf.keras.losses.MAE,
                    'MAPE':tf.keras.losses.MAPE,
                    'MeanAbsoluteError':tf.keras.losses.MeanAbsoluteError,
                    'MeanAbsolutePercentageError':tf.keras.losses.MeanAbsolutePercentageError,
                    'MeanSquaredError':tf.keras.losses.MeanSquaredError,
                    'MeanSquaredLogarithmicError':tf.keras.losses.MeanSquaredLogarithmicError,
                    'MSE':tf.keras.losses.MSE,
                    'MSLE':tf.keras.losses.MSLE,
                    'Poisson':tf.keras.losses.Poisson,
                    'SparseCategoricalCrossentropy':tf.keras.losses.SparseCategoricalCrossentropy,          
                    }

    valid_metrics = {'Accuracy':tf.keras.metrics.Accuracy,
                     'AUC':tf.keras.metrics.AUC,
                     'BinaryAccuracy':tf.keras.metrics.BinaryAccuracy,
                     'BinaryCrossentropy':tf.keras.metrics.BinaryCrossentropy,
                     'CategoricalAccuracy':tf.keras.metrics.CategoricalAccuracy,
                     'CategoricalCrossentropy':tf.keras.metrics.CategoricalCrossentropy,
                     'CategoricalHinge':tf.keras.metrics.CategoricalHinge,
                     'CosineSimilarity':tf.keras.metrics.CosineSimilarity,
                     'FalseNegatives':tf.keras.metrics.FalseNegatives,
                     'FalsePositives':tf.keras.metrics.FalsePositives,
                     'Hinge':tf.keras.metrics.Hinge,
                     'KLDivergence':tf.keras.metrics.KLDivergence,
                     'LogCoshError':tf.keras.metrics.LogCoshError,
                     'Mean':tf.keras.metrics.Mean,
                     'MeanAbsoluteError':tf.keras.metrics.MeanAbsoluteError,
                     'MeanAbsolutePercentageError':tf.keras.metrics.MeanAbsolutePercentageError,
                     'MeanIoU':tf.keras.metrics.MeanIoU,
                     'MeanRelativeError':tf.keras.metrics.MeanRelativeError,
                     'MeanSquaredError':tf.keras.metrics.MeanSquaredError,
                     'MeanSquaredLogarithmicError':tf.keras.metrics.MeanSquaredLogarithmicError,
                     'Poisson':tf.keras.metrics.Poisson,
                     'Precision':tf.keras.metrics.Precision,
                     'Recall':tf.keras.metrics.Recall,
                     'PrecisionAtRecall':tf.keras.metrics.PrecisionAtRecall,
                     'RootMeanSquaredError':tf.keras.metrics.RootMeanSquaredError,
                     'SensitivityAtSpecificity':tf.keras.metrics.SensitivityAtSpecificity,
                     'SparseCategoricalAccuracy':tf.keras.metrics.SparseCategoricalAccuracy,
                     'SparseCategoricalCrossentropy':tf.keras.metrics.SparseCategoricalCrossentropy,
                     'SparseTopKCategoricalAccuracy':tf.keras.metrics.SparseTopKCategoricalAccuracy,
                     'SpecificityAtSensitivity':tf.keras.metrics.SensitivityAtSpecificity,
                     'SquaredHinge':tf.keras.metrics.SquaredHinge,
                     'Sum':tf.keras.metrics.Sum,
                     'TopKCategoricalAccuracy':tf.keras.metrics.TopKCategoricalAccuracy,
                     'TrueNegatives':tf.keras.metrics.TrueNegatives,
                     'TruePositives':tf.keras.metrics.TruePositives,
                     
                     }


    @classmethod
    def to1hot(cls, class_label, n_classes=2):
        """ Create the one hot representation of class_label 

            Args:
                class_label: int
                    An integer number representing the a class label
                n_class: int
                    The number of classes available
            
            Returns:
                one_hot: numpy.array
                    The one hot representation of the class_label in a 1 x n_classes array.

            Examples:
                >>> NNInterface.to1hot(class_label=0, n_classes=2)
                array([1., 0.])

                >>> NNInterface.to1hot(class_label=1, n_classes=2)
                array([0., 1.])

                >>> NNInterface.to1hot(class_label=1, n_classes=3)
                array([0., 1., 0.])

                >>> NNInterface.to1hot(class_label=1, n_classes=5)
                array([0., 1., 0., 0., 0.])

        """
        one_hot = np.zeros(n_classes)
        one_hot[class_label]=1.0
        return one_hot
    
    @classmethod
    def transform_train_batch(cls, x, y, n_classes=2):
        """ Transforms a training batch into the format expected by the network.

            When this interface is subclassed to make new neural_network classes, this method can be overwritten to
            accomodate any transformations required. Common operations are reshaping of input arrays and parsing or one hot encoding of the labels.

            Args:
                x:numpy.array
                    The batch of inputs.
                y:numpy:array
                    The batch of labels.
                    Each label must be represented as on integer, ranging from zero to n_classes
                n_classes:int
                    The number of possible classes for one hot encoding.
                    
                

            Returns:
                X:numpy.array
                    The transformed batch of inputs
                Y:numpy.array
                    The transformed batch of labels

            Examples:
                >>> import numpy as np
                >>> # Create a batch of 10 5x5 arrays
                >>> inputs = np.random.rand(10,5,5)
                >>> inputs.shape
                (10, 5, 5)

                    
                >>> # Create a batch of 10 labels (0 or 1)
                >>> labels = np.random.choice([0,1], size=10)
                >>> labels.shape
                (10,)

                >>> transformed_inputs, transformed_labels = NNInterface.transform_train_batch(inputs, labels, n_classes=2)
                >>> transformed_inputs.shape
                (10, 5, 5, 1)

                >>> transformed_labels.shape
                (10, 2)
                
        """

        X = x.reshape(x.shape[0],x.shape[1], x.shape[2],1)
        Y = np.array([cls.to1hot(class_label=label, n_classes=n_classes) for label in y])
        return (X,Y)

    @classmethod
    def transform_input(cls,input):
        """ Transforms a training input to the format expected by the network.

            Similar to :func:`NNInterface.transform_train_batch`, but only acts on the inputs (not labels). Mostly used for inference, rather than training.
            When this interface is subclassed to make new neural_network classes, this method can be overwritten to
            accomodate any transformations required. Common operations are reshaping of an input.

            Args:
                input:numpy.array
                    An input instance. Must be of shape (n,m) or (1,n,m).

            Raises:
                ValueError if input does not have 2 or 3 dimensions.

            Returns:
                tranformed_input:numpy.array
                    The transformed batch of inputs

            Examples:
                >>> import numpy as np
                >>> # Create a batch of 10 5x5 arrays
                >>> batch_of_inputs = np.random.rand(10,5,5)
                >>> selected_input = batch_of_inputs[0]
                >>> selected_input.shape
                (5, 5)
                 
                >>> transformed_input = NNInterface.transform_input(selected_input)
                >>> transformed_input.shape
                (1, 5, 5, 1)

                # The input can also have shape=(1,n,m)
                >>> selected_input = batch_of_inputs[0:1]
                >>> selected_input.shape
                (1, 5, 5)
                 
                >>> transformed_input = NNInterface.transform_input(selected_input)
                >>> transformed_input.shape
                (1, 5, 5, 1)

                
        """
        if input.ndim == 2:
            transformed_input = input.reshape(1,input.shape[0], input.shape[1],1)
        elif input.ndim == 3:
            transformed_input = input.reshape(input.shape[0],input.shape[1], input.shape[2],1)
        else:
            raise ValueError("Expected input to have 2 or 3 dimensions, got {}({}) instead".format(input.ndims, input.shape))

        return transformed_input

    @classmethod
    def transform_output(cls,output):
        """ Transforms the network output 

            When this interface is subclassed to make new neural_network classes, this method can be overwritten to
            accomodate any transformations required. Common operations are reshaping of an input and returning the class wih the highest score instead of a softmax vector.

            Args:
                output:np.array
                    The output neural network output. An array of one or more vectors of float scores that each add to 1.0.
            Returns:
                transormed_output:tuple
                    The transformed output, where the first value is the integer representing the highest  classs in the rank the second is the respective score

            Example:
                >>> import numpy as np
                >>> output = np.array([[0.2,0.1,0.7]])  
                >>> NNInterface.transform_output(output)
                (array([2]), array([0.7]))

                >>> output = np.array([[0.2,0.1,0.7],[0.05,0.65,0.3]])  
                >>> NNInterface.transform_output(output)
                (array([2, 1]), array([0.7 , 0.65]))

        """
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
        """ Create a recipe-compatible optimizer object from an optimizer dictionary

            Used when building a model from a recipe dictionary.
            
            Args:
                optimizer: optimizer dictionay
                    A dictionary with the following keys: {'name':..., 'parameters':{...}}.
                    The 'name' value must be a valid name as defined in the `valid_optimizers` class attribute.
                    The 'parameters' value is a dictionary of keyword arguments to be used when building the optimizer
                    (e.g.: {'learning_rate':0.001, 'momentum': 0.01})


            Returns:
                built_optimizer: 
                    A recipe-compatible optimizer object.

            Raises:
                ValueError if the optimizer name is not included in the valid_optimizers class attribute.

        """

        name = optimizer['name']
        kwargs = optimizer['parameters']

        if name not in cls.valid_optimizers.keys():
            raise ValueError("Invalid optimizer name '{}'".format(name))
        built_optimizer = RecipeCompat(name,cls.valid_optimizers[name],**kwargs)

        return built_optimizer

    @classmethod
    def optimizer_to_recipe(cls, optimizer):
        """ Create an optimizer dictionary from a recipe-compatible optimizer object

            Used when creating a ketos recipe that can be used to recreate the model.

            Args:
                optimizer: instance of RecipeCompat
                    An optimizer wrapped in a RecipeCompat object
            Returns:
                recipe_optimizer: dict 
                    A dictionary with the 'name' and 'parameters' keys.

            Raises:
                ValueError if the optimizer name is not included in the valid_optimizers class attribute.

        """
        name = optimizer.name
        kwargs = optimizer.args

        if name not in cls.valid_optimizers.keys():
            raise ValueError("Invalid optimizer name '{}'".format(name))
        recipe_optimizer = {'name':name, 'parameters':kwargs}

        return recipe_optimizer

    @classmethod
    def loss_function_from_recipe(cls, loss_function):
        """ Create a recipe-compatible loss object from a loss function dictionary

            Used when building a model from a recipe dictionary.

            Args:
                loss_function: loss function dictionay
                    A dictionary with the following keys: {'name':..., 'parameters':{...}}.
                    The 'name' value must be a valid name as defined in the `valid_losses` class attribute.
                    The 'parameters' value is a dictionary of keyword arguments to be used when building the loss_function
                    (e.g.: {'from_logits':True, 'label_smoothing':0.5})


            Returns:
                built_loss: 
                    A recipe-compatible loss function object.

            Raises:
                ValueError if the loss function name is not included in the valid_losses class attribute.

        """
        name = loss_function['name']
        kwargs = loss_function['parameters']

        if name not in cls.valid_losses.keys():
            raise ValueError("Invalid loss function name '{}'".format(name))
        built_loss = RecipeCompat(name, cls.valid_losses[name],**kwargs)

        return built_loss

    @classmethod
    def loss_function_to_recipe(cls, loss_function):
        """ Create a loss function dictionary from a recipe-compatible loss function object

            Used when creating a ketos recipe that can be used to recreate the model.

            Args:
                loss_function: instance of RecipeCompat
                    A loss-function wrapped in a RecipeCompat object
            Returns:
                recipe_optimizer: dict 
                    A dictionary with the 'name' and 'parameters' keys.

            Raises:
                ValueError if the loss_function name is not included in the valid_losses class attribute.

        """
        name = loss_function.name
        kwargs = loss_function.args

        if name not in cls.valid_losses.keys():
            raise ValueError("Invalid loss function name '{}'".format(name))
        recipe_loss = {'name':name, 'parameters':kwargs}

        return recipe_loss


    @classmethod
    def metrics_from_recipe(cls, metrics):
        """ Create a list of recipe-compatible metric objects from a metrics dictionary

            Used when building a model from a recipe dictionary.

            Args:
                metrics: list of metrics dictionaries
                    a list of dictionaries with the following keys: {'name':..., 'parameters':{...}}.
                    The 'name' value must be a valid name as defined in the `valid_metrics` class attribute.
                    The 'parameters' value is a dictionary of keyword arguments to be used when building the metrics
                    (e.g.: {'from_logits':True})


            Returns:
                built_metrics: 
                    A list of recipe-compatible metric objects.

            Raises:
                ValueError if any of the metric names is not included in the valid_metrics class attribute.

        """
        
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
        """ Create a metrics dictionary from a list of recipe-compatible metric objects
         
            Used when creating a ketos recipe that can be used to recreate the model

            Args:
                metrics: list of RecipeCompat objects
                    A list of RecipeCompat objects, each wrapping a metric.
            Returns:
                recipe_metrics: list of dicts
                    A list dictionaries, each with 'name' and 'parameters' keys.

            Raises:
                ValueError if any of the metric names is not included in the valid_metrics class attribute.

        """
        
        recipe_metrics = []
        for m in metrics: 
            if m.name not in cls.valid_metrics.keys():
                raise ValueError("Invalid metric name '{}'".format(m['name']))
            recipe_metrics.append({'name':m.name, 'parameters':m.args})

        return recipe_metrics



    @classmethod
    def read_recipe_file(cls, json_file, return_recipe_compat=True):
        """ Read a .json_file containing a ketos recipe and builds a recipe dictionary.

            When subclassing NNInterface to create interfaces to new neural networks, this method can be overwritten to include other recipe fields relevant to the child class.

            Args:
                json_file:str
                    Path to the .json file (e.g.: '/home/user/ketos_recupes/my_recipe.json').
                return_recipe_compat:bool
                    If True, the returns a recipe-compatible dictionary (i.e.: where the values are RecipeCompat objects). If false, returns a recipe dictionary (i.e.: where the values are name+parameters dictionaries:  {'name':..., 'parameters':{...}})

            Returns:
                recipe_dict: dict
                    A recipe dictionary that can be used to rebuild a model.
        
        
        """
        with open(json_file, 'r') as json_recipe:
            recipe_dict = json.load(json_recipe)

        optimizer = cls.optimizer_from_recipe(recipe_dict['optimizer'])
        loss_function = cls.loss_function_from_recipe(recipe_dict['loss_function'])
        metrics = cls.metrics_from_recipe(recipe_dict['metrics'])
        if 'metrics_batch' in recipe_dict.keys():
            metrics_batch = cls.metrics_from_recipe(recipe_dict['metrics_batch'])
        else:
            metrics_batch = None

        if return_recipe_compat == True:
            recipe_dict['optimizer'] = optimizer
            recipe_dict['loss_function'] = loss_function
            recipe_dict['metrics'] = metrics
            if 'metrics_batch' in recipe_dict.keys():
                    recipe_dict['metrics_batch'] = metrics_batch
        else:
            recipe_dict['optimizer'] = cls.optimizer_to_recipe(optimizer)
            recipe_dict['loss_function'] = cls.loss_function_to_recipe(loss_function)
            recipe_dict['metrics'] = cls.metrics_to_recipe(metrics)
            if 'metrics_batch' in recipe_dict.keys():
                    recipe_dict['metrics_batch'] = cls.metrics_to_recipe(metrics_batch)

        return recipe_dict

    @classmethod
    def write_recipe_file(cls, json_file, recipe):
        """ Write a recipe dictionary into a .json file

            Args:
                json_file: str
                    Path to the .json file (e.g.: '/home/user/ketos_recipes/my_recipe.json').
                
                recipe:dict
                    A recipe dictionary containing the optimizer, loss function and metrics 
                    in addition to other parameters necessary to build an instance of the neural network.

                    recipe = {"optimizer": RecipeCompat('adam',tf.keras.optimizers.Adam),
                              "loss_function":RecipeCompat('categorical_cross_entropy',tf.keras.losses.CategoricalCrossEntropy),
                              "metrics":[RecipeCompat('categorical_accuracy',tf.keras.metrics.CategoricalAccuracy)],
                              "another_parameter:32}

        """

        with open(json_file, 'w') as json_recipe:
            json.dump(recipe, json_recipe)


    @classmethod
    def load(cls, recipe, weights_path):
        """ Load a model given a recipe dictionary and the saved weights.

            If multiple versions of the model are available in the folder indicated by weights_path the latest will be selected. 

            Args:
                recipe: dict
                    A dictionary containing the recipe
                weights_path:str
                    The path to the folder containing the saved weights.
                    Saved weights are tensorflow chekpoint. The path should not include the checkpoint files, only the folder containing them. (e.g.: '/home/user/my_saved_models/model_a/')

        """
        instance = cls.build_from_recipe(recipe) 
        latest_checkpoint = tf.train.latest_checkpoint(weights_path)
        instance.model.load_weights(latest_checkpoint)

        return instance

    @classmethod
    def load_model(cls, model_file, new_model_folder, overwrite=True):
        """ Load a model from a ketos (.kt) model file.

            Args:
                model_file:str
                    Path to the ketos(.kt) file
                new_model_folder:str
                    Path to folder where files associated with the model will be stored.
                overwrite: bool
                    If True, the 'new_model_folder' will be overwritten.
            Raises:
                FileExistsErros: If the 'new_model_folder' already exists and 'overwite' is False.

        """

        try:
            os.makedirs(new_model_folder)
        except FileExistsError:
            if overwrite == True:
                rmtree(new_model_folder)
                os.makedirs(new_model_folder)
            else:
                raise FileExistsError("Ketos needs a new folder for this model. Choose a folder name that does not exist or set 'overwrite' to True to replace the existing folder")

        with ZipFile(model_file, 'r') as zip:
            zip.extractall(path=new_model_folder)
        recipe = cls.read_recipe_file(os.path.join(new_model_folder,"recipe.json"))
        model_instance = cls.load(recipe,  os.path.join(new_model_folder, "checkpoints"))

        
        return model_instance
    
    @classmethod
    def build_from_recipe(cls, recipe):
        """ Build a model from a recipe dictionary

            When subclassing NNInterface to create interfaces for new neural networks, the method
            can be overwritten to include all the recipe fields relevant to the new class.

            Args:
                recipe:dict
                    A recipe dictionary

        """
       
        optimizer = recipe['optimizer']
        loss_function = recipe['loss_function']
        metrics = recipe['metrics']
        if 'metrics_batch' in recipe.keys():
            metrics_batch = recipe['metrics_batch']
        else:
            metrics_batch = None

        instance = cls(optimizer=optimizer, loss_function=loss_function, metrics=metrics, metrics_batch=metrics_batch)

        return instance

    def __init__(self, optimizer, loss_function, metrics, metrics_batch=None):
        
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics
        self.metrics_batch = metrics_batch

        self.model = None
        self.compile_model()
        #self.metrics_names = self.model.metrics_names

        
        self.log_dir = None
        self.checkpoint_dir = None
        self.tensorboard_callback = None
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None

    def write_recipe(self):
        """ Create a recipe dictionary from a neural network instance.

            The resulting recipe contains all the fields necessary to build  the same network architecture used by the instance calling this method.
            When subclassing NNInterface to create interfaces for new neural networks, this method can be overwritten to match the recipe fields expected by :func:`build_from_recipe`

            Returns:
                recipe:dict
                    A dictionary containing the recipe fields necessary to build the same network architecture used by the instance calling this method
        """
        recipe = {}
        recipe['optimizer'] = self.optimizer_to_recipe(self.optimizer)
        recipe['loss_function'] = self.loss_function_to_recipe(self.loss_function)
        recipe['metrics'] = self.metrics_to_recipe(self.metrics)
        if self.metrics_batch is not None:
                recipe['metrics_batch'] = cls.metrics_to_recipe(self.metrics_batch)

        return recipe

    def save_recipe(self, recipe_file):
        """ Creates a recipe from an existing neural network instance and save it into a .joson file.

            This method is a convenience method that wraps :func:`write_recipe` and :func:`write_recipe_file`

            Args:
                recipe_file:str
                    Path to .json file in which the recipe will be saved.

        """
        recipe = self.write_recipe()
        self.write_recipe_file(json_file=recipe_file, recipe=recipe)

    def save_model(self, model_file):
        """ Save the current neural network instance as a ketos (.kt) model file.

            The file includes the recipe necessary to build the network architecture and the current parameter weights.

            Args:
                model_file: str
                    Path to the .kt file. 

        """
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
        """ Compile the tensorflow model.

            Uses the instance attributes optimizer, loss_function and metrics.
        """
        self.model.compile(optimizer=self.optimizer.func,
                            loss = self.loss_function,
                            metrics = self.metrics)
        self.metrics_names = self.model.metrics_names

    def set_train_generator(self, train_generator):
        """ Link a batch generator (used for training) to this instance.

            Args:
                train_generator: instance of BatchGenerator
                    A batch generator that provides training data during the training loop 
        """
        self.train_generator = train_generator

    def set_val_generator(self, val_generator):
        """ Link a batch generator (used for validation) to this instance.

            Args:
                val_generator: instance of BatchGenerator
                    A batch generator that provides validation data during the training loop 
        """
        self.val_generator = val_generator

    def set_test_generator(self, test_generator):
        """ Link a batch generator (used for testing) to this instance.

            Args:
                test_generator: instance of BatchGenerator
                    A batch generator that provides testing data
        """
        self.test_generator = test_generator

    def set_log_dir(self, log_dir):
        """ Defines the directory where tensorboard log files can be stored

            Args:
                log_dir:str
                    Path to the directory 

        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
    def set_checkpoint_dir(self, checkpoint_dir):
        """ Defines the directory where tensorflow checkpoint files can be stored

            Args:
                log_dir:str
                    Path to the directory

        """

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def set_tensorboard_callback(self):
        """ Link tensorboard callback to this instances model, so that tensorboard logs can be saved
        """

        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        tensorboard_callback.set_model(self.model)
        
    def print_metrics(self, metric_values):
        """ Print the metric values to the screen.

            This method can be overwritten to customize the message.

            Args:
                metric_value:list
                    List of metric values. Usually returned by model.train_on_batch or generated by custom metrics.
        
        """

        message  = [self.metrics_names[i] + ": {} ".format(metric_values[i]) for i in range(len(self.metrics_names))]
        #import pdb; pdb.set_trace()
        print(''.join(message))

    def name_logs(self, logs, prefix="train_"):
        """ Attach the prefix string to each log name.

            Args:
                logs:list
                   List of log values
                prefix:str
                    Prefix to be added to the logged metric name
        
            Returns:
                named_log: zip
                    A zip iterator that yields a tuple: (prefix + log metric name, log value)
        """


        named_logs = {}
        for l in zip(self.metrics_names, logs):
            named_logs[prefix+l[0]] = l[1]
        return named_logs

    def train_loop(self, n_epochs, verbose=True, validate=True, log_tensorboard=False, log_csv=False,  ):
        for epoch in range(n_epochs):
            #Reset the metric accumulators
            batch_metrics = {}
            if self.metrics_batch is not None:
                for m in self.metrics_batch:
                    batch_metrics['train_' + m.name] = 0
                    batch_metrics['val_' + m.name] = 0
                    batch_metrics['test_' + m.name] = 0
            

            # train_precision = 0    
            # train_recall = 0
            # train_f_score = 0
            # train_accuracy = 0
            # val_precision = 0    
            # val_recall = 0
            # val_f_score = 0
            # val_accuracy = 0
            self.model.reset_metrics()
                
            for train_batch_id in range(self.train_generator.n_batches):
                train_X, train_Y = next(self.train_generator)  
                train_result = self.model.train_on_batch(train_X, train_Y)

                if self.metrics_batch is not None:
                    train_set_pred = self.model.predict(train_X)
                    for m in self.metrics_batch:
                        batch_metrics['train_' + m.name] += m.func(y_true=train_Y, y_pred=train_set_pred)
                        

                # train_set_pred = self.model.predict(train_X)
                # train_f_p_r = precision_recall_accuracy_f(y_true=train_Y, y_pred=train_set_pred, f_beta=0.5)

                # train_f_score += train_f_p_r['f_score']
                # train_precision += train_f_p_r['precision']
                # train_recall += train_f_p_r['recall']
                # train_accuracy += train_f_p_r['accuracy']

                # if verbose == True:
                #     print("train: ","Epoch:{} Batch:{}".format(epoch, train_batch_id))
                #     print("loss:{:.3f} accuracy:{:.3f} precision:{:.3f} recall:{:.3f} f-score:{:.3f}".format(
                #         1-train_f_p_r['f_score'], train_f_p_r['accuracy'], train_f_p_r['precision'], train_f_p_r['recall'], train_f_p_r['f_score']) 
                #     )
                #     print("")


                
                    #self.print_metrics(train_result)
            # train_precision = train_precision / self.train_generator.n_batches
            # train_recall = train_recall / self.train_generator.n_batches
            # train_f_score = train_f_score / self.train_generator.n_batches
            # train_accuracy = train_accuracy / self.train_generator.n_batches

            if self.metrics_batch is not None:
                for m in self.metrics_batch:
                    batch_metrics['train_' + m.name] = batch_metrics['train_' + m.name] / self.train_generator.n_batches
            
            # if verbose == True:
            #         print("====================================================================================")
            #         print("train: ","Epoch:{}".format(epoch))
            #         print("loss:{:.3f} accuracy:{:.3f} precision:{:.3f} recall:{:.3f} f-score:{:.3f}".format(
            #            1 - train_f_score, train_accuracy, train_precision, train_recall, train_f_score) 
            #         )
                    
            if verbose == True and self.metrics_batch is not None:
                metrics_values_msg = ""
                for m in self.metrics_batch:
                    metrics_values_msg += 'train_' + m.name + ": " + str(round(batch_metrics['train_' + m.name],3))
                
                print("====================================================================================")
                print("train: ","Epoch:{}".format(epoch))
                print(metrics_values_msg)

            if log_tensorboard == True:
                self.tensorboard_callback.on_epoch_end(epoch, name_logs(train_result, "train_"))                


            if validate == True:
                for val_batch_id in range(self.val_generator.n_batches):
                    val_X, val_Y = next(self.val_generator)
                    val_result = self.model.test_on_batch(val_X, val_Y, 
                                                # return accumulated metrics
                                                reset_metrics=False)
                    
                    if self.metrics_batch is not None:
                        val_set_pred = self.model.predict(val_X)
                        for m in self.metrics_batch:
                            batch_metrics['val_' + m.name] += m.func(y_true=val_Y, y_pred=val_set_pred)


                    # val_set_pred = self.model.predict(val_X)
                    # val_f_p_r = precision_recall_accuracy_f(y_true=val_Y, y_pred=val_set_pred, f_beta=0.5)

                    # val_f_score += val_f_p_r['f_score']
                    # val_precision += val_f_p_r['precision']
                    # val_recall += val_f_p_r['recall']
                    # val_accuracy += val_f_p_r['accuracy']


                
                    #self.print_metrics(val_result)
                # val_precision = val_precision / self.val_generator.n_batches
                # val_recall = val_recall / self.val_generator.n_batches
                # val_f_score = val_f_score / self.val_generator.n_batches
                # val_accuracy = val_accuracy / self.val_generator.n_batches

                if self.metrics_batch is not None:
                    
                    for m in self.metrics_batch:
                        batch_metrics['val_' + m.name] = batch_metrics['val_' + m.name] / self.train_generator.n_batches
                
                if verbose == True and self.metrics_batch is not None:
                    metrics_values_msg = ""
                    for m in self.metrics_batch:
                        metrics_values_msg += 'val_' + m.name + ": " + str(round(batch_metrics['val_' + m.name],3))
                    
                    print("====================================================================================")
                    print("Val: ")
                    print(metrics_values_msg)
                            


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
        """ Run the model on one input

            Args:
                input: numpy.array
                    The input in the shape expected by :func:`transform_input`
                return_raw_output:bool
                    If False, the model output will be transformed by :func:`transform_output`.
                    If true, the model output will be returned without any modifications.

            Returns:
                output
                    The model output
                
        """
        input = self.transform_input(input)
        output = self.model.predict(input)
        
        if not return_raw_output:
            return self.transform_output(output)
        else:
            return output

    
    def run_on_batch(self, input_batch, return_raw_output=False, transform_input=True):
        """ Run the model on a batch of inputs

            Args:
                input_batch: numpy.array
                    The  batch of inputs 
                transform_input:bool
                    If True, the input_batch is transformed by :func:`transform_input`
                return_raw_output:bool
                    If False, the model output will be transformed by :func:`transform_output`.
                    If true, the model output will be returned without any modifications.

            Returns:
                output
                    The corresponding batch of model outputs
        """

        if transform_input == True:
            input_batch = self.transform_input(input_batch)
        output = self.model.predict(input_batch)
        
        if not return_raw_output:
            return self.transform_output(output)
        else:
            return output


            

    
    
    