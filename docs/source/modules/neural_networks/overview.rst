
Overview
=========

Ketos provides interfaces that can be used to build and interact with a variety of neural network architectures.
Currently, the following interfaces are available:

* CNN (:class:`CNNInterface <ketos.neural_networks.cnn.CNNInterface>`): used to build and interact with typical 2D CNNs (e.g.: AlexNet, VGG, etc) that use spectrograms as inputs.
* CNN 1D (:class:`CNN1DInterface <ketos.neural_networks.cnn.CNN1DInterface>`): used to build and interact with 1D versions of CNN architectures that use waveforms as inputs. 
* ResNet (:class:`ResNetInterface <ketos.neural_networks.resnet.ResNetInterface>`): used to build and interact with typical 2D Residual Networks that use spectrograms as inputs.
* ResNet 1D (:class:`ResNet1DInterface <ketos.neural_networks.resnet.ResNet1DInterface>`): used to build and interact with 1D versions of Residual Networks that use waveforms as inputs.
* DenseNet (:class:`DenseNetInterface <ketos.neural_networks.densenet.DenseNetInterface>`): used to build and interact with Dense Networks that use spectrograms as inputs.
* Inception (:class:`InceptionInterface <ketos.neural_networks.inception.InceptionInterface>`): used to build and interact with Inception Networks that use spectrograms as inputs.

The interfaces have common methods, which will be all that most users will need to create neural networks, train on their own data, load pre-trained models, etc.


Here are a few snippets commonly used


Loading a pre-trained model
----------------------------

Ketos makes it easy to save and load trained models, so you can share your trained neural networks or use what others make available.
When you load a saved model, ketos will build the same network architecture used by the model you are loading and then populate it with the saved weights.


.. code-block:: python
            

    from ketos.neural_networks.resnet import ResNetInterface

    path_to_pre_trained_model = "my_model/killer_whale.kt"

    killer_whale_classifier = ResNetInterface.load_model_file(path_to_pre_trained_model)


In order to load the model like above, one needs to use the corresponding interface.
But version 2.1 introduced a more general function, which can load a model without specifying what kind of network it uses.

.. code-block:: python

    from ketos.neural_networks import load_model_file
    killer_whale_classifier = ResNetInterface.load_model_file("my_model/killer_whale.kt")


We will see how to save a network after training it, but first let's see how to create one.


Creating a fresh model from a recipe
------------------------------------

Ketos recipes are instructions that the network interfaces use to replicate an architecture and other supporting objects, like an otimizer.
Different from loading a pre-trained model, the recipe does not contain any weights, so the result is a fresh (untrained) network.
You might want to do this if you want to use the exact same architecture, optimizer, metrics and loss function as used 
by someone else (or yourself) in another project, but train on a different dataset.

It's also useful if you want to make small modifications (e.g.: change the learning rate) but still keep most things from another project.

.. code-block:: python
    
    
    from ketos.neuralNetworks.resnet import ResNetInterface

    path_to_recipe = "my_recipes/killer_whale_recipe.json"

    killer_whale_classifier = ResNetInterface.build_from_recipe_file(path_to_recipe)



More on Ketos recipes
----------------------

A recipe file is simply a .json file with the required information for a given ketos network interface. Each interface has its own recipe format and has a default recipe included in its module.
When a recipe is loaded, it is represented by a recipe dictionary. The items in this dictionary correspond to the fields in a recipe json file. While the .json recipe represents everything as
numbers and strings, in the dictionary some values are more complex objects. The  optimizer, loss function and metrics are converted to a Ketos RecipeCompat object, which facilitates the
conversion between the recipe and the actual optimizer, loss and metric objects using with the model.


The easiest way to modify a recipe is to directly edit the .json file.
Here is an exampple for the default resnet recipe:

.. code-block:: python
    

    {"block_sets":[2,2,2],"n_classes":2,"initial_filters":16,
    "optimizer": {"recipe_name": "Adam", "parameters": {"learning_rate": 0.005}}, 
    "loss_function": {"recipe_name": "FScoreLoss", "parameters": {}},
    "metrics": [{"recipe_name": "BinaryAccuracy", "parameters": {}}]}



The equivalent recipe dictionary (the default recipe in the resnet module). 

.. code-block:: python
    
    
    >>> from ketos.neuralNetworks.resnet import default_resnet_recipe

    >>> default_recipe

    {'block_sets': [2, 2, 2],
    'n_classes': 2,
    'initial_filters': 16,
    'optimizer': Adam ketos recipe,
    'loss_function': BinaryCrossentropy ketos recipe,
    'metrics': [BinaryAccuracy ketos recipe,
    Precision ketos recipe,
    Recall ketos recipe]}

    >>> default_recipe['optimizer'].args
    {'learning_rate':0.005}


You can also modify the recipe dictionary or create a new one without going through the .json. 
This can be useful for programatically generating recipes, but most users will find it easier to directly modify the .json file.


.. code-block:: python
    
    
    >>> import tensorflow as tf
    >>> from ketos.neural_networks.resnet import default_recipe, ResNetInterface
    >>> from ketos.neural_networks.dev_utils.nn_interface import RecipeCompat

    >>> custom_recipe = default_recipe
    >>> custom_recipe

    {'block_sets': [2, 2, 2],
    'n_classes': 2,
    'initial_filters': 16,
    'optimizer': Adam ketos recipe,
    'loss_function': BinaryCrossentropy ketos recipe,
    'metrics': [BinaryAccuracy ketos recipe,
    Precision ketos recipe,
    Recall ketos recipe]}

    >>> custom_recipe['block_sets'] = [2, 2, 2, 2]
    >>> custom_recipe['optimizer'] =  RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.001)
    
    # Build a model with the custom_recipe
    >>> custom_resnet = ResNetInterface._build_from_recipe(custom_recipe)

    # Save a .json recipe file with from the model 
    >>> custom_resnet.save_recipe_file("custom_recipe.json")



Training a model
-----------------

With a freshly built model, you can start training on your own data.
The recommended pipeline uses data stored in hdf5 databases and ketos batch generators to access that data.


.. code-block:: python
    
    
    import ketos.data_handling.database_interface as dbi
    from ketos.data_handling.data_feeding import BatchGenerator
    from ketos.neural_networks.resnet import ResNetInterface

    db = dbi.open_file("right_whale_database.h5", 'r')
    train_dataset = dbi.open_table(db, "/train/data")
    val_dataset = dbi.open_table(db, "/val/data")

    train_generator = BatchGenerator(batch_size=128, data_table=train_dataset,
                                 output_transform_func=ResNetInterface.transform_batch,
                                 shuffle=True, refresh_on_epoch_end=True)


    val_generator = BatchGenerator(batch_size=128, data_table=val_dataset,
                                 output_transform_func=ResNetInterface.transform_batch,
                                 shuffle=True, refresh_on_epoch_end=False)


    right_whale_classifier = ResNetInterface.build_from_recipe_file("custom_recipe.json")

    right_whale_classifier.train_generator = train_generator
    right_whale_classifier.val_generator = val_generator
    right_whale_classifier.checkpoint_dir = "my_checkpoints"
    right_whale_classifier.log_dir = "my_logs"

    right_whale_classifier.train_loop(100, log_csv=True)


For a more detailed guide on training a model, check the 'Train a ResNet classifier' tutorial.


Adding the ketos Neural Network interface to your own architectures.
--------------------------------------------------------------------

Advanced users who are able to implement their own neural network architectures might want to 
wrap them with the ketos interface. This will allow their architectures to use the same functionalities
available to the architectures implemented in Ketos (e.g.: saving/loading models,  saving/loading recipes, using the batch generators, etc).

These functionalities are implemented by the NNInterface class (found in :class:`NNInterface <ketos.neural_networks.dev_utils.nn_interface.NNInterface>` ).
The following examples demonstrate minimum integrations. For a comprehensive look into the interface, developers are encouraged to look
into this class' source code and how it is used within ketos (for example, in the CNNInterface class found in :class:`CNNInterface <ketos.neural_networks.cnn.CNNInterface>`).

Ketos uses architectures implemented with TensorFlow 2's subclassing API.
For the following examples, let's suppose you implemented a simple multilayer perceptron and now want to integrate it with Ketos.

.. code-block:: python
    
    import tensorflow as tf

    class MLP(tf.keras.Model): # doctest: +SKIP
            def __init__(self, n_neurons=128, activation='relu'):
                super(MLP, self).__init__()
         
                self.dense = tf.keras.layers.Dense(n_neurons, activation=activation)
                self.final_node = tf.keras.layers.Dense(1)
         
            def call(self, inputs):
                output = self.dense(inputs)
                output = self.dense(output)
                output = self.final_node(output)


    
    
With the architecture, the interface to the MLP can be created by subclassing NNInterface.

The simplest case will not overwrite any of the NNInterface's methods:

    .. code-block:: python

        from ketos.neural_networks.dev_utils.nn_interface import RecipeCompat, NNInterface
        
        class MLPInterface(NNInterface): 
        
            def __init__(self, n_neurons, activation, optimizer, loss_function, metrics):
                super(MLPInterface, self).__init__(optimizer, loss_function, metrics)
                self.n_neurons = n_neurons
                self.activation = activation
                self.model = MLP(n_neurons=n_neurons, activation=activation)

That might suffice in some cases. The MLPInterface we just created now has access to the all the infrastructure provided by the NNInterface.
However, you might want to overwrite some of the methods to make your interface easier to reuse.

For example, the NNInterface._transform_input() and NNInterface._transform_batch methods() are helpful to put input data in the network's expected format.
They can be used when building BatchGenerators (as seen in the 'Training a model' section above') or pre-processing data at inference time.
Although you could do whatever processing steps are necessary outside your Interface class, overwriting these methods makes it easier to keep
the code organized.

In our MLP example, there are two parameters: n_neurons and activation, with default values of 128 and 'relu', respectively.
By default, the NNInterface only includes the optimizer, loss function and metrics in the recipe and uses the default values for any other parameters defined in your architecture implementation.
However, you can add any of those parameters to the recipe too.
This is useful you want to share your interface with other users or if you envision reusing your interface with many different sets of parameters (eg.: for parameter searching).
 

.. code-block:: python

        from ketos.neural_networks.dev_utils.nn_interface import RecipeCompat, NNInterface
        
        
        class MLPInterface(NNInterface): 
        
            @classmethod
            def _build_from_recipe(cls, recipe, recipe_compat=True):
                n_neurons = recipe['n_neurons']    # take the n_neurons parameter from the recipe instead of using the default
                activation = recipe['activation']  # take the activation parameter from the recipe instead of using the default
                
                 if recipe_compat == True:
                    optimizer = recipe['optimizer']
                    loss_function = recipe['loss_function']
                    metrics = recipe['metrics']
                    
                else:
                    optimizer = cls._optimizer_from_recipe(recipe['optimizer'])
                    loss_function = cls._loss_function_from_recipe(recipe['loss_function'])
                    metrics = cls._metrics_from_recipe(recipe['metrics'])
        
                instance = cls(n_neurons=n_neurons, activation=activation, optimizer=optimizer, loss_function=loss_function, metrics=metrics)
        
                return instance
         
           @classmethod
          def _read_recipe_file(cls, json_file, return_recipe_compat=True):
                
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
                    recipe_dict['optimizer'] = cls._optimizer_to_recipe(optimizer)
                    recipe_dict['loss_function'] = cls._loss_function_to_recipe(loss_function)
                    recipe_dict['metrics'] = cls._metrics_to_recipe(metrics)
        
                recipe_dict['n_neurons'] = recipe_dict['n_neurons']    # read the n_neurons parameter from the recipe file
                recipe_dict['activation'] = recipe_dict['activation']  # read the activation parameter from the recipe file
                
                return recipe_dict
        
             def __init__(self, n_neurons, activation, optimizer, loss_function, metrics):
                super(MLPInterface, self).__init__(optimizer, loss_function, metrics)
                self.n_neurons = n_neurons
                self.activation = activation
                self.model = MLP(n_neurons=n_neurons, activation=activation)
               
        
            def _extract_recipe_dict(self):
           
                recipe = {}
                recipe['optimizer'] = self._optimizer_to_recipe(self.optimizer)
                recipe['loss_function'] = self._loss_function_to_recipe(self.loss_function)
                recipe['metrics'] = self._metrics_to_recipe(self.metrics)
                recipe['n_neurons'] = self.n_neurons
                recipe['activation'] = self.activation
                
                return recipe
