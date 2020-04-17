**************************************
Ketos Neural Networks Module Overview
**************************************

Ketos provides interfaces that can be used to build and interact with a variety of neural network architectures.
Currently, the following interfaces are available:

* CNN (CNNInterface): used to build and interact with typical 2D CNNs (e.g.: AlexNet, VGG, etc)
* ResNet (ResNetInterface): used to build and interact with typical 2D Residual Networks


The interfaces have common methods, which will be all that most users will need to create neural networks, train on their own data, load pre-trained models, etc.


Snnipets
########

Here are a few snippets commonly used


Loading a pre-trained model
***************************

.. code-block:: python
    :linenos:
        

    from ketos.neuralNetworks.resnet import ResNetInterface

    path_to_pre_trained_model = "my_model/killer_whale.kt"

    killer_whale_classifier = ResNetInterface.load_model_file(path_to_pre_trained_model)



Creating a fresh model from a recipe
*************************************

.. code-block:: python
    :linenos:
    
    from ketos.neuralNetworks.resnet import ResNetInterface

    path_to_recipe = "my_recipes/killer_whale_recipe.json"

    killer_whale_classifier = ResNetInterface.build_from_recipe_file(path_to_recipe)




Creating a fresh model from a recipe
*************************************

.. code-block:: python
    :linenos:
    
    from ketos.neuralNetworks.resnet import ResNetInterface

    path_to_recipe = "my_recipes/killer_whale_recipe.json"

    killer_whale_classifier = ResNetInterface.build_from_recipe_file(path_to_recipe)






subtitle
########

