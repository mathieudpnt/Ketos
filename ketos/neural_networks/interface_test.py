import tensorflow as tf


class RecipeCompat():
    def __init__(self, name, func, **kwargs):
        self.name = name
        self.func = func
        self.args = kwargs

    def __call__(self, *args, **kwargs):
        result = self.func(*args, **kwargs)
        return result



# Example Metric
p = tf.keras.metrics.Precision()
dec_p = RecipeCompat("precision", p, top_k=5)

#Example Optimizer
opt = tf.keras.optimizers.Adam()
dec_opt = RecipeCompat("adam", opt, learning_rate=0.001)

# Example Loss
loss = tf.keras.losses.BinaryCrossentropy()
dec_loss = RecipeCompat('binary_crossentropy', loss, from_logits=True)



from ketos.neural_networks.resnet import ResNetInterface
from  ketos.neural_networks.losses import FScoreLoss
from ketos.neural_networks.nn_interface import RecipeCompat
import tensorflow as tf


opt = RecipeCompat('Adam',tf.keras.optimizers.Adam,lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)
loss_function = RecipeCompat('FScoreLoss', FScoreLoss)


recipe = {"block_list": [2, 2, 2, 2],
            "n_classes":2,
            "initial_filters":16,
            "optimizer": opt,
            "loss_function":loss_function,
            "metrics":[RecipeCompat('CategoricalAccuracy',tf.keras.metrics.CategoricalAccuracy)]}

ResNet = ResNetInterface.build_from_recipe(recipe)

ResNet.save_recipe("/home/fsfrazao/resnet_recipe.json")






