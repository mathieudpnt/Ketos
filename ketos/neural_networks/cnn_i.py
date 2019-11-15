import tensorflow as tf


conv_layers = [{'n_filters':96, "filter_shape": (11,11), 'stride': 4, 'activation':'relu', 'max_pool': True},
               {'n_filters':96, "filter_shape": (3,3), 'stride': 1, 'activation':'relu', 'max_pool': True},
               {'n_filters':256, "filter_shape": (5,5), 'stride': 1, 'activation':'relu', 'max_pool': True},
               {'n_filters':384, "filter_shape": (3,3), 'stride': 2, 'activation':'relu', 'max_pool': False},
               {'n_filters':384, "filter_shape": (3,3), 'stride': 1, 'activation':'relu', 'max_pool': False},
               {'n_filters':256, "filter_shape": (3,3), 'stride': 2, 'activation':'relu', 'max_pool': False}
               




class CNN(tf.keras.Model):
    def __init__(self, )