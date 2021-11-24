from ketos.neural_networks.cnn import CNNInterface
from ketos.data_handling.data_feeding import BatchGenerator
import numpy as np
from ketos.neural_networks.dev_utils.nn_interface import RecipeCompat
from ketos.neural_networks.dev_utils.losses import FScoreLoss
import tensorflow as tf


recipe = {'interface': 'CNNInterface',
            'conv_set':[[8, False], [16, True]],
            'dense_set': [32],
            'n_classes':2,        
            'optimizer': RecipeCompat('Adam', tf.keras.optimizers.Adam, learning_rate=0.005),
            'loss_function': RecipeCompat('FScoreLoss', FScoreLoss),  
            'metrics': [RecipeCompat('CategoricalAccuracy',tf.keras.metrics.CategoricalAccuracy)]
    
}

cnn = CNNInterface._build_from_recipe(recipe) 

data = np.vstack([np.ones((10,8,8,1)), np.zeros((10,8,8,1))])
labels = np.concatenate([np.array([[0,1] for i in range(10)]), np.array([[1,0] for i in range(10)])])

train_generator = BatchGenerator(batch_size=5, x=data, y=labels, shuffle=True)
val_generator = BatchGenerator(batch_size=5, x=data, y=labels, shuffle=True)

cnn.checkpoint_dir = 'checkpoint'
cnn.log_dir = 'log'

cnn.train_generator = train_generator
cnn.val_generator = val_generator

cnn.train_loop(200, checkpoint_freq=1)

cnn.save_model("test_model.kt")
