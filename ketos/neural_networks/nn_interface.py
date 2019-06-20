import tensorflow as tf

class NNInterface():
    def __init__(self, neural_network, data_input):
        self.neural_network = neural_network


        self.loss_object = tf.keras.losses.BinaryCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(lr = 0.00006)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')

   