import tensorflow as tf

class NNInterface():
    """ General interface for neural network architectures in the ketos.neural_networks module.

        This class implements common methods for neural network models.

    Args:
        neural_network: 
            An instance of one of the architectures available in the neural_networks module.
        data_input: data_feeder
            An object containing or able to read the data.

    
    """
    def __init__(self, neural_network, data_input):
        self.neural_network = neural_network


        self.loss_object = tf.keras.losses.BinaryCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(lr = 0.00006)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')

    @tf.function
    def train_step(input, label):
    with tf.GradientTape() as tape:
        predictions = self.neural_network(input)
        loss = self.loss_object(label, predictions)
    gradients = tape.gradient(loss, self.neural_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, self.neural_network.trainable_variables))

    self.train_loss(loss)
    self.train_accuracy(label, predictions)
    

    @tf.function
    def val_step(input, label):
        predictions = self.neural_network(input)
        t_loss = self.loss_object(label, predictions)

        self.val_loss(t_loss)
        self.val_accuracy(label, predictions)

    @tf.function
    def predict_step(input):
        predictions = self.neural_network(input)
    return predictions        

    

