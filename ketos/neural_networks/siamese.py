import tensorflow as tf

class SiameseCNNBranch(tf.keras.Model):
    def __init__(self):
        super(SiameseCNNBranch, self).__init__()
        self.seq = tf.keras.models.Sequential(name="siamese_branch")
        self.conv_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(10,10), use_bias=False,
                                            kernel_initializer=tf.random_normal_initializer(),
                                            kernel_regularizer=tf.keras.regularizers.l2(2e-4))
        self.max_pool_1 = tf.keras.layers.MaxPooling2D()


        self.conv_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(7,7),
                                            kernel_initializer=tf.random_normal_initializer(),
                                            kernel_regularizer=tf.keras.regularizers.l2(2e-4),
                                            bias_initializer=tf.random_normal_initializer())
        self.max_pool_2 = tf.keras.layers.MaxPooling2D()

        self.conv_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(4,4),
                                            kernel_initializer=tf.random_normal_initializer(),
                                            kernel_regularizer=tf.keras.regularizers.l2(2e-4),
                                            bias_initializer=tf.random_normal_initializer())
        self.max_pool_3 = tf.keras.layers.MaxPooling2D()

        self.conv_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(4,4),
                                            kernel_initializer=tf.random_normal_initializer(),
                                            kernel_regularizer=tf.keras.regularizers.l2(2e-4),
                                            bias_initializer=tf.random_normal_initializer())
        self.max_pool_3 = tf.keras.layers.MaxPooling2D()

        self.flatten = tf.keras.layers.Flatten()
        self.fully_connected = tf.keras.layers.Dense(512,kernel_initializer=tf.random_normal_initializer(),
                                            kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                                            bias_initializer=tf.random_normal_initializer())

    def call(self,inputs):
        branch = self.seq(inputs)
        branch = self.conv_1(branch)
        branch = tf.nn.relu(branch)
        branch = self.max_pool_1(branch)
        branch = self.conv_2(branch)
        branch = tf.nn.relu(branch)
        branch = self.max_pool_2(branch)
        branch = self.conv_3(branch)
        branch = tf.nn.relu(branch)
        branch = self.max_pool_3(branch)
        branch = self.flatten(branch)
        branch = self.fully_connected(branch)
        branch = tf.keras.activations.sigmoid(branch)

        return branch
