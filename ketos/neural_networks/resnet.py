class ResNetBlock(tf.keras.Model):
    def __init__(self, channels, strides=1, residual_path=False):
        super(ResNetBlock, self).__init__()

        self.channels = channels
        self.strides = strides
        self.residual_path = residual_path
        self.conv_1 = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=(3,3), strides=self.strides,
                                                padding="same", use_bias=False,
                                                kernel_initializer=tf.random_normal_initializer())
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.conv_2 = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=(3,3), strides=1,
                                                padding="same", use_bias=False,
                                                kernel_initializer=tf.random_normal_initializer())
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()

        if residual_path == True:
            self.conv_down = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=(1,1), strides=self.strides,
                                                padding="same", use_bias=False,
                                                kernel_initializer=tf.random_normal_initializer())
            self.batch_norm_down = tf.keras.layers.BatchNormalization()

    def call(self,inputs, training=None):
        residual = inputs

        x = self.batch_norm_1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv_1(x)
        x = self.batch_norm_2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv_2(x)

        if self.residual_path:
            residual = self.batch_norm_down(inputs, training=training)
            residual = tf.nn.relu(residual)
            residual = self.conv_down(residual)

        x = x + residual
        return x

class ResNet(tf.keras.Model):

    def __init__(self, block_list, n_classes, initial_filters=16, **kwargs):
        super(ResNet, self).__init__(**kwargs)

        self.n_blocks = len(block_list)
        self.n_classes = n_classes
        self.block_list = block_list
        self.input_channels = initial_filters
        self.output_channels = initial_filters
        self.conv_initial = tf.keras.layers.Conv2D(filters=self.output_channels, kernel_size=(3,3), strides=1,
                                                padding="same", use_bias=False,
                                                kernel_initializer=tf.random_normal_initializer())

        self.blocks = tf.keras.models.Sequential(name="dynamic_blocks")

        for block_id in range(self.n_blocks):
            for layer_id in range(self.block_list[block_id]):
                #Frst layer of every block except the first
                if block_id != 0 and layer_id == 0:
                    block = ResNetBlock(self.output_channels, strides=2, residual_path=True)
                
                else:
                    if self.input_channels != self.output_channels:
                        residual_path = True
                    else:
                        residual_path = False
                    block = ResNetBlock(self.output_channels, residual_path=residual_path)

                self.input_channels = self.output_channels

                self.blocks.add(block)
            
            self.output_channels *= 2

        self.batch_norm_final = tf.keras.layers.BatchNormalization()
        self.average_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fully_connected = tf.keras.layers.Dense(self.n_classes)

    def call(self, inputs, training=None):

        output = self.conv_initial(inputs)

        output = self.blocks(output, training=training)
        output = self.batch_norm_final(output, training=training)
        output = tf.nn.relu(output)
        output = self.average_pool(output)
        output = self.fully_connected(output)

        return output

