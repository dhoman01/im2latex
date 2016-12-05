import tensorflow as tf

class CNN(object):
    def __init__(self, mode, config):
        self.mode = mode
        self.config = config

    def is_training(self):
        return self.mode == "train"

    def build_cnn_layer(self, inputs, num_outputs, strides, filter_size,
                        padding, initializer, scope):
        conv = tf.contrib.layers.convolution2d(inputs,
                                               num_outputs=num_outputs,
                                               kernel_size=filter_size
                                               strides=strides,
                                               padding=padding,
                                               rate=1,
                                               activation_fn=None,
                                               weights_initializer=initializer,
                                               bias_initializer=initializer,
                                               trainable=self.is_training(),
                                               scope=scope)
        conv = tf.nn.relu(conv)
        pool = tf.contrib.layers.max_pool2d(conv, filter_size, strides, padding,scope=scope)
        return pool

    def get_net(self, inputs, scope):
        layer = build_cnn_layer(inputs,
                                num_outputs=self.config.num_outputs,
                                strides=self.config.strides,
                                filter_size=self.config.filter_size,
                                padding=self.config.padding,
                                initializer=self.config.initializer,
                                scope=scope)
        for i in range(self.config.num_layers - 1):
            layer = build_cnn_layer(layer,
                                    num_outputs=self.config.num_outputs,
                                    strides=self.config.strides,
                                    filter_size=self.config.filter_size,
                                    padding=self.config.padding,
                                    initializer=self.config.initializer,
                                    scope=scope)
        return layer

class ImageEmbeddings(object):
    def __init__(self, mode, config):
        self.mode = mode
        self.config = config

    def is_training(self):
        return self.mode == "train"

    def get_image_embeddings(self, images):
        cnn = CNN(self.mode, self.config.cnn_config)
        with tf.variable_scope("cnn") as scope:
            cnn_outputs = cnn.get_net(images, scope)

        with tf.variable_scope("image_embeddings") as scope:
            image_embeddings = tf.contrib.layers.fully_connected(
              inputs=cnn_outputs,
              num_outputs=self.config.embedding_size,
              activation_fn=None,
              weights_initializer=self.initializer,
              biases_initializer=self.initializer,
              scope=scope)

        # Save the embedding size in the graph.
        tf.constant(self.config.embedding_size, name="embedding_size")

        return image_embeddings

class seq2seq_Attention(object):
    def __init__(self, mode, config):
        self.mode = mode
        self.config = config

    def get_final_state(self, image_embeddings, labels):
        lstm_cell = tf.rnn_cell.LSTMBasicCell()
