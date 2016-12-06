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
                                               kernel_size=filter_size,
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
        with tf.variable_scope("show") as scope:
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

class Model(object):
    def __init__(self, mode, config, buckets):
        self.mode = mode
        self.config = config
        self.buckets = buckets

        lstm_cell = tf.rnn_cell.BasicLSTMCell(self.config.rnn_size)
        if self.config.num_layers > 1:
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.config.num_layers)

        if self.is_training():
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                cell,
                input_keep_prob=self.config.lstm_droput_keep_prob,
                output_keep_prob=self.config.lstm_droput_keep_prob)

        with tf.variable_scope("attend-tell", initializer=self.initializer) as attend_scope:
             # If we use sampled softmax, we need an output projection.
            output_projection = None
            softmax_loss_function = None

            # Sampled softmax
            w_t = tf.get_variable("proj_w", [self.target_vocab_size, size], dtype=dtype)
            w = tf.transpose(w_t)
            b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
            output_projection = (w, b)

            self.output_projection = output_projection

            def sampled_loss(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])
                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(inputs, tf.float32)
                return tf.cast(tf.nn.sampled_softmax_loss(local_w_t,
                                            local_b,
                                            local_inputs,
                                            labels,
                                            num_samples,
                                            self.target_vocab_size), tf.float32)
            softmax_loss_function = sampled_loss

            def seq2seq_Attention(encoder_inputs, decoder_inputs):
                return tf.rnn.seq2seq.embedding_attention_seq2seq(
                            encoder_inputs,
                            decoder_inputs,
                            lstm_cell,
                            num_encoder_symbols=self.config.embedding_size,
                            num_decoder_symbols=self.config.vocab_size,
                            embedding_size=self.config.embedding_size,
                            output_projection=output_projection,
                            feed_previous=True,
                            dtype=tf.float32)

            self.encoder_inputs = []
            for i in range([-1][0]):
                self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))

            for i in range(buckets[-1][1] + 1):
                self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
                self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))

            targets = [self.decoder_inputs[i + 1] for i in range(len(self.decoder_inputs) - 1)]

            self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets, lambda x, y: seq2seq(x, y),
                softmax_loss_function=softmax_loss_function)

            if output_projection is not None:
                for b in range(len(buckets)):
                    self.outputs[b] = [tf.matmul(output, output_projection[0]) + output_projection[1]
                                       for output in self.outputs[b]]

    def is_training(self):
        return self.mode == "train"

    def step(self, session, encoder_inputs, decoder_inputs, target_weights, bucket_id):
        encoder_size, decoder_size = self.buckets[bucket_id]

        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Targets are shifted by 1 so add 1 extra
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.confg.batch_size], dtype=np.int32)

        output_feed = [self.losses[bucket_id]]
        for l in range(decoder_size):
            output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)

        return outputs[0], outputs[1:] # return loss, outputs
