import tensorflow as tf
import time

class ShowAttendTellModel(object):
    def __init__(self, mode, config):
        self.mode = mode
        self.config = config
        self.initializer = tf.random_uniform_initializer(
                minval=-config.initializer_scale,
                maxval=config.initializer_scale)

        self.inputs = tf.placeholder(tf.float32, shape=[None, 1500, 1500, 3])
        self.input_seqs = tf.placeholder(tf.int32, shape=[None, None])

        def _conv_f(inputs, num_outputs, filter_size, stride, padding,
                    initializer, reuse, trainable, scope):
            return tf.contrib.layers.convolution2d(inputs,
                                            num_outputs=num_outputs,
                                            kernel_size=filter_size,
                                            stride=stride,
                                            padding=padding,
                                            rate=None,
                                            activation_fn=None,
                                            weights_initializer=initializer,
                                            biases_initializer=initializer,
                                            reuse=reuse,
                                            trainable=trainable,
                                            scope=scope)
        def _fc_f(inputs, num_outputs, initializer, reuse, scope):
            return tf.contrib.layers.fully_connected(inputs,
                                            num_outputs=num_outputs,
                                            weights_initializer=initializer,
                                            biases_initializer=initializer,
                                            reuse=reuse,
                                            scope=scope)

        with tf.variable_scope("cnn_input_1", initializer=self.initializer) as scope:
            conv_1_1 = _conv_f(self.inputs, 750, 2, 2, "SAME", self.initializer, False, True, scope)
            relu_1_1 = tf.nn.relu(conv_1_1)
            max_pool_1_1 = tf.contrib.layers.max_pool2d(relu_1_1, 2, 2, "SAME")

        with tf.variable_scope("cnn_hidden_1", initializer=self.initializer) as scope:
            conv_1_2 = _conv_f(max_pool_1_1, 375, 2, 2, "SAME", self.initializer, False, True, scope)
            relu_1_2 = tf.nn.relu(conv_1_2)
            max_pool_1_2 = tf.contrib.layers.max_pool2d(relu_1_2, 2, 2, "SAME")

        with tf.variable_scope("cnn_fc_1", initializer=self.initializer) as scope:
            fc_1 = _fc_f(max_pool_1_2, config.embedding_size, self.initializer, False, scope)

        tf.constant(self.config.embedding_size, name="embedding_size")

        self.image_embeddings = tf.reshape(fc_1, shape=[-1, config.embedding_size])

        with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
            embedding_map = tf.get_variable(
                name="map",
                shape=[config.vocab_size, config.embedding_size],
                initializer=self.initializer)
            seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs)

        self.seq_embeddings = seq_embeddings

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(config.rnn_size)
        if mode == "train":
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                            input_keep_prob=config.lstm_dropout_keep_prob,
                            output_keep_prob=config.lstm_dropout_keep_prob)

        print("lstm_cell.output_size %s" % lstm_cell.output_size)

        # lstm_cell = tf.contrib.rnn.AttentionCellWrapper(lstm_cell, config.attn_length, input_size=512 * 512, state_is_tuple=True)

        if config.rnn_layers > 1:
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.rnn_layers)


        self.lstm_cell = lstm_cell

        with tf.variable_scope("attend-tell", initializer=self.initializer) as attend_scope:
            zero_state = lstm_cell.zero_state(batch_size=config.batch_size, dtype=tf.float32)
            _, initial_state = lstm_cell(self.image_embeddings, zero_state)

            attend_scope.reuse_variables()

            if self.mode == "inference":
            # In inference mode, use concatenated states for convenient feeding and
            # fetching.
                tf.concat(1, initial_state, name="initial_state")

                # Placeholder for feeding a batch of concatenated states.
                state_feed = tf.placeholder(dtype=tf.float32,
                    shape=[None, sum(lstm_cell.state_size)],
                    name="state_feed")
                state_tuple = tf.split(1, 2, state_feed)

                # Run a single LSTM step.
                lstm_outputs, state_tuple = lstm_cell(
                    inputs=tf.squeeze(self.seq_embeddings, squeeze_dims=[1]),
                    state=state_tuple)

                # Concatentate the resulting state.
                tf.concat(1, state_tuple, name="state")
            else:
                # Run the batch of sequence embeddings through the LSTM.
                sequence_length = tf.reduce_sum(self.seq_embeddings, 1)
                lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                    inputs=self.seq_embeddings,
                                                    sequence_length=sequence_length,
                                                    initial_state=initial_state,
                                                    dtype=tf.float32,
                                                    scope=attend_scope)

        # Stack batches
        lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

        with tf.variable_scope("logits") as logits_scope:
            logits = _fc_f(lstm_outputs, config.vocab_size, self.initializer, False, logits_scope)

        # if infering perform simple softmax
        if mode == "inference":
            tf.nn.softmax(logits, name="softmax")
        else:
            targets = tf.reshape(self.input_seqs, [-1])
            weights = tf.to_float(tf.reshape(self.input_seqs, [-1]))

            # Compute losses.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)
            batch_loss = tf.div(tf.reduce_sum(tf.mul(losses, weights)),
                              tf.reduce_sum(weights),
                              name="batch_loss")
            tf.contrib.losses.add_loss(batch_loss)
            total_loss = tf.contrib.losses.get_total_loss()

            # Add summaries.
            tf.scalar_summary("batch_loss", batch_loss)
            tf.scalar_summary("total_loss", total_loss)
            for var in tf.trainable_variables():
                tf.histogram_summary(var.op.name, var)

            self.total_loss = total_loss
            self.target_cross_entropy_losses = losses  # Used in evaluation.
            self.target_cross_entropy_loss_weights = weights  # Used in evaluation.

        self.global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.VARIABLES])

        learning_rate = tf.constant(config.initial_learning_rate)
        num_batches_per_epoch = (config.num_examples_per_epoch / config.batch_size)
        decay_steps = int(num_batches_per_epoch * config.num_epochs_per_decay)

        def _learning_rate_decay_fn(learning_rate, global_step):
            return tf.train.exponential_decay(learning_rate,
                                          global_step,
                                          decay_steps=decay_steps,
                                          decay_rate=config.learning_rate_decay_factor,
                                          staircase=True)

        learning_rate_decay_fn = _learning_rate_decay_fn

        self.train_op = tf.contrib.layers.optimize_loss(
            loss=self.total_loss,
            global_step=self.global_step,
            learning_rate=learning_rate,
            optimizer=config.optimizer,
            clip_gradients=config.clip_gradients,
            learning_rate_decay_fn=learning_rate_decay_fn)

        self.saver = tf.train.Saver(max_to_keep=config.max_checkpoints_to_keep)

    def step(self, sess, train_op, global_step, train_step_kwargs):
        start_time = time.time()
        trace_run_options = None
        run_metadata = None
        if 'should_trace' in train_step_kwargs:
            if 'logdir' not in train_step_kwargs:
                raise ValueError('logdir must be present in train_step_kwargs when '
                    'should_trace is present')
            if sess.run(train_step_kwargs['should_trace']):
                trace_run_options = config_pb2.RunOptions(
                    trace_level=config_pb2.RunOptions.FULL_TRACE)
                run_metadata = config_pb2.RunMetadata()

        total_loss ,np_global_step = sess.run([train_op, global_step],
                                              options=trace_run_options,
                                              run_metadata=run_metadata,
                                              feed_dict=train_step_kwargs['feed_dict'])

        time_elapsed = time.time() - start_time

        if run_metadata is not None:
            tl = timeline.Timeline(run_metadata.step_stats)
            trace = tl.generate_chrome_trace_format()
            trace_filename = os.path.join(train_step_kwargs['logdir'],
                'tf_trace-%d.json' % np_global_step)
            logging.info('Writing trace to %s', trace_filename)
            file_io.write_string_to_file(trace_filename, trace)
            if 'summary_writer' in train_step_kwargs:
                train_step_kwargs['summary_writer'].add_run_metadata(
                    run_metadata, 'run_metadata-%d' % np_global_step)

        if 'should_log' in train_step_kwargs:
            if sess.run(train_step_kwargs['should_log']):
                logging.info('global step %d: loss = %.4f (%.2f sec/step)',
                    np_global_step, total_loss, time_elapsed)

        if 'should_stop' in train_step_kwargs:
            should_stop = sess.run(train_step_kwargs['should_stop'])
        else:
            should_stop = False

        return total_loss, should_stop
