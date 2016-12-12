# Copyright 2016 Dustin E. Homan. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Show, attend, and tell Model. Based on https://arxiv.org/pdf/1502.03044.pdf"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf

from im2latex.ops import image_embeddings as cnn
from im2latex.ops import image_processing
from im2latex.ops import inputs as input_ops

class ShowAttendTellModel(object):
    def __init__(self, mode, config):
        self.mode = mode
        self.config = config
        self.initializer = tf.random_uniform_initializer(
                minval=-config.initializer_scale,
                maxval=config.initializer_scale)

        # Reader for the input data.
        self.reader = tf.TFRecordReader()

    def is_training(self):
        return self.mode == "train"

    def process_image(self, encoded_image, thread_id=0):
        return image_processing.process_image(encoded_image,
                                              thread_id=thread_id)

    def build_inputs(self):
        if self.mode == "inference":
            # In inference mode, images and inputs are fed via placeholders.
            image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
            input_feed = tf.placeholder(dtype=tf.int64,
                                      shape=[None],  # batch_size
                                      name="input_feed")

            # Process image and insert batch dimensions.
            images = tf.expand_dims(self.process_image(image_feed), 0)
            input_seqs = tf.expand_dims(input_feed, 1)

            # No target sequences or input mask in inference mode.
            target_seqs = None
            input_mask = None
        else:
            # Prefetch serialized SequenceExample protos.
            input_queue = input_ops.prefetch_input_data(
                self.reader,
                self.config.input_file_pattern,
                is_training=self.is_training(),
                batch_size=self.config.batch_size,
                values_per_shard=self.config.values_per_input_shard,
                input_queue_capacity_factor=self.config.input_queue_capacity_factor,
                num_reader_threads=self.config.num_input_reader_threads)

            assert self.config.num_preprocess_threads % 2 == 0
            images_and_captions = []
            for thread_id in range(self.config.num_preprocess_threads):
                serialized_sequence_example = input_queue.dequeue()
                encoded_image, caption = input_ops.parse_sequence_example(
                    serialized_sequence_example,
                    image_feature=self.config.image_feature_name,
                    caption_feature=self.config.caption_feature_name)
                image = self.process_image(encoded_image, thread_id=thread_id)
                images_and_captions.append([image, caption])

            # Batch inputs.
            queue_capacity = (2 * self.config.num_preprocess_threads *
                self.config.batch_size)
            images, input_seqs, target_seqs, input_mask = (
                input_ops.batch_with_dynamic_pad(images_and_captions,
                batch_size=self.config.batch_size,
                queue_capacity=queue_capacity))

        self.images = images
        self.input_seqs = input_seqs
        self.target_seqs = target_seqs
        self.input_mask = input_mask

    def build_model(self):
        with tf.variable_scope("image_embeddings") as scope:
            cnn_outputs = cnn.cnn(self.images, self.config, self.initializer)
            self.image_embeddings = tf.squeeze(cnn_outputs)

        tf.constant(self.config.embedding_size, name="embedding_size")

        with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
            embedding_map = tf.get_variable(
                name="map",
                shape=[self.config.vocab_size, self.config.embedding_size],
                initializer=self.initializer)
            seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs)

        self.seq_embeddings = seq_embeddings

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.rnn_size)

        if self.mode == "train":
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                            input_keep_prob=self.config.lstm_dropout_keep_prob,
                            output_keep_prob=self.config.lstm_dropout_keep_prob)

        # lstm_cell = tf.contrib.rnn.AttentionCellWrapper(lstm_cell, 1, input_size=self.config.embedding_size, state_is_tuple=True)

        if self.config.rnn_layers > 1:
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.config.rnn_layers)


        self.lstm_cell = lstm_cell

        with tf.variable_scope("attend-tell", initializer=self.initializer) as attend_scope:
            zero_state = lstm_cell.zero_state(batch_size=self.image_embeddings.get_shape()[0], dtype=tf.float32)
            _, initial_state = lstm_cell(self.image_embeddings, tf.reshape(zero_state, [-1]))

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
                sequence_length = tf.reduce_sum(self.input_mask, 1)
                lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                    inputs=self.seq_embeddings,
                                                    sequence_length=sequence_length,
                                                    initial_state=initial_state,
                                                    dtype=tf.float32,
                                                    scope=attend_scope)

        # Stack batches
        lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

        with tf.variable_scope("logits") as logits_scope:
            logits = cnn._fc_f(lstm_outputs, self.config.vocab_size, self.initializer, False, logits_scope)

        # if infering perform simple softmax
        if self.mode == "inference":
            tf.nn.softmax(logits, name="softmax")
        else:
            targets = tf.reshape(self.target_seqs, [-1])
            weights = tf.to_float(tf.reshape(self.input_mask, [-1]))

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

        learning_rate = tf.constant(self.config.initial_learning_rate)
        num_batches_per_epoch = (self.config.num_examples_per_epoch / self.config.batch_size)
        decay_steps = int(num_batches_per_epoch * self.config.num_epochs_per_decay)

        def _learning_rate_decay_fn(learning_rate, global_step):
            return tf.train.exponential_decay(learning_rate,
                                          global_step,
                                          decay_steps=decay_steps,
                                          decay_rate=self.config.learning_rate_decay_factor,
                                          staircase=True)

        learning_rate_decay_fn = _learning_rate_decay_fn

        self.train_op = tf.contrib.layers.optimize_loss(
            loss=self.total_loss,
            global_step=self.global_step,
            learning_rate=learning_rate,
            optimizer=self.config.optimizer,
            clip_gradients=self.config.clip_gradients,
            learning_rate_decay_fn=learning_rate_decay_fn)

        self.saver = tf.train.Saver(max_to_keep=self.config.max_checkpoints_to_keep)

    def build(self):
        self.build_inputs()
        self.build_model()
