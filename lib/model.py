import numpy as np
import tensorflow as tf

from latex_vocab import Vocab

class Model(object):
    def __init__(self, args):
        # convinence functions
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        def conv2d(x, W):
          return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
          return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')

        def max_pool_3x3(x):
          return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                                strides=[1, 3, 3, 1], padding='SAME')

        # placeholders
        self.x = tf.placeholder(tf.float32, shape=[4096, 4096], name="x")
        self.y_ = tf.placeholder(tf.float32, shape=[None, 100], name="y")
        x_image = tf.reshape(self.x, [-1,128,128,1])

        self.sent_placeholder = tf.placeholder(tf.int32, shape=[1024, None, None], name='sent_ph')
        self.dropout_placeholder = tf.placeholder(tf.float32, name='dropout_placeholder')
        self.targets_placeholder = tf.placeholder(tf.int32, shape=[args.batch_size, None], name='targets')

        self.vocab = Vocab().id2vocab
        self.vocab_size = len(self.vocab)

        with tf.variable_scope('encode_cnn'):
            # First CNN layer (encoder)
            W_conv1 = weight_variable([5,5,1,32])
            b_conv1 = bias_variable([32])

            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
            h_pool1 = max_pool_3x3(h_conv1)

        with tf.variable_scope('hidden_cnn_1'):
            # Hidden CNN Layer 1
            W_conv2 = weight_variable([5, 5, 32, 64])
            b_conv2 = bias_variable([64])

            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_3x3(h_conv2)
            print 'Img: ', h_pool2.get_shape()

        with tf.variable_scope("sent_input"):
            word_embeddings = tf.get_variable('word_embeddings', shape=[self.vocab_size, 64])
            sent_inputs = tf.nn.embedding_lookup(word_embeddings, self.sent_placeholder)
            print 'Sent:', sent_inputs.get_shape()

        with tf.variable_scope("all_input"):
            all_inputs = tf.concat(1, [h_pool2, sent_inputs])
            print 'All: ', all_inputs.get_shape()

        with tf.variable_scope("lstm"):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(args.rnn_size, forget_bias=1)
            lstm_dropout = tf.nn.rnn_cell.DropoutWrapper(lstm, input_keep_prob=self.dropout_placeholder,output_keep_prob=self.dropout_placeholder)
            stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_dropout] * args.num_layers)
            initial_state = stacked_lstm.zero_state(args.batch_size, tf.float32)
            lstm_output, final_state = tf.nn.dynamic_rnn(stacked_lstm, all_inputs, initial_state=initial_state)
            self.final_state = final_state

        with tf.variable_scope('hidden_dnn'):
            # Fully-Connected DNN 1
            W_fc1 = weight_variable([8 * 8 * 64, 1024])
            b_fc1 = bias_variable([1024])

            lstm_output = tf.reshape(lstm_output, [-1, 8 * 8 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(lstm_output, W_fc1) + b_fc1)

            self.keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        with tf.variable_scope('output_dnn'):
            # Fully-Connected DNN 2
            W_fc2 = weight_variable([1024, 100])
            b_fc2 = bias_variable([100])

        logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        targets_reshaped = tf.reshape(self.targets_placeholder,[-1])
        with tf.variable_scope('loss'):
            self.loss = loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets_reshaped, name="ce_loss"))

        with tf.variable_scope('optimizer'):
            self.train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
            #train variables
            # self.correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(self.y_,1))
            # self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            # self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, self.y_))
            # self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
