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

"""Show (CNN) portion of Model. Based on https://arxiv.org/pdf/1502.03044.pdf"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

def _conv_f(inputs, num_outputs, filter_size, stride, padding,
            initializer, reuse, trainable, scope):
    return tf.contrib.layers.convolution2d(inputs,
                                    num_outputs=num_outputs,
                                    kernel_size=filter_size,
                                    stride=stride,
                                    padding=padding,
                                    rate=None,
                                    activation_fn=tf.nn.relu,
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

def cnn(inputs, config, initializer, is_training=True):
    with tf.variable_scope("cnn_input_1") as scope:
        conv = _conv_f(inputs, 375, [2,2], [2,2], "SAME", initializer, False, True, scope)
        max_pool = tf.contrib.layers.max_pool2d(conv, 2, 2, "SAME")

    with tf.variable_scope("cnn_hidden_1") as scope:
        conv_1 = _conv_f(max_pool, 125, 3, 3, "SAME", initializer, False, True, scope)
        max_pool_1 = tf.contrib.layers.max_pool2d(conv_1, 3, 3, "SAME")

    with tf.variable_scope("cnn_fc_1") as scope:
        fc_1 = _fc_f(max_pool_1, config.embedding_size, initializer, False, scope)
        if is_training:
            fc_1 = tf.nn.dropout(fc_1, 0.8)

    with tf.variable_scope("cnn_hidden_2") as scope:
        conv_2 = _conv_f(fc_1, 62, 2, 3, "SAME", initializer, False, True, scope)
        max_pool_2 = tf.contrib.layers.max_pool2d(conv_2, 2, 2, "SAME")

    with tf.variable_scope("cnn_hidden_3") as scope:
        conv_3 = _conv_f(max_pool_2, 32, 2, 2, "SAME", initializer, False, True, scope)
        max_pool_3 = tf.contrib.layers.max_pool2d(conv_3, 2, 2, "SAME")

    with tf.variable_scope("cnn_fc_2") as scope:
        fc_2 = _fc_f(max_pool_3, config.embedding_size, initializer, False, scope)
        if is_training:
            fc_2 = tf.nn.dropout(fc_2, 0.8)

    return fc_2
