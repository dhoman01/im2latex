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
"""Train the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf

from tensorflow.python.ops import math_ops
from model import ShowAttendTellModel as Model
from configurations import Configurations
from data_loader import DataLoader

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("inception_checkpoint_file", "",
                       "Path to a pretrained inception_v3 model.")
tf.flags.DEFINE_string("train_dir", "",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_boolean("train_inception", False,
                        "Whether to train inception submodel variables.")
tf.flags.DEFINE_integer("number_of_steps", 10, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 1,
                        "Frequency at which loss and global step are logged.")

def main(_):
    config = Configurations()
    data_loader = DataLoader(config)

    g = tf.Graph()
    with g.as_default():
        model = Model("train", config)

        train_op = model.train_op
    def _feed_fn():
        batch_x, batch_y = data_loader.next_batch()
        feed_dict = {model.inputs: batch_x, model.input_seqs: batch_y}
        return feed_dict

    train_step_kwargs = {}
    train_step_kwargs['should_stop'] = math_ops.greater_equal(model.global_step, FLAGS.number_of_steps)
    train_step_kwargs['should_log'] = math_ops.equal(math_ops.mod(model.global_step, FLAGS.log_every_n_steps), 0)
    train_step_kwargs['feed_dict'] = _feed_fn()


    # tf.contrib.slim.learning.train(
    #     model.train_op,
    #     config.train_dir,
    #     train_step_fn=model.step,
    #     train_step_kwargs=train_step_kwargs,
    #     log_every_n_steps=FLAGS.log_every_n_steps,
    #     graph=g,
    #     global_step=model.global_step,
    #     number_of_steps=FLAGS.number_of_steps,
    #     saver=model.saver)

if __name__ == "__main__":
  tf.app.run()
