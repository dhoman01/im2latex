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

"""Model and Training Configurations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class Configurations(object):
    def __init__(self):
        self.vocab_size = 180000
        self.batch_size = 16
        self.initializer_scale = 0.08
        self.lstm_dropout_keep_prob = 0.7
        self.num_examples_per_epoch=84000
        self.optimizer = "Adam"
        self.initial_learning_rate = 0.2
        self.learning_rate_decay_factor = 0.01
        self.num_epochs_per_decay = 8.0
        self.clip_gradients = 5.0
        self.max_checkpoints_to_keep = 2
        self.train_dir = "train_dir"
        self.input_file_pattern = ""
        self.values_per_input_shard = 2300
        self.input_queue_capacity_factor = 2
        self.num_input_reader_threads = 1
        self.num_preprocess_threads = 4
        # Name of the SequenceExample context feature containing image data.
        self.image_feature_name = "image/data"
        # Name of the SequenceExample feature list containing integer captions.
        self.caption_feature_name = "image/formula_ids"

        # LSTM input and output dimensionality, respectively.
        self.embedding_size = 512
        self.rnn_size = 512
        self.rnn_layers = 3
        self.attn_length = 512
