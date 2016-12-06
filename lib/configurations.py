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

"""CNN model and SAT configurations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class CNNConfigurations(object):
    # The number of outputs filters for a CNN layer
    # should be batch_size * image_height * image_width * image_channels
    self.num_outputs = 50 * 256 * 256 * 3

    # A sequence of N positive integers specifying the stride at which
    # to compute output. Can be a single integer to specify the same
    # value for all spatial dimensions.
    self.strides = [1, 2, 2, 1]

    # A sequence of N positive integers specifying the spatial
    # dimensions of of the filters. Can be a single integer
    # to specify the same value for all spatial dimensions.
    self.filter_size = [1, 2, 2, 1]

    # one of "VALID" or "SAME".
    self.padding = "VALID"

    self.initializer = tf.random_uniform_initializer(
        minval=-0.08,
        maxval=0.08)
