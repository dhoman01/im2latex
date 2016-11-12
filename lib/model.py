import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell, seq2seq
from tensorflow.contrib import grid_rnn


class Model(object):
    def __init__(self, args, infer=False):
        self.W_conv1 = weight_variable([5,5,1,32])
        self.b_conv1 = bais_variable([32])
