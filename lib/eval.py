import os
import time

import tensorflow as tf

from model import Model
from utils import im2latexArgumentParser

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#from utils import DataLoader

def main():
    parser = im2latexArgumentParser().parser
    args = parser.parse_args()
    eval(args)

def eval(args):
    # init model
    model = Model(args)

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        data_loader = mnist

        train_op = model.train_op
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver(tf.all_variables())
        checkpoint_path = os.path.join(args.train_dir, 'model.ckpt')
        saver.restore(sess, checkpoint_path)
        print("\ntest accuracy %g" % model.accuracy.eval(feed_dict={
            model.x: data_loader.test.images, model.y_: data_loader.test.labels, model.keep_prob: 1.0}))

if __name__ == '__main__':
    main()
