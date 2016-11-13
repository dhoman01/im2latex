import os
import time

import tensorflow as tf
import numpy as np

from model import Model
from utils import im2latexArgumentParser, DataLoader

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#from utils import DataLoader

def main():
    parser = im2latexArgumentParser().parser
    args = parser.parse_args()
    train(args)

# def train(args):
#     # init model
#     model = Model(args)
#
#     with tf.Session() as sess:
#         data_loader = DataLoader()
#         # data_loader = mnist
#
#         train_op = model.train_op;
#         sess.run(tf.initialize_all_variables())
#         saver = tf.train.Saver(tf.all_variables())
#         checkpoint_path = os.path.join(args.train_dir, 'model.ckpt')
#         for i in range(args.num_epochs):
#         #   batch = data_loader.train.next_batch(args.batch_size)
#           batch = data_loader.next_batch()
#           print("batch[0]", batch[0])
#           print("batch[1]", batch[1])
#           if i % args.save_every == 0:
#             train_accuracy = model.accuracy.eval(feed_dict={
#                 model.x:batch[0], model.y_: batch[1], model.keep_prob: 1.0})
#             print("step %d, training accuracy %g"%(i, train_accuracy))
#             saver.save(sess, checkpoint_path, global_step=i)
#           train_op.run(feed_dict={model.x: batch[0], model.y_: batch[1], model.keep_prob: 0.5})
#         saver.save(sess, checkpoint_path)

def train(args):
    # init model
    model = Model(args)

    with tf.Session() as sess:
        data_loader = DataLoader()

        train_op = model.train_op
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver(tf.all_variables())
        checkpoint_path = os.path.join(args.train_dir, 'model.ckpt')
        for i in range(args.num_epochs):
            image, markup = enumerate(data_loader.next_batch())
            sent = np.array([markup])
            loss, _ = sess.run([model.loss, train_op], feed_dict={model.sent_placeholder: sent,
                                                                     model.x: image,
                                                                     model.dropout_placeholder: model.keep_prob})
            if i % args.save_every == 0:
                print("step %d, loss %g"%(i, loss))
                saver.save(sess, checkpoint_path, global_step=i)

        saver.save(sess, checkpoint_path)

if __name__ == '__main__':
    main()
