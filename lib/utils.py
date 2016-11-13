import os
import argparse
import linecache
import random
from PIL import Image, ImageFilter

import numpy as np

class im2latexArgumentParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--data_dir', type=str, default='data_dir',
                             help='directory containing the sub-dir "formula_images" and the files "im2latex_train.lst" and "im2latex_formulas.lst"')
        self.parser.add_argument('--train_dir', type=str, default='train_dir',
                             help='directory to save TF checkpoints')
        self.parser.add_argument('--rnn_size', type=int, default=128,
                            help='size of RNN hidden state')
        self.parser.add_argument('--num_layers', type=int, default=4,
                            help='number of layers in the RNN')
        self.parser.add_argument('--model', type=str, default='lstm',
                            help='rnn, gru, lstm, gridlstm, gridgru')
        self.parser.add_argument('--batch_size', type=int, default=50,
                            help='minibatch size')
        self.parser.add_argument('--seq_length', type=int, default=50,
                            help='RNN sequence length')
        self.parser.add_argument('--num_epochs', type=int, default=20000,
                            help='number of epochs')
        self.parser.add_argument('--save_every', type=int, default=1000,
                            help='save frequency')
        self.parser.add_argument('--grad_clip', type=float, default=5.,
                            help='clip gradients at this value')
        self.parser.add_argument('--learning_rate', type=float, default=0.002,
                            help='learning rate')
        self.parser.add_argument('--decay_rate', type=float, default=0.97,
                            help='decay rate for rmsprop')

class DataLoader(object):
    def __init__(self, data_dir="data_dir", batch_size=50, num_epochs=100, num_threads=1, train=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_threads = num_threads
        self.train = train
        self.image_path = os.path.join(self.data_dir, "formula_images/")

        self.create_batches()
        self.reset_batch_pointer()

    def create_batches(self):
        if(self.train):
            imagelist = os.path.join(self.data_dir, "im2latex_train.lst")
        else:
            imagelist = os.path.join(self.data_dir, "im2latex_test.lst")

        self.batches = []

        with open(imagelist) as f:
            for line in f:
                values = line.split()
                self.batches.append([values[1] + ".png", values[0]])

    def reset_batch_pointer(self):
        self.pointer = 0

    def next_batch(self):
        batch = self.batches[self.pointer]
        self.pointer += 1
        example, label = self.get_example(batch)
        return example, label

    def get_example(self, batch):
        raw_image, label = self.decoder(batch)
        example = self.preprocess(raw_image)
        print("size %d" % example.size)
        return example, label

    def decoder(self, batch):
        image_path = os.path.join(self.data_dir, "formula_images")
        raw_image = Image.open(os.path.join(image_path, batch[0]))

        label_path = os.path.join(self.data_dir, "im2latex_formulas.lst")
        formula_line = int(batch[1]) + 1
        print("formula_line: %d" % formula_line)
        print("key: %s" % batch[0])
        label = linecache.getline(label_path, formula_line).rstrip('\n')
        return raw_image, label

    def preprocess(self, raw_image):
        cropped = raw_image.crop(raw_image.getbbox())
        example = Image.new("L", (4096, 4096), 255)
        example.paste(cropped, cropped.getbbox())
        if random.uniform(0,1) > .9:
            example = example.filter(ImageFilter.GaussianBlur(radius=1))
        example = np.array(example)
        return example
