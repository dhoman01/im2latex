from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from collections import namedtuple
from datetime import datetime
import csv
import os.path
import random
import sys
import threading
import linecache

from PIL import Image, ImageChops

import nltk.tokenize
import numpy as np
import tensorflow as tf

tf.flags.DEFINE_string("images_dir", os.path.join(os.path.expanduser('~'), "im2latex/data_dir/formula_images"),
                       "Directory containing images of LaTeX formulas")

tf.flags.DEFINE_integer("num_threads", 8,
                        "Number of threads to preprocess the images.")

tf.flags.DEFINE_string("image_format", "PNG",
                       "The encoding of the input images")

tf.flags.DEFINE_string("pattern", "*.png",
                       "The file pattern to glob the images")

FLAGS = tf.flags.FLAGS

def _crop_with_bbox(image):
    bg = Image.new(image.mode, image.size, image.getpixel((0,0)))
    diff = ImageChops.difference(image,bg)
    bbox = diff.getbbox()
    return image.crop(bbox)

def _process_image_files(thread_index, ranges, images):
    num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for i in range(num_images_in_thread):
        imagename =  images[ranges[thread_index][0] + i]

        image = Image.open(os.path.join(FLAGS.images_dir, imagename))

        image = _crop_with_bbox(image)
        image.save(os.path.join(FLAGS.images_dir, imagename), FLAGS.image_format)
        counter += 1
        if not counter % 100:
            print("%s [thread %d]: Processed %d of %d images in thread batch." %
                    (datetime.now(), thread_index, counter, num_images_in_thread))
            sys.stdout.flush()
        image.close()
    print("%s [thread %d]: Cropped %d images to bounding box" %
            (datetime.now(), thread_index, counter))
    sys.stdout.flush()

def _process(images):
    num_threads = FLAGS.num_threads
    spacing = np.linspace(0, len(images), num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    coord = tf.train.Coordinator()

    print("%s: Launching %d threads for spacing: %s" % (datetime.now(), num_threads, ranges))
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, images)
        t = threading.Thread(target=_process_image_files, args=args)
        t.start()
        threads.append(t)

    coord.join(threads)
    print("%s: Finished processing all %d images." % (datetime.now(), len(images)))

def main(unused_argv):
    assert tf.gfile.IsDirectory(FLAGS.images_dir), (
        "Please indicate an existing directory for the images")

    images = tf.gfile.Glob(os.path.join(FLAGS.images_dir, FLAGS.pattern))

    _process(images)

if __name__ == "__main__":
    tf.app.run()
