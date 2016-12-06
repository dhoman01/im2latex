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



import nltk.tokenize
import numpy as np
import tensorflow as tf


tf.flags.DEFINE_string("train_formulas_file", os.path.join(os.path.expanduser('~'), "im2latex/data_dir/im2latex_train.lst"),
                       "Training formulas LST file.")
tf.flags.DEFINE_string("val_formulas_file", os.path.join(os.path.expanduser('~'), "im2latex/data_dir/im2latex_validate.lst"),
                       "Validation formulas LST file.")
tf.flags.DEFINE_string("test_formulas_file", os.path.join(os.path.expanduser('~'), "im2latex/data_dir/im2latex_test.lst"),
                       "Test formulas LST file.")

tf.flags.DEFINE_string("formulas_file", os.path.join(os.path.expanduser('~'), "im2latex/data_dir/im2latex_formulas.lst"),
                       "Latex Formula LST file.")

tf.flags.DEFINE_string("output_dir", os.path.join(os.path.expanduser('~'), "im2latex/data_dir/"), "Output data directory.")

tf.flags.DEFINE_integer("train_shards", 256,
                        "Number of shards in training TFRecord files.")
tf.flags.DEFINE_integer("val_shards", 4,
                        "Number of shards in validation TFRecord files.")
tf.flags.DEFINE_integer("test_shards", 8,
                        "Number of shards in testing TFRecord files.")

tf.flags.DEFINE_string("start_word", "<GO>",
                       "Special word added to the beginning of each sentence.")
tf.flags.DEFINE_string("end_word", "<EOS>",
                       "Special word added to the end of each sentence.")
tf.flags.DEFINE_string("unknown_word", "<UNK>",
                       "Special word meaning 'unknown'.")
tf.flags.DEFINE_integer("min_word_count", 1,
                        "The minimum number of occurrences of each word in the "
                        "training set for inclusion in the vocabulary.")
tf.flags.DEFINE_string("word_counts_output_file", "word_counts.txt",
                       "Output vocabulary file of word counts.")

tf.flags.DEFINE_integer("num_threads", 8,
                        "Number of threads to preprocess the images.")

tf.flags.DEFINE_string("image_format", "png",
                       "The encoding of the input images")

FLAGS = tf.flags.FLAGS

ImageMetadata = namedtuple("ImageMetadata",
                            ["image_id", "filename", "formula_line"])

class Vocabulary(object):
  """Simple vocabulary wrapper."""

  def __init__(self, vocab, unk_id):
    """Initializes the vocabulary.
    Args:
      vocab: A dictionary of word to word_id.
      unk_id: Id of the special 'unknown' word.
    """
    self._vocab = vocab
    self._unk_id = unk_id

  def word_to_id(self, word):
    """Returns the integer id of a word string."""
    if word in self._vocab:
      return self._vocab[word]
    else:
      return self._unk_id

class ImageDecoder(object):
    """Helper class for decoding images in TensorFlow."""

    def __init__(self):
        # Create a single TensorFlow Session for all image decoding calls.
        self._sess = tf.Session()

        # TensorFlow ops for Image decoding.
        self._encoded_image = tf.placeholder(dtype=tf.string)
        if FLAGS.image_format == "jpeg":
            self._decode_image = tf.image.decode_jpeg(self._encoded_image, channels=3)
        elif FLAGS.image_format == "png":
            self._decode_image = tf.image.decode_png(self._encoded_image, channels=3)

    def decode_image(self, encoded_image):
        image = self._sess.run(self._decode_image,
            feed_dict={self._encoded_image: encoded_image})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def _get_formula(formula_line):
    return linecache.getline(FLAGS.formulas_file, formula_line)

def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))

def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

def _to_sequence_example(image, decoder, vocab):
    """Builds a SequenceExample proto for an image-formula pair.
    Args:
    image: An ImageMetadata object.
    decoder: An ImageDecoder object.
    vocab: A Vocabulary object.
    Returns:
    A SequenceExample proto.
    """
    with tf.gfile.FastGFile(image.filename, "r") as f:
        encoded_image = f.read()

    try:
        decoder.decode_image(encoded_image)
    except (tf.errors.InvalidArgumentError, AssertionError):
        print("Skipping file with invalid JPEG data: %s" % image.filename)
        return

    context = tf.train.Features(feature={
        "image/image_id": _bytes_feature(image.image_id),
        "image/data": _bytes_feature(encoded_image),
    })

    formula = _process_formula(_get_formula(int(image.formula_line)))
    formula_ids = [vocab.word_to_id(word) for word in formula]
    feature_lists = tf.train.FeatureLists(feature_list={
        "image/formula": _bytes_feature_list(formula),
        "image/formula_ids": _int64_feature_list(formula_ids)
    })
    sequence_example = tf.train.SequenceExample(
    context=context, feature_lists=feature_lists)

    return sequence_example

def _process_image_files(thread_index, ranges, name, images, decoder, vocab, num_shards):
    """Processes and saves a subset of images as TFRecord files in one thread.
    Args:
        thread_index: Integer thread identifier within [0, len(ranges)].
        ranges: A list of pairs of integers specifying the ranges of the dataset to
            process in parallel.
        name: Unique identifier specifying the dataset.
        images: List of ImageMetadata.
        decoder: An ImageDecoder object.
        vocab: A Vocabulary object.
        num_shards: Integer number of shards for the output files.
    """
    # Each thread produces N shards where N = num_shards / num_threads. For
    # instance, if num_shards = 128, and num_threads = 2, then the first thread
    # would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
        num_shards_per_batch + 1).astype(int)
    num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in xrange(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = "%s-%.5d-of-%.5d.tfrecords" % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        images_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in images_in_shard:
            image = images[i]

            sequence_example = _to_sequence_example(image, decoder, vocab)
            if sequence_example is not None:
                writer.write(sequence_example.SerializeToString())
                shard_counter += 1
                counter += 1

            if not counter % 1000:
                print("%s [thread %d]: Processed %d of %d items in thread batch." %
                    (datetime.now(), thread_index, counter, num_images_in_thread))
                sys.stdout.flush()

        writer.close()
        print("%s [thread %d]: Wrote %d image-formula pairs to %s" %
            (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print("%s [thread %d]: Wrote %d image-formula pairs to %d shards." %
        (datetime.now(), thread_index, counter, num_shards_per_batch))
    sys.stdout.flush()

def _process_dataset(name, images, vocab, num_shards):
    """Processes a complete data set and saves it as a TFRecord.
    Args:
        name: Unique identifier specifying the dataset.
        images: List of ImageMetadata.
        vocab: A Vocabulary object.
        num_shards: Integer number of shards for the output files.
    """
    # Break up each image into a separate entity for each formula.
    images = [ImageMetadata(image.image_id, image.filename, image.formula_line)
        for image in images]

    # Shuffle the ordering of images. Make the randomization repeatable.
    random.seed(12345)
    random.shuffle(images)

    # Break the images into num_threads batches. Batch i is defined as
    # images[ranges[i][0]:ranges[i][1]].
    num_threads = min(num_shards, FLAGS.num_threads)
    spacing = np.linspace(0, len(images), num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a utility for decoding JPEG images to run sanity checks.
    decoder = ImageDecoder()

    # Launch a thread for each batch.
    print("Launching %d threads for spacings: %s" % (num_threads, ranges))
    for thread_index in xrange(len(ranges)):
        args = (thread_index, ranges, name, images, decoder, vocab, num_shards)
        t = threading.Thread(target=_process_image_files, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print("%s: Finished processing all %d image-formula pairs in data set '%s'." %
        (datetime.now(), len(images), name))

def _create_vocab(formulas):
    """Creates the vocabulary of word to word_id.

    The vocabulary is saved to disk in a text file of word counts. The id of each
    word in the file is its corresponding 0-based line number.

    Args:
        formulas: A list of lists of strings.
    Returns:
        A Vocabulary object.
    """
    print("Creating vocabulary.")
    counter = Counter()
    for c in formulas:
        counter.update(c)
    print("Total words:", len(counter))

    # Filter uncommon words and sort by descending count.
    word_counts = [x for x in counter.items() if x[1] >= FLAGS.min_word_count]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print("Words in vocabulary:", len(word_counts))

    # Write out the word counts file.
    with tf.gfile.FastGFile(FLAGS.word_counts_output_file, "w") as f:
        f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
    print("Wrote vocabulary file:", FLAGS.word_counts_output_file)

    # Create the vocabulary dictionary.
    reverse_vocab = [x[0] for x in word_counts]
    unk_id = len(reverse_vocab)
    vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
    vocab = Vocabulary(vocab_dict, unk_id)

    return vocab

# TODO: Replace with a better LaTeX tokenizer
def _process_formula(formula):
    """Processes a formula string into a list of tonenized words.
    Args:
        formula: A string formula.
    Returns:
        A list of strings; the tokenized formula.
    """
    tokenized_formula = [FLAGS.start_word]
    tokenized_formula.extend(nltk.tokenize.word_tokenize(formula.lower()))
    tokenized_formula.append(FLAGS.end_word)
    return tokenized_formula

def _load_and_process_metadata(formulas_file):
    """Loads image metadata from a JSON file and processes the formulas.
    Args:
        formulas_file: LST file containing formula annotations.
        image_dir: Directory containing the image files.
    Returns:
        A list of ImageMetadata.
    """
    with tf.gfile.FastGFile(formulas_file, "r") as f:
        formula_data = csv.reader(f, delimiter=' ')

    # Extract the filenames.
    id_to_filename = [(x[1], "%s.%s" % (x[1], FLAGS.image_format), x[0]) for x in formula_data]

    print("Loaded formula metadata for %d images from %s" %
        (len(id_to_filename), formulas_file))

    # Process the formulas and combine the data into a list of ImageMetadata.
    print("Proccessing formulas.")
    image_metadata = []
    num_formulas = 0
    for image_id, base_filename, formula_line in id_to_filename:
        filename = os.path.join("data_dir/formula_images", base_filename)
        image_metadata.append(ImageMetadata(image_id, filename, formula_line))
        num_formulas += 1
    print("Finished processing %d LaTeX formulas in %s" %
        (len(id_to_filename), formulas_file))

    return image_metadata

def _process_formulas(file):
    formulas = []
    with tf.gfile.FastGFile(file, "r") as f:
        for _, line in enumerate(f):
            formulas.append(_process_formula(line))
    return formulas

def main(unused_argv):
    def _is_valid_num_shards(num_shards):
        """Returns True if num_shards is compatible with FLAGS.num_threads."""
        return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

    assert _is_valid_num_shards(FLAGS.train_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.train_shards")
    assert _is_valid_num_shards(FLAGS.val_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.val_shards")
    assert _is_valid_num_shards(FLAGS.test_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.test_shards")

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)


    # Load image metadata from formula files.
    train_dataset = _load_and_process_metadata(FLAGS.train_formulas_file)
    val_dataset = _load_and_process_metadata(FLAGS.val_formulas_file)
    test_dataset = _load_and_process_metadata(FLAGS.test_formulas_file)

    formulas = _process_formulas(FLAGS.formulas_file)

    # Create vocabulary from the LaTeX formulas.
    vocab = _create_vocab(formulas)

    _process_dataset("train", train_dataset, vocab, FLAGS.train_shards)
    _process_dataset("val", val_dataset, vocab, FLAGS.val_shards)
    _process_dataset("test", test_dataset, vocab, FLAGS.test_shards)


if __name__ == "__main__":
    tf.app.run()
