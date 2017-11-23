import pickle
import numpy as np
import tensorflow as tf
from urllib.request import urlretrieve
from tqdm import tqdm
import tarfile
import os


LABELS_NAMES = {0: 'airplane',
                1: 'automobile',
                2: 'bird',
                3: 'cat',
                4: 'deer',
                5: 'dog',
                6: 'frog',
                7: 'horse',
                8: 'ship',
                9: 'truck'}


def get_data():
    root_dir = os.getcwd()
    data_dir = os.path.join(root_dir, 'cifar-10-batches-py')

    maybe_download_and_extract(root_dir, data_dir)

    train_data_files = [os.path.join(data_dir, 'data_batch_' + str(i)) for i in range(1, 6)]
    test_data_files = [os.path.join(data_dir, 'test_batch')]

    train_images, train_labels = get_images_labels(train_data_files)
    test_images, test_labels = get_images_labels(test_data_files)


    # mean = np.mean(train_images, axis=(0, 1, 2, 3))
    # std = np.std(train_images, axis=(0, 1, 2, 3))
    # mean= 0.47336489, std=0.25156906

    return train_images, train_labels, test_images, test_labels


# ============================================================================================================ #
# Downloading the data
# ============================================================================================================ #
class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def download_file(url, save_path):
    with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:  # all optional kwargs
        urlretrieve(url, filename=save_path, reporthook=t.update_to, data=None)


def unpack(fname):
    if fname.endswith('tar.gz'):
        tar = tarfile.open(fname, 'r:gz')
        tar.extractall()
        tar.close()
    elif fname.endswith('tar'):
        tar = tarfile.open(fname, 'r:')
        tar.extractall()
        tar.close()
    else:
        raise ValueError('File was not recognized')


def maybe_download_and_extract(root_dir, data_dir):
    if not os.path.exists(data_dir):
        url_cifar = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        archive_path = os.path.join(root_dir, 'cifar.tar.gz')
        download_file(url_cifar, archive_path)
        unpack(archive_path)
        os.remove(archive_path)

# ============================================================================================================ #
# Reading images from binary files
# ============================================================================================================ #
def unpickle(file):
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary


def get_images_labels(files):
    data_dictionaries = [unpickle(file) for file in files]
    images = np.vstack(data_dict[b'data'] for data_dict in data_dictionaries)
    images = images.reshape(images.shape[0], 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255
    labels = np.concatenate([data_dict[b'labels'] for data_dict in data_dictionaries])

    return images, labels


# ============================================================================================================ #
# Creating tfrecords
# ============================================================================================================ #
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def maybe_create_tfrecords_train_test():
    root_dir = os.getcwd()
    tfrecords_dir = os.path.join(root_dir, 'tfrecords')

    if not os.path.exists(tfrecords_dir):
        os.makedirs(tfrecords_dir)
        train_images, train_labels, test_images, test_labels = get_data()
        create_tfrecords(tfrecords_dir, train_images, train_labels, 'train')
        create_tfrecords(tfrecords_dir, test_images, test_labels, 'test')


# mode is on of 'train', 'test'
def create_tfrecords(data_path, images, labels, mode):
    num_images = images.shape[0]
    train_filename = mode + '.tfrecords'

    writer = tf.python_io.TFRecordWriter(os.path.join(data_path, train_filename))

    for i in tqdm(range(num_images)):
        image_raw = images[i].tostring()
        label = labels[i]
        height = 32
        width = 32
        depth = 3

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'depth': _int64_feature(depth),
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(image_raw)}))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    maybe_create_tfrecords_train_test()