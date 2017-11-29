from NeuralNet import CifarNeuralNet
import tensorflow as tf
from config import FLAGS


def main():
    if FLAGS.train_mode == 'train':
        model = CifarNeuralNet()

        with tf.Session() as sess:
            model.load_or_init(sess)
            model.train(sess, ['tfrecords/train.tfrecords'], 50000, ['tfrecords/train_eval_1000.tfrecords'], 1000,
                        ['tfrecords/test_eval_1000.tfrecords'], 1000)
    elif FLAGS.train_mode == 'overfit_100':
        FLAGS.train_batch_size = 100
        FLAGS.eval_batch_size = 100

        model = CifarNeuralNet()

        with tf.Session() as sess:
            model.load_or_init(sess)
            model.train(sess, ['tfrecords/train_overfit_100.tfrecords'], 100, ['tfrecords/train_overfit_100.tfrecords'], 100)
    elif FLAGS.train_mode == 'overfit_1000':
        FLAGS.train_batch_size = 100
        FLAGS.eval_batch_size = 500

        model = CifarNeuralNet()

        with tf.Session() as sess:
            model.load_or_init(sess)
            model.train(sess, ['tfrecords/train_overfit_1000.tfrecords'], 1000, ['tfrecords/train_overfit_1000.tfrecords'], 1000)
    else:
        raise ValueError('Unrecognized train_mode')


if __name__ == '__main__':
    main()
