from NeuralNet import CifarNeuralNet
import tensorflow as tf
from config import FLAGS


def main():
    if FLAGS.run_mode == 'train':
        model = CifarNeuralNet()

        with tf.Session() as sess:
            model.load_or_init(sess)
            model.train(sess, ['tfrecords/train.tfrecords'], 50000, ['tfrecords/train_eval_1000.tfrecords'], 1000,
                        ['tfrecords/test_eval_1000.tfrecords'], 1000)
    elif FLAGS.run_mode == 'overfit_100':
        FLAGS.train_batch_size = 100
        FLAGS.eval_batch_size = 100

        model = CifarNeuralNet()

        with tf.Session() as sess:
            model.load_or_init(sess)
            model.train(sess, ['tfrecords/train_overfit_100.tfrecords'], 100, ['tfrecords/train_overfit_100.tfrecords'], 100)
    elif FLAGS.run_mode == 'overfit_1000':
        FLAGS.train_batch_size = 100
        FLAGS.eval_batch_size = 500

        model = CifarNeuralNet()

        with tf.Session() as sess:
            model.load_or_init(sess)
            model.train(sess, ['tfrecords/train_overfit_1000.tfrecords'], 1000, ['tfrecords/train_overfit_1000.tfrecords'], 1000)
    elif FLAGS.run_mode == 'predict':
        model = CifarNeuralNet()

        with tf.Session() as sess:
            model.load_or_init(sess)

            train_accuracy, train_loss = model.eval(sess, 50000, FLAGS.eval_batch_size,
                                                    ['tfrecords/train.tfrecords'], disable_bar=False)
            print("Train Accuracy: {0:.3f}".format(train_accuracy))
            print("Train Loss: {0:.3f}".format(train_loss))

            test_accuracy, test_loss = model.eval(sess, 10000, FLAGS.eval_batch_size,
                                                  ['tfrecords/test.tfrecords'], disable_bar=False)
            print("Test Accuracy: {0:.3f}".format(test_accuracy))
            print("Test Loss: {0:.3f}".format(test_loss))
    else:
        raise ValueError('Unrecognized train_mode')


if __name__ == '__main__':
    main()
