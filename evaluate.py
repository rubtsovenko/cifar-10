from NeuralNet import CifarNeuralNet
import tensorflow as tf
from config import FLAGS


def main():
    model = CifarNeuralNet()

    with tf.Session() as sess:
        model.load_or_init(sess)

        train_accuracy, train_loss = model.eval(sess, FLAGS.train_size, FLAGS.eval_train_batch_size,
                                               ['tfrecords/train.tfrecords'], disable_bar=False)
        print("Train Accuracy: {0:.3f}".format(train_accuracy))
        print("Train Loss: {0:.3f}".format(train_loss))

        test_accuracy, test_loss = model.eval(sess, FLAGS.test_size, FLAGS.eval_test_batch_size,
                                             ['tfrecords/test.tfrecords'], disable_bar=False)
        print("Test Accuracy: {0:.3f}".format(test_accuracy))
        print("Test Loss: {0:.3f}".format(test_loss))


if __name__ == '__main__':
    main()
