from NeuralNet import CifarNeuralNet
import tensorflow as tf
from config import FLAGS


def main():
    FLAGS.mode = 'eval_train'
    model = CifarNeuralNet()

    with tf.Session() as sess:
        model.load_or_init(sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        accuracy = model.eval(sess)
        print('Train accuracy:', accuracy)

        coord.request_stop()
        coord.join(threads)
    print('\n')

    tf.reset_default_graph()

    FLAGS.mode = 'eval_test'
    model = CifarNeuralNet()

    with tf.Session() as sess:
        model.load_or_init(sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        accuracy = model.eval(sess)
        print('Test accuracy:', accuracy)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
