from NeuralNet import CifarNeuralNet
import tensorflow as tf
from config import FLAGS


def main():
    FLAGS.mode = 'train'
    model = CifarNeuralNet()

    with tf.Session() as sess:
        model.load_or_init(sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        model.train(sess)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
