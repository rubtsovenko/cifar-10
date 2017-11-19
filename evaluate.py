from NeuralNet import CifarNeuralNet
import tensorflow as tf


def main():
    model = CifarNeuralNet()

    with tf.Session() as sess:
        model.load_or_init(sess)
        train_accuracy, test_accuracy = model.eval(sess)


if __name__ == '__main__':
    main()
