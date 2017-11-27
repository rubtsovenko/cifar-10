from NeuralNet import CifarNeuralNet
import tensorflow as tf


def main():
    model = CifarNeuralNet()

    with tf.Session() as sess:
        model.load_or_init(sess)
        train_accuracy, test_accuracy = model.eval(sess)
        print("Train accuracy:", train_accuracy)
        print("Test accuracy:", test_accuracy)


if __name__ == '__main__':
    main()
