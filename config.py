import os
import tensorflow as tf

# ============================================================================================================ #
# General Flags
# ============================================================================================================ #
tf.app.flags.DEFINE_string(
    'run_name', 'experiment_1',
    'Create a new folder for the experiment with a set name')
tf.app.flags.DEFINE_integer(
    'ckpt', 0,
    'Restore model from the checkpoint, 0 - restore from the latest one or from scratch if no ckpts.')
tf.app.flags.DEFINE_integer(
    'num_epochs', 1,
    'Number of training epochs.')
tf.app.flags.DEFINE_string(
    'trunk', 'net_2',
    'Name of the network\'s trunk, one of "net_1", "net_2", "net_3", "resnet20".')
tf.app.flags.DEFINE_integer(
    'train_batch_size', 128,
    'Mini-batch size')
tf.app.flags.DEFINE_float(
    'keep_prob', 1.0,
    'Probability for a neuron to be opened in a dropout layers.')
tf.app.flags.DEFINE_integer(
    'eval_train_batch_size', 500,
    'Mini-batch size. It has to divide the number of elements in a train dataset to read from queue without troubles')
tf.app.flags.DEFINE_integer(
    'eval_test_batch_size', 500,
    'Mini-batch size. It has to divide the number of elements in a test dataset to read from queue without troubles')
tf.app.flags.DEFINE_integer(
    'eval_train_size', 1000,
    'Size of the data using for evaluation model\'s performance on the train set.')
tf.app.flags.DEFINE_integer(
    'eval_test_size', 2000,
    'Size of the data using for evaluation model\'s performance on the test set.')
tf.app.flags.DEFINE_integer(
    'save_freq', 50,
    'Save model\'s parameters every n iterations.')
tf.app.flags.DEFINE_integer(
    'eval_freq', 0,
    'Eval model\'s performance every n iterations.')
tf.app.flags.DEFINE_integer(
    'random_seed_tf', 1,
    'Particular random initialization of model\'s parameters, 0 correspond to a random init without particular seed.')
tf.app.flags.DEFINE_integer(
    'random_seed_np', 1,
    'Particular random initialization of numpy operations, i.e batch iterations and consequently computed gradients'
    'and weights update, 0 - random without particular seed')
tf.app.flags.DEFINE_integer(
    'queue_random_seed', 1,
    'Queue randomness')
tf.app.flags.DEFINE_integer(
    'track_vars_grads_freq', 0,
    'Save statistics about variables and gradients each specified number of iterations (gradient updates)'
    '0 means ')
tf.app.flags.DEFINE_float(
    'decay_bn', 0.999,
    'decay parameter for batch normalization layers')
tf.app.flags.DEFINE_string(
    'shortcut_mode', 'conv1x1',
    'One of "conv1x1" or "padding"')
tf.app.flags.DEFINE_string(
    'mode', 'train',
    'One of "train" or "eval"')


# ============================================================================================================ #
# Optimization Flags
# ============================================================================================================ #
tf.app.flags.DEFINE_float(
    'weight_decay', 0.0,
    'The weight decay on the model weights.')
tf.app.flags.DEFINE_string(
    'optimizer', 'adam',
    'Name of the optimizer, one of "sgd", "momentum", "adam".')
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_bool(
    'use_nesterov', False,
    'Use Accelerated Nesterov momentum or a general momentum oprimizer')
tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float(
    'opt_epsilon', 1e-08,
    'Epsilon term for the optimizer.')


# ============================================================================================================ #
# Learning Rate Flags
# ============================================================================================================ #
tf.app.flags.DEFINE_string(
    'learning_rate_decay_type', 'fixed',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential", "polynomial".')
tf.app.flags.DEFINE_float(
    'learning_rate', 1e-3,
    'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 1e-6,
    'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.9,
    'Learning rate decay factor.')
tf.app.flags.DEFINE_integer(
    'num_epochs_per_decay', 2,
    'Number of epochs after which learning rate decreases.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

# ============================================================================================================ #
# CIFAR-10
# ============================================================================================================ #
tf.app.flags.DEFINE_integer(
    'train_size', 50000,
    'Size of the training dataset')
tf.app.flags.DEFINE_integer(
    'test_size', 10000,
    'Size of the test dataset')
tf.app.flags.DEFINE_integer(
    'num_classes', 10,
    'Size of the test dataset')


FLAGS = tf.app.flags.FLAGS

# I want to train with only full batches (othervise there is a problem with queue when I fetch the last batch)
FLAGS.num_batches_train = int(FLAGS.train_size / FLAGS.train_batch_size)

FLAGS.root_dir = os.getcwd()
FLAGS.tfrecords_dir = os.path.join(FLAGS.root_dir, 'tfrecords')
FLAGS.data_dir = os.path.join(FLAGS.root_dir, 'cifar-10-batches-py')
experiments_folder = os.path.join(FLAGS.root_dir, 'experiments')
if not os.path.exists(experiments_folder):
    os.makedirs(experiments_folder)
FLAGS.experiment_dir = os.path.join(experiments_folder, FLAGS.run_name)
if not os.path.exists(FLAGS.experiment_dir):
    os.makedirs(FLAGS.experiment_dir)
FLAGS.summary_dir = os.path.join(FLAGS.experiment_dir, 'summary')
FLAGS.ckpt_dir = os.path.join(FLAGS.experiment_dir, 'checkpoints/')
