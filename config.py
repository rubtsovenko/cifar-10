import os
import tensorflow as tf

# ============================================================================================================ #
# General Flags
# ============================================================================================================ #
tf.app.flags.DEFINE_string(
    'run_mode', 'train',
    'One of train, overfit_100, overfit_1000, predict')
tf.app.flags.DEFINE_string(
    'run_name', 'run_1',
    'Create a new folder for the experiment with a set name')
tf.app.flags.DEFINE_integer(
    'ckpt', 0,
    'Restore model from the checkpoint, 0 - restore from the latest one or from scratch if no ckpts.')
tf.app.flags.DEFINE_integer(
    'num_epochs', 200,
    'Number of training epochs.')
tf.app.flags.DEFINE_string(
    'trunk', 'resnet20',
    'Name of the network\'s trunk, one of "net_1", "net_2", "net_3", "net_4", "resnet20".')
tf.app.flags.DEFINE_float(
    'decay_bn', 0.9,
    'decay parameter for batch normalization layers')
tf.app.flags.DEFINE_integer(
    'num_threads', 8,
    'Number of threads to read and preprocess.')

# batch settings
tf.app.flags.DEFINE_integer(
    'train_batch_size', 256,
    'Mini-batch size')
tf.app.flags.DEFINE_integer(
    'eval_batch_size', 500,
    'Mini-batch size for evaluation. It shout take optimal value for prediction speed')

# randomness
tf.app.flags.DEFINE_integer(
    'random_seed_tf', 1,
    'Particular random initialization of model\'s parameters, 0 correspond to a random init without particular seed.')

# save info
tf.app.flags.DEFINE_integer(
    'save_freq', 1,
    'Save model\'s parameters every n iterations.')


# ============================================================================================================ #
# Optimization Flags
# ============================================================================================================ #
tf.app.flags.DEFINE_string(
    'optimizer', 'momentum',
    'Name of the optimizer, one of "sgd", "momentum", "adam", "rmsprop".')
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
    'rmsprop_momentum', 0.9,
    'Momentum for RMSProp')
tf.app.flags.DEFINE_float(
    'rmsprop_decay', 0.9,
    'Decay term for RMSProp.')

# learning rate
tf.app.flags.DEFINE_float(
    'learning_rate', 0.1,
    'Initial learning rate.')

# weight decay regularization
tf.app.flags.DEFINE_float(
    'weight_decay', 0.0,
    'The weight decay on the model weights.')


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
tf.app.flags.DEFINE_float(
    'MEAN', 0.47336489,
    'MEAN pixel computed in CifarLoader')
tf.app.flags.DEFINE_float(
    'STD', 0.25156906,
    'STD pixels computed in CifarLoader')


FLAGS = tf.app.flags.FLAGS


FLAGS.root_dir = os.getcwd()
FLAGS.tfrecords_dir = os.path.join(FLAGS.root_dir, 'tfrecords')
FLAGS.data_dir = os.path.join(FLAGS.root_dir, 'cifar-10-batches-py')
experiments_folder = os.path.join(FLAGS.root_dir, 'experiments')
if not os.path.exists(experiments_folder):
    os.makedirs(experiments_folder)
FLAGS.experiment_dir = os.path.join(experiments_folder, FLAGS.run_name)
FLAGS.log_dir = os.path.join(FLAGS.experiment_dir, 'logs')
if not os.path.exists(FLAGS.experiment_dir):
    os.makedirs(FLAGS.experiment_dir)
    os.makedirs(FLAGS.log_dir)
FLAGS.summary_dir = os.path.join(FLAGS.experiment_dir, 'summary')
FLAGS.ckpt_dir = os.path.join(FLAGS.experiment_dir, 'checkpoints/')
