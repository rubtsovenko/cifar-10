import tensorflow as tf
from tqdm import tqdm
from config import FLAGS
from architectures import net_1, net_2, net_3, resnet20
import numpy as np
import os
import tensorlayer


class CifarNeuralNet(object):
    def __init__(self):
        set_random_seed()
        self.X, self.y_, self.filenames, self.augment, self.batch_size, self.num_epochs, self.iterator = network_input()
        self.is_train = tf.placeholder(tf.bool)

        self.y_logits_op = build_trunk(self.X, self.is_train)
        self.loss_op = add_loss(self.y_, self.y_logits_op)
        with tf.name_scope('softmax'):
            self.y_preds_op = tf.nn.softmax(self.y_logits_op)
        self.correct_preds_op = tf.equal(tf.argmax(self.y_preds_op, 1), tf.argmax(self.y_, 1))
        with tf.name_scope('accuracy'):
            self.accuracy_op = tf.reduce_mean(tf.cast(self.correct_preds_op, tf.float32))

        self.optimizer_op = add_optimizer()
        self.train_op, self.global_step = add_train_op(self.loss_op, self.optimizer_op)

        with tf.name_scope('init'):
            self.init_op = tf.group(tf.global_variables_initializer())

        self.summary_metrics, self.summary_variables, self.summary_grads = self._create_summaries()
        self.saver = tf.train.Saver(max_to_keep=100)

    def load_or_init(self, sess):
        if FLAGS.ckpt == 0:
            saved_ckpt = tf.train.latest_checkpoint(FLAGS.ckpt_dir)
            if saved_ckpt is None:
                print('There is not any saved model.\nUsed random initialization')
                sess.run(self.init_op)
            else:
                FLAGS.ckpt = int(saved_ckpt.split('/')[-1][1:])
                print('Load model from ckpt {}'.format(FLAGS.ckpt))
                self.saver.restore(sess, saved_ckpt)
        else:
            chosen_ckpt = os.path.join(FLAGS.ckpt_dir, '-'+str(FLAGS.ckpt))
            if os.path.exists(chosen_ckpt+'.index'):
                print('Load model from ckpt {}'.format(FLAGS.ckpt))
                self.saver.restore(sess, chosen_ckpt)
            else:
                raise ValueError('No ckpt {} exists in {}'.format(FLAGS.ckpt, FLAGS.ckpt_dir))

    def train(self, sess):
        for epoch in range(FLAGS.ckpt + 1, FLAGS.ckpt + 1 + FLAGS.num_epochs):
            sess.run(self.iterator.initializer, {self.filename: ['tfrecords/train.tfrecords'],
                                                 self.batch_size: FLAGS.train_batch_size,
                                                 self.num_epochs: 1,
                                                 self.augment: True})
            for _ in tqdm(range(FLAGS.num_batches_train), desc='Epoch {:3d}'.format(epoch)):
                sess.run(self.train_op, {self.is_train: True})

            self.track_performance(sess)
            if epoch % FLAGS.save_freq == 0:
                self.saver.save(sess, FLAGS.ckpt_dir, global_step=epoch)

    def track_performance(self, sess):
        sess.run(self.iterator.initializer, {self.filename: ['tfrecords/train.tfrecords'],
                                             self.batch_size: FLAGS.eval_train_batch_size,
                                             self.num_epochs: 1,
                                             self.augment: False})
        total = 0
        for _ in range(FLAGS.num_batches_eval_train):
            total += sum(sess.run(self.correct_preds_op, {self.is_train: False}))
        print("Train accuracy: ", total / FLAGS.eval_train_size)

        sess.run(self.iterator.initializer, {self.filename: ['tfrecords/test.tfrecords'],
                                             self.batch_size: FLAGS.eval_test_batch_size,
                                             self.num_epochs: 1,
                                             self.augment: False})
        total = 0
        for _ in range(FLAGS.num_batches_eval_test):
            total += sum(sess.run(self.correct_preds_op, {self.is_train: False}))
        print("Test accuracy: ", total / FLAGS.eval_test_size)

    def eval(self, sess):
        sess.run(self.iterator.initializer, {self.filename: ['tfrecords/train.tfrecords'],
                                             self.batch_size: FLAGS.eval_train_batch_size,
                                             self.num_epochs: 1,
                                             self.augment: False})
        num_batches = FLAGS.train_size // FLAGS.eval_train_batch_size
        correct_preds = 0
        for _ in tqdm(range(num_batches), desc='Train eval'):
            correct_preds += sum(sess.run(self.correct_preds_op, {self.is_train: False}))
        train_accuracy = correct_preds / FLAGS.train_size

        sess.run(self.iterator.initializer, {self.filename: ['tfrecords/test.tfrecords'],
                                             self.batch_size: FLAGS.eval_test_batch_size,
                                             self.num_epochs: 1,
                                             self.augment: False})
        num_batches = FLAGS.test_size // FLAGS.eval_test_batch_size
        correct_preds = 0
        for _ in tqdm(range(num_batches), desc='Test eval'):
            correct_preds += sum(sess.run(self.correct_preds_op, {self.is_train: False}))
        test_accuracy = correct_preds / FLAGS.test_size

        return train_accuracy, test_accuracy


def set_random_seed():
    if FLAGS.random_seed_tf != 0:
        tf.set_random_seed(FLAGS.random_seed_tf)
    if FLAGS.random_seed_np != 0:
        np.random.seed(FLAGS.random_seed_np)


def parce_tfrecord(serialized_example):
    features = {'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'depth': tf.FixedLenFeature([], tf.int64),
                'label': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string)}
    parsed_record = tf.parse_single_example(serialized_example, features)

    # Reshape image data into the original shape
    height = tf.cast(parsed_record['height'], tf.int32)
    width = tf.cast(parsed_record['width'], tf.int32)
    depth = tf.cast(parsed_record['depth'], tf.int32)

    image = tf.decode_raw(parsed_record['image_raw'], tf.float32)
    image = tf.reshape(image, shape=[height, width, depth])

    # Preprocessing
    label = tf.cast(parsed_record['label'], tf.int32)
    label = tf.one_hot(label, FLAGS.num_classes)

    return image, label


def train_transform(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.contrib.image.rotate(image, tf.random_uniform(shape=(), minval=-15/180*3.14, maxval=15/180*3.14))
    image = tf.py_func(tensorlayer.prepro.shift, inp=[image, 0.1, 0.1, True], Tout='float32')
    image.set_shape([32,32,3])

    return image


def test_transform(image):
    image.set_shape([32,32,3])

    return image


def data_augmentation(image, label, augment):
    image = (image - FLAGS.MEAN) / (FLAGS.STD+1e-7)
    transformation = tf.cond(augment, lambda: train_transform(image), lambda: test_transform(image))
    image = transformation
    return image, label


def network_input():
    with tf.name_scope('input'):
        filenames = tf.placeholder(tf.string, shape=[None])
        augment = tf.placeholder(tf.bool)
        batch_size = tf.placeholder(tf.int64)
        num_epochs = tf.placeholder(tf.int64)

        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(parce_tfrecord, num_parallel_calls=FLAGS.num_threads)
        dataset = dataset.map(lambda image, label: data_augmentation(image, label, augment),
                              num_parallel_calls=FLAGS.num_threads)
        dataset = dataset.shuffle(10000)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(100)

        iterator = dataset.make_initializable_iterator()
        images, labels = iterator.get_next()

    return images, labels, filenames, augment, batch_size, num_epochs, iterator


def build_trunk(X, is_train):
    if FLAGS.trunk == 'net_1':
        y_logits = net_1(X, is_train)
    elif FLAGS.trunk == 'net_2':
        y_logits = net_2(X, is_train)
    elif FLAGS.trunk == 'net_3':
        y_logits = net_3(X, is_train)
    elif FLAGS.trunk == 'resnet20':
        y_logits = resnet20(X, is_train)
    else:
        raise ValueError('Network architecture {} was not recognized'.format(FLAGS.trunk))

    return y_logits


def add_loss(y_, y_logits):
    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y_))
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = cross_entropy + sum(reg_losses)

    return loss


def add_optimizer():
    with tf.name_scope('optimizer'):
        if FLAGS.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        elif FLAGS.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.adam_beta1,
                                               beta2=FLAGS.adam_beta2)
        elif FLAGS.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, momentum=FLAGS.momentum,
                                                   use_nesterov=FLAGS.use_nesterov)
        elif FLAGS.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate, decay=FLAGS.rmsprop_decay,
                                                  momentum=FLAGS.rmsprop_momentum)
        else:
            raise ValueError('Optimizer {} was not recognized'.format(FLAGS.optimizer))

        return optimizer


def add_train_op(loss, optimizer):
    with tf.name_scope('train_step'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # next line is necessary for batch normalization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op, global_step