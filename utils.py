import tensorflow as tf
slim = tf.contrib.slim


def conv_bn_relu(x, filters, kernel, stride, scope):
    with tf.variable_scope(scope):
        net = slim.conv2d(x, filters, [kernel, kernel], stride, scope='conv')
        net = slim.batch_norm(net, scope='bn')
        net = tf.nn.relu(net, name='relu')
    return net






