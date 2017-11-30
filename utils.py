import tensorflow as tf
slim = tf.contrib.slim


def add_weights(shape, name, trainable=True):
    with tf.variable_scope(name):
        W = tf.get_variable(name='weights',
                            shape=shape,
                            initializer=tf.truncated_normal_initializer(stddev=0.1),
                            trainable=trainable)
    return W


def add_biases(shape, name, trainable=True):
    with tf.variable_scope(name):
        b = tf.get_variable(name='biases',
                            shape=shape,
                            initializer=tf.constant_initializer(0.1),
                            trainable=trainable)
    return b


def conv(input, filter_h, filter_w, num_filters, stride_y, stride_x, name, padding='VALID', trainable=True):
    with tf.name_scope(name):
        input_channels = int(input.get_shape()[3])
        W = add_weights([filter_h, filter_w, input_channels, num_filters], name, trainable)
        b = add_biases([num_filters], name, trainable)
        activations = tf.nn.relu(tf.nn.conv2d(input, W, strides=[1,stride_y,stride_x,1], padding=padding) + b,
                                 name='activations')
    return activations


def max_pool(input, filter_h, filter_w, stride_y, stride_x, name, padding='VALID'):
    with tf.name_scope(name):
        activations = tf.nn.max_pool(input, ksize=[1, filter_h, filter_w, 1], strides=[1, stride_y, stride_x, 1],
                                     padding=padding, name='activations')
    return activations


def fc(input, out_neurons, name, relu=True, trainable=True):
    with tf.name_scope(name):
        in_neurons = int(input.get_shape()[1])
        W = add_weights([in_neurons, out_neurons], name, trainable)
        b = add_biases([out_neurons], name, trainable)

        if relu:
            activations = tf.nn.relu(tf.matmul(input, W) + b, name='activations')
        else:
            activations = tf.add(tf.matmul(input, W), b, name='activations')
    return activations


def dropout(input, keep_prob, name):
    with tf.name_scope(name):
        drop = tf.nn.dropout(input, keep_prob, name='dropout')
    return drop


def flatten_3d(input, name):
    with tf.name_scope(name):
        heigth = int(input.get_shape()[1])
        width = int(input.get_shape()[2])
        deepth = int(input.get_shape()[3])
        output = tf.reshape(input, [-1, heigth*width*deepth], name='flattening')
    return output


def prob_close(is_train, prob):
    with tf.name_scope('dropout_controller'):
        return 1-tf.cast(is_train, tf.float32)*prob


def conv_bn_relu(x, filters, kernel, stride, scope):
    with tf.name_scope(scope):
        net = slim.conv2d(x, filters, [kernel, kernel], stride, scope='conv')
        net = slim.batch_norm(net, scope='bn')
        net = tf.nn.relu(net, name='relu')
    return net


def residual_unit(x, filters, scope, change_dim):
    with tf.name_scope(scope):
        if change_dim:
            shortcut = slim.conv2d(x, filters, [1, 1], stride=2, scope='match_shortcut')
            stride = 2
        else:
            shortcut = tf.identity(x, name='iden_shortcut')
            stride = 1
        net = slim.conv2d(x, filters, [3, 3], stride, scope='conv1')
        net = slim.batch_norm(net, scope='bn1')
        net = tf.nn.relu(net, name='relu1')
        net = slim.conv2d(net, filters, [3, 3], 1, scope='conv2')
        net = slim.batch_norm(net, scope='bn2')
        net = tf.add(net, shortcut, name='merge')
        net = tf.nn.relu(net, name='relu2')
    return net









