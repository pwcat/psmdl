import tensorflow as tf
import hyperparam
from layers import convbn


def resblock(inputs, in_channels, out_channels, stride, dilation, name, downsample=False):
    with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
        if dilation == 1:
            conv1_weights = tf.get_variable('weight1', [3, 3, in_channels, out_channels],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv1 = tf.nn.conv2d(inputs, conv1_weights, strides=[1, stride, stride, 1], padding='SAME')
            conv1 = tf.layers.batch_normalization(conv1, training=hyperparam.is_training)
            conv1 = tf.nn.relu(conv1)
            conv2_weights = tf.get_variable('weight2', [3, 3, out_channels, out_channels],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2 = tf.nn.conv2d(conv1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
            conv2 = tf.layers.batch_normalization(conv2, training=hyperparam.is_training)
        else:
            conv1_weights = tf.get_variable('weight1', [3, 3, in_channels, out_channels],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv1 = tf.nn.atrous_conv2d(inputs, conv1_weights, dilation, padding='SAME')
            conv1 = tf.nn.relu(conv1)
            conv1 = tf.nn.relu(conv1)
            conv2_weights = tf.get_variable('weight2', [3, 3, out_channels, out_channels],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2 = tf.nn.atrous_conv2d(conv1, conv2_weights, dilation, padding='SAME')
            conv2 = tf.layers.batch_normalization(conv2, training=hyperparam.is_training)

        # print(name + ' before ' + str(inputs.shape))

        if downsample and (stride != 1 or 32 != out_channels):
            downsample_weights = tf.get_variable('weight3', [1, 1, in_channels, out_channels],
                                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
            inputs = tf.nn.conv2d(inputs, downsample_weights, [1, stride, stride, 1], padding='SAME')
            inputs = tf.layers.batch_normalization(inputs, training=hyperparam.is_training)
        # print(name + ' after ' + str(inputs.shape))
        # print(name + ' conv2 ' + str(conv2.shape))
        res = conv2 + inputs
        return res
