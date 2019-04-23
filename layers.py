import tensorflow as tf
import numpy as np

import hyperparam
from hyperparam import maxdisp, ps___


def convbn(input, in_channels, out_channels, strides, padding, name):
    with tf.variable_scope(name):
        weight = tf.get_variable('weightxxx', [3, 3, in_channels, out_channels],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(input, weight, strides=strides, padding=padding)
        conv = tf.layers.batch_normalization(conv, training=hyperparam.is_training)
    return conv


def convbn_3d(input, in_channels, out_channels, stride, pad, name):
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [3, 3, 3, in_channels, out_channels],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv3d(input, weight, strides=[1, stride, stride, stride, 1], padding=pad)
        conv = tf.layers.batch_normalization(conv, training=hyperparam.is_training)
    return conv


def conv3d_trans(input, in_channels, out_channels, stride, pad, name):
    # with tf.variable_scope(name):
    #     weight = tf.get_variable('weight', [3, 3, 3, out_channels, in_channels],
    #                                     initializer=tf.truncated_normal_initializer(stddev=0.1))

    conv = tf.layers.conv3d_transpose(input, out_channels, 3, strides=(stride, stride, stride), padding=pad)

    conv = tf.layers.batch_normalization(conv, training=hyperparam.is_training)
    return conv

def hourglass(in_channels, x, presqu, postsqu, name):
    out = convbn_3d(x, in_channels, in_channels*2, 2, 'SAME', name+'_1')
    out = tf.nn.relu(out)
    pre = convbn_3d(out, in_channels*2, in_channels*2, 1, 'SAME', name+'_2')
    if postsqu is not None:
        pre = tf.nn.relu(pre + postsqu)
    else:
        pre = tf.nn.relu(pre)
    out = convbn_3d(pre, in_channels*2, in_channels*2, 2, 'SAME', name+'_3')
    out = tf.nn.relu(out)
    out = convbn_3d(out, in_channels*2, in_channels*2, 1, 'SAME', name+'_4')
    if presqu is not None:
        post = conv3d_trans(out, in_channels*2, in_channels*2, 2, 'SAME', name+'_5') + presqu
        post = tf.nn.relu(post)
    else:
        post = conv3d_trans(out, in_channels * 2, in_channels * 2, 2, 'SAME', name + '_5') + pre
        post = tf.nn.relu(post)
    out = conv3d_trans(post, in_channels * 2, in_channels, 2, 'SAME', name + '_6')

    return out, pre, post

def disparity_regression(input, name):
    # with tf.variable_scope(name):
    #     disp = tf.convert_to_tensor(np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1]), dtype=tf.float32)
    #     disp = tf.tile(disp, [input.shape[0], 1, input.shape[2], input.shape[3]])
    #     out = tf.reduce_sum(input*disp, 1)
    # return out
    with tf.variable_scope(name):
        probability_volume = tf.nn.softmax(tf.scalar_mul(-1, input),
                                           dim=1, name='prob_volume')
        volume_shape = tf.shape(probability_volume)
        soft_1d = tf.cast(tf.range(0, volume_shape[1], dtype=tf.int32),tf.float32)
        soft_4d = tf.tile(soft_1d, tf.stack([volume_shape[0] * volume_shape[2] * volume_shape[3]]))
        soft_4d = tf.reshape(soft_4d, [volume_shape[0], volume_shape[2], volume_shape[3], volume_shape[1]])
        soft_4d = tf.transpose(soft_4d, [0, 3, 1, 2])
        estimated_disp_image = tf.reduce_sum(soft_4d * probability_volume, axis=1)
        return estimated_disp_image
