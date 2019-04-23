import tensorflow as tf
from tensorflow._api.v1 import keras

import hyperparam
from resblock import resblock
from hyperparam import maxdisp
from layers import *

class PSMNet:
    def __init__(self, sess, height=256, weight=512, batch_size=4):
        self.sess = sess
        self.height = height
        self.weight = weight
        self.batch_size = batch_size

        self.left = tf.placeholder(tf.float32, shape=[self.batch_size, self.height, self.weight, 3])
        self.right = tf.placeholder(tf.float32, shape=[self.batch_size, self.height, self.weight, 3])
        self.label = tf.placeholder(tf.float32, shape=[self.batch_size, self.height, self.weight])
        self.image_size_tf = tf.shape(self.left)[1:3]

        self.disps = self.inference(self.left, self.right)
        # only compute valid labeled points
        disps_mask = tf.where(self.label > 0., self.disps, self.label)
        self.loss = self._smooth_l1_loss(disps_mask, self.label)

        optimizer = tf.train.AdamOptimizer(learning_rate=hyperparam.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)
        try:
          self.sess.run(tf.global_variables_initializer())
        except:
          self.sess.run(tf.initialize_all_variables())

    def train(self, left, right, label):
        _, loss = self.sess.run([self.train_op, self.loss],
                                feed_dict={self.left: left, self.right: right, self.label: label
                                           })
        return loss


    def predict(self, left, right):
        pred = self.sess.run([self.disps],
                             feed_dict={self.left: left, self.right: right})
        return pred

    def _smooth_l1_loss(self, disps_pred, disps_targets, sigma=1.0):
        sigma_2 = sigma ** 2
        box_diff = disps_pred - disps_targets
        in_box_diff = box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                      + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(out_loss_box))
        return loss_box

    def feature_extraction(self, input_tensor):
        with tf.variable_scope('cnn_conv0_1',reuse=tf.AUTO_REUSE):
            conv0_1_weights = tf.get_variable('cweight01', [3, 3, 3, 32],
                                              initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv0_1 = tf.nn.conv2d(input_tensor, conv0_1_weights, strides=[1, 2, 2, 1], padding='SAME')
            conv0_1 = tf.layers.batch_normalization(conv0_1, training=hyperparam.is_training)
            relu0_1 = tf.nn.relu(conv0_1)

        with tf.variable_scope('cnn_conv0_2',reuse=tf.AUTO_REUSE):
            conv0_2_weights = tf.get_variable('cweight02', [3, 3, 32, 32],
                                              initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv0_2 = tf.nn.conv2d(relu0_1, conv0_2_weights, strides=[1, 1, 1, 1], padding='SAME')
            conv0_2 =  tf.layers.batch_normalization(conv0_2, training=hyperparam.is_training)
            relu0_2 = tf.nn.relu(conv0_2)

        with tf.variable_scope('cnn_conv0_3',reuse=tf.AUTO_REUSE):
            conv0_3_weights = tf.get_variable('cweight03', [3, 3, 32, 32],
                                              initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv0_3 = tf.nn.conv2d(relu0_2, conv0_3_weights, strides=[1, 1, 1, 1], padding='SAME')
            conv0_3 =  tf.layers.batch_normalization(conv0_3, training=hyperparam.is_training)
            relu0_3 = tf.nn.relu(conv0_3)

        res_conv1_x = resblock(relu0_3, 32, 32, 1, 1, 'res_con1_1', True)
        for i in range(1, 3):
            res_conv1_x = resblock(res_conv1_x, 32, 32, 1, 1, 'res_con1_%d' % (i + 1))

        res_conv2_x = resblock(res_conv1_x, 32, 64, 2, 1, 'res_con2_1', True)
        for i in range(1, 16):
            res_conv2_x = resblock(res_conv2_x, 64, 64, 1, 1, 'res_con2_%d' % (i + 1))

        res_conv3_x = resblock(res_conv2_x, 64, 128, 1, 1, 'res_con3_1', True)
        for i in range(1, 3):
            res_conv3_x = resblock(res_conv3_x, 128, 128, 1, 1, 'res_con3_%d' % (i + 1))

        res_conv4_x = resblock(res_conv3_x, 128, 128, 1, 2, 'res_con4_1', True)
        for i in range(1, 3):
            res_conv4_x = resblock(res_conv4_x, 128, 128, 1, 2, 'res_con4_%d' % (i + 1))

        with tf.variable_scope('spp_b1',reuse=tf.AUTO_REUSE):
            #31 15 7 3
            branch1_weights = tf.get_variable('weightb1', [1, 1, 128, 32],
                                              initializer=tf.truncated_normal_initializer(stddev=0.1))
            branch1 = tf.nn.atrous_conv2d(res_conv4_x, branch1_weights, 31, padding='SAME')
            branch1 = tf.layers.batch_normalization(branch1, training=hyperparam.is_training)
            branch1 = tf.nn.relu(branch1)

        with tf.variable_scope('spp_b2',reuse=tf.AUTO_REUSE):
            branch2_weights = tf.get_variable('weightb2', [1, 1, 128, 32],
                                              initializer=tf.truncated_normal_initializer(stddev=0.1))
            branch2 = tf.nn.atrous_conv2d(res_conv4_x, branch2_weights, 15, padding='SAME')
            branch2 =  tf.layers.batch_normalization(branch2, training=hyperparam.is_training)
            branch2 = tf.nn.relu(branch2)

        with tf.variable_scope('spp_b3',reuse=tf.AUTO_REUSE):
            branch3_weights = tf.get_variable('weightb3', [1, 1, 128, 32],
                                              initializer=tf.truncated_normal_initializer(stddev=0.1))
            branch3 = tf.nn.atrous_conv2d(res_conv4_x, branch3_weights, 7, padding='SAME')
            branch3 =  tf.layers.batch_normalization(branch1, training=hyperparam.is_training)
            branch3 = tf.nn.relu(branch3)

        with tf.variable_scope('spp_b4',reuse=tf.AUTO_REUSE):
            branch4_weights = tf.get_variable('weightb4', [1, 1, 128, 32],
                                              initializer=tf.truncated_normal_initializer(stddev=0.1))
            branch4 = tf.nn.atrous_conv2d(res_conv4_x, branch4_weights, 3, padding='SAME')
            branch4 =  tf.layers.batch_normalization(branch4, training=hyperparam.is_training)
            branch4 = tf.nn.relu(branch4)

        spp_output = tf.concat([res_conv2_x, res_conv4_x, branch4, branch3, branch2, branch1], 3)
        with tf.variable_scope('spp_last',reuse=tf.AUTO_REUSE):
            spp_last_weight1 = tf.get_variable('lweight1', [3, 3, 320, 128],
                                               initializer=tf.truncated_normal_initializer(stddev=0.1))
            spp_output = tf.nn.conv2d(spp_output, spp_last_weight1, strides=[1, 1, 1, 1], padding='SAME')
            spp_output =  tf.layers.batch_normalization(spp_output, training=hyperparam.is_training)
            spp_output = tf.nn.relu(spp_output)
            spp_last_weight2 = tf.get_variable('lweight2', [1, 1, 128, 32],
                                               initializer=tf.truncated_normal_initializer(stddev=0.1))
            spp_output = tf.nn.conv2d(spp_output, spp_last_weight2, strides=[1, 1, 1, 1], padding='VALID')

        return spp_output

    def inference(self, left, right):
        refimg_feature = self.feature_extraction(left)
        targetimg_feature = self.feature_extraction(right)

        # with tf.variable_scope('inf'):
        #     cost = tf.get_variable('cost', [refimg_feature.shape[0], maxdisp / 4, refimg_feature.shape[1],
        #                                     refimg_feature.shape[2],
        #                                     refimg_feature.shape[3] * 2], initializer=tf.zeros_initializer())
        # for i in range(int(maxdisp) // 4):
        #     if i > 0:
        #         cost[:, :refimg_feature.shape[1], i, :, i:].assign(refimg_feature[:, :, :, i:])
        #         cost[:, refimg_feature.shape[1]:, i, :, i:].assign(targetimg_feature[:, :, :, :-i])
        #     else:
        #         cost[:, :refimg_feature.shape[1], i, :, :].assign(refimg_feature)
        #         cost[:, refimg_feature.shape[1]:, i, :, :].assign(targetimg_feature)
        with tf.variable_scope('cost_vol'):
            shape = tf.shape(targetimg_feature)
            right_tensor = keras.backend.spatial_2d_padding(targetimg_feature, padding=((0, 0), (maxdisp // 4, 0)))
            disparity_costs = []
            for d in reversed(range(maxdisp // 4)):
                left_tensor_slice = refimg_feature
                right_tensor_slice = tf.slice(right_tensor, begin=[0, 0, d, 0], size=shape)
                right_tensor_slice.set_shape(tf.TensorShape([None, None, None, 32]))
                cost = tf.concat([left_tensor_slice, right_tensor_slice], axis=3)
                disparity_costs.append(cost)
            cost_vol = tf.stack(disparity_costs, axis=1)

        cost0 = convbn_3d(cost_vol, 64, 32, 1, 'SAME', name='dres0_0')
        cost0 = tf.nn.relu(cost0)
        cost0 = convbn_3d(cost0, 32, 32, 1, 'SAME', name='dres0_1')
        cost0 = tf.nn.relu(cost0)

        temp0 = convbn_3d(cost0, 32, 32, 1, 'SAME', name='dres1_0')
        temp0 = tf.nn.relu(temp0)
        temp0 = convbn_3d(temp0, 32, 32, 1, 'SAME', name='dres1_1')
        cost0 = temp0 + cost0

        out1, pre1, post1 = hourglass(32, cost0, None, None, 'stack1')
        out1 = out1 + cost0
        out2, pre2, post2 = hourglass(32, out1, pre1, post1, 'stack2')
        out2 = out2 + cost0
        out3, pre3, post3 = hourglass(32, out2, pre1, post2, 'stack3')
        out3 = out3 + cost0

        cost1 = convbn_3d(out1, 32, 32, 1, 'SAME', 'classif1_1')
        cost1 = tf.nn.relu(cost1)
        with tf.variable_scope('classif1_2'):
            classif1_weight = tf.get_variable('weight', [3, 3, 3, 32, 1],
                                              initializer=tf.truncated_normal_initializer(stddev=0.1))
            cost1 = tf.nn.conv3d(cost1, classif1_weight, strides=[1, 1, 1, 1, 1], padding='SAME')

        cost2 = convbn_3d(out2, 32, 32, 1, 'SAME', 'classif2_1')
        cost2 = tf.nn.relu(cost2)
        with tf.variable_scope('classif2_2'):
            classif2_weight = tf.get_variable('weight', [3, 3, 3, 32, 1],
                                              initializer=tf.truncated_normal_initializer(stddev=0.1))
            cost2 = tf.nn.conv3d(cost2, classif2_weight, strides=[1, 1, 1, 1, 1], padding='SAME') + cost1

        cost3 = convbn_3d(out3, 32, 32, 1, 'SAME', 'classif3_1')
        cost3 = tf.nn.relu(cost3)
        with tf.variable_scope('classif3_2'):
            classif3_weight = tf.get_variable('weight', [3, 3, 3, 32, 1],
                                              initializer=tf.truncated_normal_initializer(stddev=0.1))
            cost3 = tf.nn.conv3d(cost3, classif3_weight, strides=[1, 1, 1, 1, 1], padding='SAME') + cost2

        # #if hyperparam.is_training:
        #     #cost1 = tf.image.resize_images(cost1, [left.shape[0], maxdisp, left.shape[2], left.shape[3], left.shape[4]],
        #                                    method=tf.image.ResizeMethod.BICUBIC)
        #     #cost2 = tf.image.resize_images(cost2, [left.shape[0], maxdisp, left.shape[2], left.shape[3], left.shape[4]],
        #                                    method=tf.image.ResizeMethod.BICUBIC)
        #     #cost1 = tf.squeeze(cost1, squeeze_dims=1)
        #     #pred1 = tf.nn.softmax(cost1, dim=1)
        #     #pred1 = disparity_regression(pred1, 'dispreg1')
        #
        #     #cost2 = tf.squeeze(cost2, squeeze_dims=1)
        #     #pred2 = tf.nn.softmax(cost2, dim=1)
        #     #pred2 = disparity_regression(pred2, 'dispreg2')

        # cost3 = tf.image.resize_images(cost3, [maxdisp, left.shape[2], left.shape[3]],
        #                                method=tf.image.ResizeMethod.BILINEAR)
        cost3 = tf.squeeze(cost3, axis=4)
        transpose = tf.transpose(cost3, [0, 2, 3, 1])

        upsample = tf.transpose(tf.image.resize_images(transpose, self.image_size_tf), [0, 3, 1, 2])
        upsample = tf.image.resize_images(upsample, tf.constant([maxdisp, self.height], dtype=tf.int32))
        #pred3 = tf.nn.softmax(cost3, dim=1)

        pred3 = disparity_regression(upsample, 'dispreg3')
        return pred3
        # if hyperparam.is_training:
        #     return pred1, pred2, pred3
        # else:
        #     return pred3
