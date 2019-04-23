import os
import time

import tensorflow as tf
import psmnet
from hyperparam import *
from kittiloader import DataLoaderKITTI

def train():
    if is_server:
        left_img = '/input/training/image_2/'
        right_img = '/input/training/image_3/'
        disp_img = '/input/training/disp_occ_0/'
        batch_size = 2
        epoches = 200
        MODEL_SAVE_PATH = './results/'
        MODEL_NAME = 'psmnet'
        TB_PATH = '/output'
    else:
        left_img = 'D:\\data\\kitti\\training\\image_2\\'
        right_img = 'D:\\data\\kitti\\training\\image_3\\'
        disp_img = 'D:\\data\\kitti\\training\\disp_occ_0\\'
        batch_size = 2
        epoches = 200
        MODEL_SAVE_PATH = './results/'
        MODEL_NAME = 'psmnet'
        TB_PATH = './log'

    dg = DataLoaderKITTI(left_img, right_img, disp_img, batch_size)

    with tf.Session() as sess:
        PSMNet = psmnet.PSMNet(sess, height=256, weight=512, batch_size=batch_size)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('loaded')
        for epoch in range(1, epoches + 1):
            total_train_loss = 0

            for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(dg.generator(is_training=True)):
                start_time = time.time()
                train_loss = PSMNet.train(imgL_crop, imgR_crop, disp_crop_L)
                print('Epoch %d Iter %d training loss = %.3f , time = %.2f' % (epoch, batch_idx, train_loss, time.time() - start_time))
                total_train_loss += train_loss
                # saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
                # print('saved')
            avg_loss = total_train_loss / (200 // batch_size)
            if epoch % 5 == 0 and epoch != 0:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
                print('saved')
            print('epoch %d avg training loss = %.3f' % (epoch, avg_loss))

        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
        print('final saved')

if __name__ == '__main__':
    train()