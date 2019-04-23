from psmnet import PSMNet
from kittiloader import DataLoaderKITTI
import tensorflow as tf
import numpy as np
import cv2
import hyperparam

def main():
    left_img = 'D:\\data\\kitti2012\\training\\colored_0\\000017_10.png'
    right_img = 'D:\\data\\kitti2012\\training\\colored_1\\000017_10.png'
    left_img = 'D:\\data\\kitti\\testing\\image_2\\000199_10.png'
    right_img = 'D:\\data\\kitti\\testing\\image_3\\000199_10.png'
    height = 368
    width = 1232
    path='D:\\data\\psmbak\\r\\input\\psmdl\\results'
    #path = './results'
    hyperparam.is_training = False
    bat_size = 1
    TB_PATH = './log'

    with tf.Session() as sess:
        psmnet = PSMNet(sess, height=height, weight=width, batch_size=bat_size)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('loaded')

        # reader = tf.train.NewCheckpointReader(ckpt.model_checkpoint_path)
        # all_v = reader.get_variable_to_shape_map()
        # ww = reader.get_tensor('cnn_conv0_1/cweight01')
        # print(ww)

        img_L = cv2.cvtColor(cv2.imread(left_img), cv2.COLOR_BGR2RGB)
        img_L = cv2.resize(img_L, (width, height))
        img_R = cv2.cvtColor(cv2.imread(right_img), cv2.COLOR_BGR2RGB)
        img_R = cv2.resize(img_R, (width, height))

        img_L = DataLoaderKITTI.mean_std(img_L)
        img_L = np.expand_dims(img_L, axis=0)
        img_R = DataLoaderKITTI.mean_std(img_R)
        img_R = np.expand_dims(img_R, axis=0)
        pred = psmnet.predict(img_L, img_R)
        pred_mat = np.array(pred)
        pred_mat = np.squeeze(pred_mat)
        print(pred_mat)
        item = (pred_mat * 255 / pred_mat.max()).astype(np.uint8)
        pred_rainbow = cv2.applyColorMap(item, cv2.COLORMAP_RAINBOW)
        cv2.imwrite('prediction.png', pred_rainbow)


if __name__ == '__main__':
    main()
