import tensorflow as tf
import numpy as np
import os
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
import cv2
import sys

sys.path.append('..')
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# def load_img(path):
#     img = skimage.io.imread(path)
#     img = img / 255.0
#     short_edge = min(img.shape[:2])
#     yy = int((img.shape[0] - short_edge)/2)
#     xx = int((img.shape[1] - short_edge)/2)
#     crop_img = img[yy:yy+short_edge,xx:xx+short_edge]
#     resized_img = skimage.transform.resize(crop_img, (224,224))[None, :, :, :]  # [1,224,224,3]
#     return resized_img
DEST_DIR = 'stats7_0507'
TEST_DIR = 'tests7'
# PIC_RESIZE = (600, 200)  # 工位1011


PIC_RESIZE = (224, 224)  # 工位7


def load_img(path):
    # img = cv2.imread(path)
    img = cv2.imread(path, 0)
    # img = img[10:810, :]
    # H, W, _ = img.shape
    # crop_ratio = 0.2
    # crop_img = img[int(H*crop_ratio):int(H*(1-crop_ratio)), int(W*crop_ratio):int(W*(1-crop_ratio))]
    # resized_img = skimage.transform.resize(img, (320, 512))[None, :, :, :]        # [1,224,224,3]
    resized_img = cv2.resize(img, PIC_RESIZE)
    norm_img = resized_img / 255.0
    # ret_img = norm_img[None, :, :, :]
    ret_img = norm_img[None, :, :, None]
    # resized_img = skimage.transform.resize(img, (224, 224))[None, :, :, None]        # [1,224,224,3]
    return ret_img


def load_img_by_name(name_list):
    # dir_ = '../data/train_data_bk/train/训练图1227/'
    dir_ = '../data/train/%s/' % DEST_DIR
    batch_img = []
    for img_name in name_list:
        img0 = cv2.imread(dir_ + 'qual4x/' + img_name, 0)
        if img0 is None:
            img0 = cv2.imread(dir_ + 'unqual4x/' + img_name, 0)
            # print('---------this pic is unqual')
        # else:
        # print('---------this pic is qual')
        if img0 is None:
            print('PIC NOT FOUND')
        resized_img = cv2.resize(img0, PIC_RESIZE)
        norm_img = resized_img / 255.0
        ret_img = norm_img[:, :, None]
        # resized_img = skimage.transform.resize(img0, (224, 224))[:, :, None]
        batch_img.append(ret_img)
    return batch_img


def load_data():  # 2 kinds
    imgs = {'qual': [], 'unqual': []}
    for k in imgs.keys():
        dir_ = '../data/train_data_bk/' + 'train/' + k + '4x'
        for i, file in enumerate(os.listdir(dir_)):
            if "_" in file:
                print("ATTENTIN ", file, 'passed')
                continue
            try:
                print('the %d pic is reading' % i, 'pic', file)
                resized_img = load_img(os.path.join(dir_, file))
            except OSError:
                continue
            imgs[k].append(resized_img)
            print('the %d pic already readed' % i, 'pic', file)
            if len(imgs[k]) == 10:
                break
        print('the Num IS ', len(imgs[k]))
    aa, bb = [1, 0], [0, 1]
    len_qual, len_unqual = len(imgs['qual']), len(imgs['unqual'])
    qual_y = np.array(len_qual * aa).reshape(len_qual, 2)
    unqual_y = np.array(len_unqual * bb).reshape(len_unqual, 2)

    return imgs['qual'], imgs['unqual'], qual_y, unqual_y


def load_all_img_names():  # 2 kinds
    img_names = {'qual': [], 'unqual': []}
    train_data_num = 0
    for k in img_names.keys():
        dir_ = '../data/train/%s/' % DEST_DIR + k + '4x'
        # dir_ = '../data/train_data_bk/' + 'train/训练图1227/' + k + '4x'
        for i, file_name in enumerate(os.listdir(dir_)):
            if "_" in file_name:
                print("ATTENTIN ", file_name, 'passed')
                continue
            img_names[k].append(file_name)
            train_data_num += 1
            # print('the %d pic already readed' % i, 'pic', file_name)
            # if len(img_names[k]) == 10:
            #     break
    print('The num of train data is %d' % train_data_num)
    return img_names['qual'], img_names['unqual'], train_data_num


def load_test_data():  # 2 kinds
    test_data_files = []
    imgs = {'qual': [], 'unqual': []}
    for k in imgs.keys():
        dir_ = '../data/%s/' % TEST_DIR + k
        for i, file in enumerate(os.listdir(dir_)):
            if "_" in file:
                print("ATTENTIN ", file, 'passed')
                continue
            try:
                print('the %d pic is reading' % i, 'pic', file)
                resized_img = load_img(os.path.join(dir_, file))
            except OSError:
                continue
            imgs[k].append(resized_img)
            # if not file in test_data_files:
            test_data_files.append(file)
            print('the %d pic already readed' % i, 'pic', file, 'test file length is %s' % (len(test_data_files)))
            # if len(imgs[k]) == 20:
            #     break
        # print('the Num IS ', len(imgs[k]))
    aa, bb = [1, 0], [0, 1]
    len_qual, len_unqual = len(imgs['qual']), len(imgs['unqual'])
    qual_y = np.array(len_qual * aa).reshape(len_qual, 2)
    unqual_y = np.array(len_unqual * bb).reshape(len_unqual, 2)
    return imgs['qual'], imgs['unqual'], qual_y, unqual_y, test_data_files


def load_test_data4pb():  # 2 kinds
    test_data_files = []
    imgs = {'qual': [], 'unqual': []}  #
    for k in imgs.keys():
        dir_ = './data/' + 'testpb/' + k
        for i, file in enumerate(os.listdir(dir_)):
            if "_" in file:
                print("ATTENTIN ", file, 'passed')
                continue
            try:
                print('the %d pic is reading' % i, 'pic', file)
                resized_img = load_img(os.path.join(dir_, file))
            except OSError:
                continue
            imgs[k].append(resized_img)
            # if not file in test_data_files:
            test_data_files.append(file)
            print('the %d pic already readed' % i, 'pic', file, 'test file length is %s' % (len(test_data_files)))
            if len(imgs[k]) == 5:
                break
        # print('the Num IS ', len(imgs[k]))
    aa, bb = [1, 0], [0, 1]
    len_qual, len_unqual = len(imgs['qual']), len(imgs['unqual'])
    qual_y = np.array(len_qual * aa).reshape(len_qual, 2)
    unqual_y = np.array(len_unqual * bb).reshape(len_unqual, 2)
    return imgs['qual'], imgs['unqual'], qual_y, unqual_y, test_data_files
#
# class Vgg16:
#     vgg_mean = [103.939, 116.779, 123.68]
#
#     def __init__(self, npy_path=None, restore_from=None):
#         try:
#             self.data_dict = np.load(npy_path, encoding='latin1',allow_pickle=True).item()
#         except FileNotFoundError:
#             print('file not found ,download it again')
#         with tf.device('/gpu:1'):
#             self.tfx = tf.compat.v1.placeholder(tf.float32, [None, 224, 224, 3])
#             self.tfy = tf.compat.v1.placeholder(tf.float32, [None, 3])
#         # convert RGB to BGR
#         red, green, blue = tf.split(value=self.tfx*255.0, num_or_size_splits=3, axis=3)
#         bgr = tf.concat(axis=3, values=[
#             blue - self.vgg_mean[0],
#             green - self.vgg_mean[1],
#             red - self.vgg_mean[2]
#         ])
#         # pre_trained VGG network are fixed
#         conv1_1 = self.conv_layer(bgr, 'conv1_1')
#         conv1_2 = self.conv_layer(conv1_1, 'conv1_2')
#         pool1 = self.max_pool(conv1_2, 'pool1')
#
#         conv2_1 = self.conv_layer(pool1, 'conv2_1')
#         conv2_2 = self.conv_layer(conv2_1, 'conv2_2')
#         pool2 = self.max_pool(conv2_2, 'pool2')
#
#         conv3_1 = self.conv_layer(pool2, 'conv3_1')
#         conv3_2 = self.conv_layer(conv3_1, 'conv3_2')
#         conv3_3 = self.conv_layer(conv3_2, 'conv3_3')
#         pool3 = self.max_pool(conv3_3, 'pool3')
#
#         conv4_1 = self.conv_layer(pool3, 'conv4_1')
#         conv4_2 = self.conv_layer(conv4_1, 'conv4_2')
#         conv4_3 = self.conv_layer(conv4_2, 'conv4_3')
#         pool4 = self.max_pool(conv4_3, 'pool4')
#
#         conv5_1 = self.conv_layer(pool4, 'conv5_1')
#         conv5_2 = self.conv_layer(conv5_1, 'conv5_2')
#         conv5_3 = self.conv_layer(conv5_2, 'conv5_3')
#         pool5 = self.max_pool(conv5_3, 'pool5')
#
#         self.flatten = tf.reshape(pool5, [-1, 7*7*512])
#         self.fc6 = tf.layers.dense(self.flatten, 256, tf.nn.relu, name='fc6')
#         self.out = tf.layers.dense(self.fc6, 3, name='out')
#
#         # self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#         gpu_options = tf.GPUOptions(allow_growth=True)
#         self.sess = tf.compat.v1.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#         if restore_from:
#             saver = tf.train.Saver()
#             saver.restore(self.sess, restore_from)
#         else:
#             # self.loss = tf.losses.mean_squared_error(self.tfy, self.out)
#             self.loss = tf.losses.softmax_cross_entropy(self.tfy, self.out)
#             self.train_op = tf.compat.v1.train.RMSPropOptimizer(0.001).minimize(self.loss)
#             self.sess.run(tf.compat.v1.global_variables_initializer())
#
#     def conv_layer(self, bottom, name):
#         with tf.compat.v1.variable_scope(name):
#             conv = tf.nn.conv2d(bottom, self.data_dict[name][0], strides=[1, 1, 1, 1], padding='SAME')
#             lout = tf.nn.relu(tf.nn.bias_add(conv, self.data_dict[name][1]))
#             return lout
#
#     def max_pool(self, bottom, name):
#         return tf.nn.max_pool2d(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
#
#     def train(self, x, y):
#         loss, _ = self.sess.run([self.loss, self.train_op], {self.tfx: x, self.tfy: y})
#         # print('current_out_when_training:', self.sess.run(tf.nn.softmax(self.out),
#         #                                                   {self.tfx: x}), '\n', 'the_right_answer:', y)
#         return loss
#
#     def predict(self, paths):
#         # fig, axs = plt.subplots(1,2)
#         err_count = 0
#         # all_ = 0
#         for i, path in enumerate(paths):
#             try:
#                 x = load_img(path)
#                 print(np.shape(x))
#             except Exception as e:
#                 print('load_image error')
#             # if not np.shape(x) == (1, 224, 224, 3):
#             #     continue
#             # length = self.sess.run(tf.nn.softmax(self.out, {self.tfx:x}))
#             category = self.sess.run(self.out, {self.tfx: x})
#             category_out = self.sess.run(tf.nn.softmax(category))
#             if 'AAC'in path:
#                 if category_out[0][0] < 0.5:
#                     err_count += 1
#             elif 'AAE'in path:
#                 if category_out[0][1] < 0.5:
#                     err_count += 1
#             elif 'AAT'in path:
#                 if category_out[0][2] < 0.5:
#                     err_count += 1
#             print('type_of_pic:', category_out, 'err_count:', err_count, i+1, 'img', path)
#
#     def save(self, path):
#         saver = tf.train.Saver()
#         saver.save(self.sess, path, write_meta_graph=False)
#
#
# def train():
#     aac_x, aae_x, aat_x, aac_y, aae_y, aat_y = load_data()
#     # plt.hist(tiger_y, bins=20, label='Tiger')
#     # plt.hist(cat_y, bins=10, label='Cat')
#     # plt.legend()
#     # plt.xlabel('length')
#     # plt.show()
#
#     xs = np.concatenate(aac_x + aae_x + aat_x, axis=0)
#     ys = np.concatenate((aac_y, aae_y, aat_y), axis=0)
#     vgg = Vgg16(npy_path='./vgg16.npy')
#     print('Net build')
#     for i in range(100):
#         b_idx = np.random.randint(0, len(xs), 20)
#         train_loss = vgg.train(xs[b_idx], ys[b_idx])
#         print(i, 'train loss:', train_loss)
#     vgg.save('./model/transfer_learn')
#
#
# def eval():
#     vgg = Vgg16(npy_path='./vgg16.npy', restore_from='./model/transfer_learn')
#     aac_pic = []
#     aae_pic = []
#     aat_pic = []
#     for file in [file for file in os.listdir('../data/1016/AAC') if not os.path.isdir('../data/1016/AAC/'+file)]:
#         aac_pic.append('../data/1016/AAC/'+file)
#     for file1 in [file for file in os.listdir('../data/1016/AAE') if not os.path.isdir('../data/1016/AAE/'+file)]:
#         aae_pic.append('../data/1016/AAE/'+file1)
#     for file2 in [file for file in os.listdir('../data/1016/AAT') if not os.path.isdir('../data/1016/AAT/'+file)]:
#         aat_pic.append('../data/1016/AAT/'+file2)
#
#     # vgg.predict(tiger_pic)
#     # print('**********cat*cat***********')
#     vgg.predict(aac_pic[:20])
#     vgg.predict(aae_pic[:20])
#     vgg.predict(aat_pic[:20])
#
#
# if __name__ == "__main__":
#     # train()
#     eval()
