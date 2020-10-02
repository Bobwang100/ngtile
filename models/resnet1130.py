import tensorflow.contrib.slim as slim
import tensorflow as tf
from models import vgg_train
import numpy as np
from tensorflow.contrib.slim.nets import vgg as vgg
from tensorflow.contrib.slim.nets import resnet_v2 as resnet_v2
from tensorflow.contrib.slim.nets import resnet_v1 as resnet_v1
from tensorflow.python.client import device_lib
import os
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
loss_unbalance_w = 1.05

print([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'])
tfx = tf.placeholder(tf.float32, [None, 224, 224, 1])
tfy = tf.placeholder(tf.float32, [None, 2])
out, end_points = resnet_v2.resnet_v2_152(tfx, num_classes=2)     # 将VGG16升级为VGG19试试呢
out = tf.reshape(out, [-1, 2])
# net_flatten = tf.reshape(fc8, [-1, 1*6*2])
# out = tf.layers.dense(net_flatten, 2, name='vgg_out')
loss = tf.losses.softmax_cross_entropy(tfy, out)
# aa, bb = tf.nn.softmax(tfy), tf .nn.softmax(out)
# loss = -tf.reduce_mean(aa[0][0]*tf.log(bb[0][0]) + aa[0][1]*tf.log(bb[0][1])*loss_unbalance_w)
train_op = tf.train.MomentumOptimizer(0.0005, 0.9).minimize(loss)
# out, end_points = vgg.vgg_16(tfx, num_classes=2)
# loss = tf.losses.softmax_cross_entropy(tfy, out)
# train_op = tf.train.MomentumOptimizer(0.0005, 0.9).minimize(loss)
correct_prediction = tf.equal(
            tf.argmax(out, 1),
            tf.argmax(tfy, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
# tf.summary.scalar('loss', loss)
# merged_summaries = tf.summary.merge_all()
fileWriter = tf.summary.FileWriter('./logs/vgg', graph=sess.graph)
#


def train():
    saver.restore(sess, './model_resnet_1204ts/transfer_learn_1420000')
    img_names_q, img_names_uq, img_num = vgg_train.load_all_img_names()
    aa, bb = [1, 0], [0, 1]
    len_qual, len_unqual = len(img_names_q), len(img_names_uq)
    qual_y = np.array(len_qual * aa).reshape(len_qual, 2)
    unqual_y = np.array(len_unqual * bb).reshape(len_unqual, 2)
    # qual_x, unqual_x, qual_y, unqual_y = vgg_train.load_data()
    # print(aac_x, aae_x, aat_x, aac_y, aae_y, aat_y)
    # xs = np.concatenate(qual_x + unqual_x, axis=0)
    all_img_names = np.array(img_names_q + img_names_uq)
    ys = np.concatenate((qual_y, unqual_y), axis=0)
    print('Net build')
    for i in range(5000000):
        b_idx = np.random.randint(0, img_num, 1)
        xs = vgg_train.load_img_by_name(all_img_names[b_idx])
        losses, _, accuracy1 = sess.run((loss, train_op, accuracy), feed_dict={tfx: xs, tfy: ys[b_idx]})
        if i % 500 == 0 or losses > 0.069:
            print(i, 'train loss:', losses, 'accuracy', accuracy1, '****', 'ys:', ys[b_idx], 'pred:',
                  sess.run(out, feed_dict={tfx: xs}))
        if i % 10000 == 0:
            saver.save(sess, './model_resnet_1204ts/transfer_learn_%d' % i)
        # summary = sess.run(merged_summaries, feed_dict={loss: losses})
        # fileWriter.add_summary(summary=summary, global_step=i)
        #  add summary ---better method
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='training_loss', simple_value=float(losses))
        ])
        fileWriter.add_summary(summary, i)
        # print(i, 'train loss:', losses)
    saver.save(sess, './model_resnet_1204ts/transfer_learn')


def test():

    saver.restore(sess, './model_resnet_1130/transfer_learn_160000')
    qual_xt, unqual_xt, qual_yt, unqual_yt, test_files = vgg_train.load_test_data()
    # print(aac_x, aae_x, aat_x, aac_y, aae_y, aat_y)
    xst = np.concatenate(qual_xt + unqual_xt, axis=0)
    yst = np.concatenate((qual_yt, unqual_yt), axis=0)
    state = np.random.get_state()
    np.random.shuffle(xst)
    np.random.set_state(state)
    np.random.shuffle(yst)
    np.random.set_state(state)
    np.random.shuffle(test_files)
    print('Net build')
    err_nums = 0
    for i, img_test in enumerate(xst):
        img0 = cv2.imread('../data/test/qual/'+test_files[i])
        if img0 is None:
            img0 = cv2.imread('../data/test/unqual/'+test_files[i])
            print('---------this pic is unqual')
        else:
            print('---------this pic is qual')
        if img0 is None:
            print('PIC NOT  FOUND')
        # xs_stack = np.stack((xst[i], xst[i], xst[i], xst[i], xst[i], xst[i], xst[i], xst[i], xst[i], xst[i]))
        pre_out = sess.run((tf.nn.softmax(out)), feed_dict={tfx: xst[i][np.newaxis, :], tfy: yst[i][np.newaxis, :]})
        if pre_out[0][1] > 0.1:  # 此时判断为不合格
            if np.argmax(yst[i]) == 0:  # 真实值为合格
                err_nums += 1
                cv2.putText(img0, "WUJIAN Answer:%s,  Pred: %s" % (yst[i], pre_out), (50, 50), cv2.FONT_HERSHEY_COMPLEX,
                            1.0, (0, 0, 255), 2)
                cv2.imwrite('../data/resrst/i am wrong/qual/' + test_files[i], img0)
            else:
                cv2.putText(img0, "GOOD Answer:%s,  Pred: %s" % (yst[i], pre_out), (50, 50), cv2.FONT_HERSHEY_COMPLEX,
                            1.0, (0, 0, 255), 2)
                cv2.imwrite('../data/resrst/i am right/' + test_files[i], img0)
        else:  # 判断合格！！
            if np.argmax(yst[i]) == 1:
                err_nums += 1
                cv2.putText(img0, "LOUJIAN Answer:%s,  Pred: %s" % (yst[i], pre_out), (50, 50),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1.0, (0, 0, 255), 2)
                cv2.imwrite('../data/resrst/i am wrong/unqual/' + test_files[i], img0)
            else:
                cv2.putText(img0, "GOOD Answer:%s,  Pred: %s" % (yst[i], pre_out), (50, 50), cv2.FONT_HERSHEY_COMPLEX,
                            1.0, (0, 0, 255), 2)
                cv2.imwrite('../data/resrst/i am right/' + test_files[i], img0)
        # if not np.argmax(pre_out) == np.argmax(yst[i]) and pre_out[0][np.argmax(pre_out)] > 0.5:
        #     err_count += 1
        #     cv2.putText(img0, "NG Answer:%s,  Pred: %s" % (yst[i], pre_out), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
        #     if np.argmax(pre_out) == 0:
        #         cv2.imwrite('../data/vggrst/i am wrong/qual/'+test_files[i], img0)
        #     else:
        #         cv2.imwrite('../data/vggrst/i am wrong/unqual/' + test_files[i], img0)
        # else:
        #     cv2.putText(img0, "GOOD Answer:%s,  Pred: %s" % (yst[i], pre_out), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
        #     cv2.imwrite('../data/vggrst/i am right/' + test_files[i], img0)

        print("the %d test image, predict is %s,ys is %s, error count is %d" % (i, pre_out, yst[i], err_nums))


if __name__ == "__main__":
    # train()
    test()
