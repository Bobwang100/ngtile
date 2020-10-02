import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
# sys.path.append(os.path.abspath('%s/../..' % sys.path[0]))
import tensorflow as tf
from models import vgg_train
import numpy as np
from tensorflow.contrib.slim.nets import vgg as vgg
from models import fpn
from models import netvgg19
from tensorflow.contrib.slim.nets import resnet_v2 as resnet_v2
# from tensorflow.contrib.slim.nets import resnet_v1 as resnet_v1
# from tensorflow.python.client import device_lib
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
loss_unbalance_w = 1.05
TST_DIR = 'tests1011'
TRAINING_BATCH = 20


def choose_net(inputs, net_name):
    if net_name == 'vgg':
        net_out, end_points = vgg.vgg_19(inputs, num_classes=2)  # is_training=False
    elif net_name == 'resnet':
        net_out, end_points = resnet_v2.resnet_v2_152(inputs, num_classes=2)
        net_out = tf.reshape(net_out, (-1, 2))
    elif net_name == 'fpn_vgg':
        fpn_net = fpn.FPN(inputs=inputs,
                          net_name='vgg_19')
        net_out = fpn_net.net_scores
    elif net_name == 'fpn_res':
        fpn_net = fpn.FPN(inputs=inputs,
                          net_name='resnet_v1_101')
        net_out = fpn_net.net_scores
    else:
        raise ValueError('the chosen model ')
    return net_out


# print([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'])
# tfx = tf.placeholder(tf.float32, [None, 200, 600, 1])
tfx = tf.placeholder(tf.float32, [None, 224, 224, 1])

tfy = tf.placeholder(tf.float32, [None, 2])
out = choose_net(tfx, 'fpn_res')

# out, end_points = vgg.vgg_19(tfx, num_classes=2, is_training=False)
# out, end_points = resnet_v2.resnet_v2_152(tfx, num_classes=2, )    #  is_training=False
# out = tf.reshape(out, (-1, 2))

# out = netvgg19.VGG19(tfx, 0.9, 2).out
# net_flatten = tf.reshape(fc8, [-1, 1*6*2])
# out = tf.layers.dense(net_flatten, 2, name='vgg_out')
loss = tf.losses.softmax_cross_entropy(tfy, out)
# aa, bb = tf.nn.softmax(tfy), tf .nn.softmax(out)
# loss = -tf.reduce_mean(aa[0][0]*tf.log(bb[0][0]) + aa[0][1]*tf.log(bb[0][1])*loss_unbalance_w)
train_op = tf.train.MomentumOptimizer(0.0005, 0.8).minimize(loss)

# loss = tf.losses.softmax_cross_entropy(tfy, out)
# train_op = tf.train.MomentumOptimizer(0.0005, 0.9).minimize(loss)
correct_prediction = tf.equal(
    tf.argmax(out, 1),
    tf.argmax(tfy, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
# tf.summary.scalar('loss', loss)
# merged_summaries = tf.summary.merge_all()
fileWriter = tf.summary.FileWriter('./logs/vgg', graph=sess.graph)


#
def train():
    # saver.restore(sess, './model_fpn101_res_v1_0305/transfer_learn_105000')
    # saver.restore(sess, './model_resnet1113/transfer_learn_2250000')
    # saver.restore(sess, './model_vgg19_1211gpu/transfer_learn_10000')
    # saver.restore(sess, './model_vgg19_s70507/transfer_learn_35000')
    saver.restore(sess, './MARK_vgg19S7_1227/transfer_learn_110000')
    # saver.restore(sess, './MARK_model_resS1011_picadd_0219/transfer_learn_130000')
    img_names_q, img_names_uq, img_num = vgg_train.load_all_img_names()
    aa, bb = [1, 0], [0, 1]    # qual 【1，0】
    len_qual, len_unqual = len(img_names_q), len(img_names_uq)
    qual_y = np.array(len_qual * aa).reshape(len_qual, 2)
    unqual_y = np.array(len_unqual * bb).reshape(len_unqual, 2)
    # qual_x, unqual_x, qual_y, unqual_y = vgg_train.load_data()
    # print(aac_x, aae_x, aat_x, aac_y, aae_y, aat_y)
    # xs = np.concatenate(qual_x + unqual_x, axis=0)
    all_img_names = np.array(img_names_q + img_names_uq)
    ys = np.concatenate((qual_y, unqual_y), axis=0)
    print('Net build')
    for i in range(10000000):
        b_idx = np.random.randint(0, img_num, TRAINING_BATCH)
        xs = vgg_train.load_img_by_name(all_img_names[b_idx])
        losses, _ = sess.run((loss, train_op),
                             feed_dict={tfx: xs, tfy: ys[b_idx]})

        accuracy1 = sess.run(accuracy,
                             feed_dict={tfx: xs, tfy: ys[b_idx]})
        if i % 1 == 0 or losses > 0.069:
            print(i, 'train loss:',
                  losses, 'accuracy', accuracy1)
            # '****', 'ys:', ys[b_idx], 'pred:',sess.run(out, feed_dict={tfx: xs}))
        if i % 10 == 0:
            saver.save(sess, './MARK_vgg19S7_1227/transfer_learn_%d' % i)

        # summary = sess.run(merged_summaries, feed_dict={loss: losses})
        # fileWriter.add_summary(summary=summary, global_step=i)
        #  add summary ---better method
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='training_loss', simple_value=float(losses))
        ])
        fileWriter.add_summary(summary, i)
        # print(i, 'train loss:', losses)
    # saver.save(sess, './model_res_s1011_0219/transfer_learn')


def test():
    saver.restore(sess, './MARK_model_resS1011_picadd_0219/transfer_learn_130000')
    # saver.restore(sess, './model_fpn101_res_v1_0305/transfer_learn_105000')
    # saver.restore(sess, './model_fpn_vgg19_0306/transfer_learn_575000')
    # saver.restore(sess, './model_vgg19_1227/transfer_learn_110000')
    qual_xt, unqual_xt, qual_yt, unqual_yt, test_files = vgg_train.load_test_data()
    # print(aac_x, aae_x, aat_x, aac_y, aae_y, aat_y)
    xst = np.concatenate(qual_xt + unqual_xt, axis=0)
    yst = np.concatenate((qual_yt, unqual_yt), axis=0)
    # state = np.random.get_state()
    # np.random.shuffle(xst)
    # np.random.set_state(state)
    # np.random.shuffle(yst)
    # np.random.set_state(state)
    # np.random.shuffle(test_files)
    print('Net build')
    err_nums = 0
    for i, img_test in enumerate(xst):
        img0 = cv2.imread('../data/%s/qual/' % TST_DIR + test_files[i])
        if img0 is None:
            img0 = cv2.imread('../data/%s/unqual/' % TST_DIR + test_files[i])
            print('---------this pic is unqual')
        else:
            print('---------this pic is qual')
        if img0 is None:
            print('PIC NOT  FOUND')
        pre_out = sess.run((tf.nn.softmax(out)), feed_dict={tfx: xst[i][np.newaxis, :],
                                                            tfy: yst[i][np.newaxis, :]})

        if pre_out[0][1] > 0.5:  # declare ng   # 第一个是合格分数，第二个是不合格分数
            if np.argmax(yst[i]) == 0:  # truth is qual
                err_nums += 1
                cv2.putText(img0, "WJ Ans:%s,Pr:%s,f:%s" % (yst[i], pre_out, test_files[i]), (50, 50),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1.0, (0, 0, 255), 2)
                cv2.imwrite('../data/vggrst/i am wrong/qual/' + test_files[i], img0)
            else:
                cv2.putText(img0, "G Ans:%s,Pr: %s,f:%s" % (yst[i], pre_out, test_files[i]), (50, 50),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1.0, (0, 0, 255), 2)
                cv2.imwrite('../data/vggrst/i am right/qual/' + test_files[i], img0)
        else:  # declare qual
            if np.argmax(yst[i]) == 1:
                err_nums += 1
                cv2.putText(img0, "LJ Ans:%s,Pr: %s,f:%s" % (yst[i], pre_out, test_files[i]), (50, 50),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1.0, (0, 0, 255), 2)
                cv2.imwrite('../data/vggrst/i am wrong/unqual/' + test_files[i], img0)
            else:
                cv2.putText(img0, "G Ans:%s,Pr:%s,f:%s" % (yst[i], pre_out, test_files[i]), (50, 50),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1.0, (0, 0, 255), 2)
                cv2.imwrite('../data/vggrst/i am right/unqual/' + test_files[i], img0)
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

        print("the %d test image, predict is %s,ys is %s, error count is %d" % (i, pre_out, yst[i], err_nums),
              test_files[i])
        # saver.save(sess, './model_vgg19s9_0103/no_dropout')


if __name__ == "__main__":
    train()
    # test()
