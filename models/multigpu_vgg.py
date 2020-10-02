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
import time
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
loss_unbalance_w = 1.5
num_gpus = 2
batch_size = 10
num_steps = 1000000

def get_available_gpus():
    """
    code from http://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    """
    from tensorflow.python.client import device_lib as _device_lib
    local_device_protos = _device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']   # if x.device_type == 'GPU'


num_gpus = len(get_available_gpus())
print("Available GPU Number :" + str(num_gpus))
print('all_available_devices', get_available_gpus())

print([x.name for x in device_lib.list_local_devices()])                # if x.device_type == 'GPU'

PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']
def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return '/' + ps_device
        else:
            return device
    return _assign

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = [tf.expand_dims(g, 0) for g, _ in grad_and_vars]
        grads = tf.concat(grads, 0)
        grad = tf.reduce_mean(grads, 0)
        grad_and_var = (grad, grad_and_vars[0][1])
        average_grads.append(grad_and_var)
    return average_grads


def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        img_names_q, img_names_uq, img_num = vgg_train.load_all_img_names()
        aa, bb = [1, 0], [0, 1]
        len_qual, len_unqual = len(img_names_q), len(img_names_uq)
        qual_y = np.array(len_qual * aa).reshape(len_qual, 2)
        unqual_y = np.array(len_unqual * bb).reshape(len_unqual, 2)
        all_img_names = np.array(img_names_q + img_names_uq)
        ys = np.concatenate((qual_y, unqual_y), axis=0)
        print('Net build')
        global_step = tf.train.get_or_create_global_step()
        tower_grads = []
        tfx = tf.placeholder(tf.float32, [None, 224, 224, 1])
        tfy = tf.placeholder(tf.float32, [None, 2])
        opt = tf.train.MomentumOptimizer(0.0005, 0.9)
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpus):
                with tf.device(assign_to_device('/gpu:{}'.format(i), ps_device='/cpu:0')):
                    _tfx = tfx[i*batch_size: (i+1)*batch_size]
                    _tfy = tfy[i*batch_size: (i+1)*batch_size]
                    out, end_points = vgg.vgg_19(_tfx, num_classes=2)
                    tf.get_variable_scope().reuse_variables()
                    loss = tf.losses.softmax_cross_entropy(_tfy, out)
                    grads = opt.compute_gradients(loss)
                    tower_grads.append(grads)
                    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(_tfy, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        grads = average_gradients(tower_grads)
        train_op = opt.apply_gradients(grads)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, './model_vgg19_1225/transfer_learn_1')
            for step in range(1, num_steps):
                b_idx = np.random.randint(0, img_num, num_gpus*batch_size)
                batch_x, batch_y = vgg_train.load_img_by_name(all_img_names[b_idx]), ys[b_idx]
                ts = time.time()
                sess.run(train_op, feed_dict={tfx: batch_x, tfy: batch_y})
                te = time.time() - ts
                loss_value, acc = sess.run([loss, accuracy], feed_dict={tfx: batch_x, tfy: batch_y})
                if step % 500 == 0 or loss_value > 0.069:
                    print(step, 'train loss:', loss_value, 'accuracy', acc, '****', 'ys:', ys[b_idx], 'pred:',
                          sess.run(out, feed_dict={tfx: batch_x}))
                if step % 20 == 0:
                    saver.save(sess, './model_vgg19_1225/transfer_learn_%d' % step)

        qual_x, unqual_x, qual_y, unqual_y = vgg_train.load_data()
        xs = np.concatenate(qual_x + unqual_x, axis=0)
        ys = np.concatenate((qual_y, unqual_y), axis=0)
        b_idx = np.random.randint(0, len(xs), 2)
        img_bch, label_bch = [xs[b_idx], ys[b_idx]]
        out, end_points = vgg.vgg_19(img_bch, num_classes=2)  # 将VGG16升级为VGG19试试呢
        # net_flatten = tf.reshape(fc8, [-1, 1 * 6 * 2])
        # out = tf.layers.dense(net_flatten, 2, name='vgg_out')
        # print(aac_x, aae_x, aat_x, aac_y, aae_y, aat_y)
        print('Net build')
        opt = tf.train.MomentumOptimizer(0.0005, 0.9)

        sess = tf.Session()
        # sess.run(tf.global_variables_initializer())

        for i in range(20000000):
            tower_grads = []
            tower_loss = []
            for d in range(num_gpus):
                gpu_device_name = '/GPU:0' if d == 0 else '/device:XLA_GPU:0'
                # with tf.device('/gpu:%d' % d):
                with tf.device(gpu_device_name):
                    print('calculated by device /gpu: %d' % d)
                    with tf.name_scope('%s_%s' % ('tower', d)):
                        # tf.train.Saver().restore(sess, './model_vgg_1119/transfer_learn_2000')
                        # loss = tf.losses.softmax_cross_entropy(tfy, out)
                        aa, bb = tf.nn.softmax(tf.cast(label_bch, tf.float64)), tf.nn.softmax(out)
                        ce_2 = -tf.reduce_mean(
                            aa[0][0] * tf.math.log(bb[0][0]) + aa[0][1] * tf.math.log(bb[0][1]) * loss_unbalance_w)
                        # opt = tf.train.MomentumOptimizer(0.0005, 0.9)
                        # train_op = tf.train.MomentumOptimizer(0.0005, 0.9).minimize(loss)
                        with tf.device('/cpu:0'):
                            correct_prediction = tf.equal(
                                tf.argmax(out, 1),
                                tf.argmax(label_bch, 1))
                            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                        with tf.variable_scope('loss'):
                            grads = opt.compute_gradients(ce_2)
                            tower_grads.append(grads)
                            tower_loss.append(ce_2)
                            tf.get_variable_scope().reuse_variables()
            mean_loss = tf.stack(axis=0, values=tower_loss)
            mean_loss = tf.reduce_mean(mean_loss, axis=0)
            mean_grads = average_gradients(tower_grads)
            apply_gradient_op = opt.apply_gradients(mean_grads)

            # example, lbl, name = sess.run([image, label, img_name])  # 在会话中取出image和label   example shape [800,1200], 保存像素值

            # img = Image.fromarray(example)   #这里Image是之前提到的
            # img.save(cwd+str(i)+'_''Label_'+str(lbl)+'.jpg')#存下图片
            sess.run(tf.global_variables_initializer())
            _, losses, accuracy1 = sess.run((apply_gradient_op, ce_2, accuracy))
            tf.summary.scalar('loss', losses)
            merged_summaries = tf.summary.merge_all()
            fileWriter = tf.summary.FileWriter('./logs/vgg', graph=sess.graph)

            # losses, _, accuracy1 = sess.run((ce_2, train_op, accuracy), feed_dict={tfx: xs[b_idx], tfy: ys[b_idx]})
            if i % 500 == 0 or losses > 0.069:
                print(i, 'train loss:', losses, 'accuracy', accuracy1)
            if i % 10000 == 0:
                tf.train.Saver().save(sess, './model_vgg_1119/transfer_learn_%d' % i)
            # summary = sess.run(merged_summaries, feed_dict={loss: losses})
            # fileWriter.add_summary(summary=summary, global_step=i)
            #  add summary ---better method
            summary = tf.Summary(value=[
                tf.Summary.Value(tag='training_loss', simple_value=float(losses))
            ])
            fileWriter.add_summary(summary, i)
        # print(i, 'train loss:', losses)
    tf.train.Saver().save(sess, './model_vgg_1119/transfer_learn')


def test():
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, './model_vgg_1119/transfer_learn_380000')
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
    err_count = 0
    for i, img_test in enumerate(xst):
        img0 = cv2.imread('../data/test/qual/'+test_files[i])
        if img0 is None:
            img0 = cv2.imread('../data/test/unqual/'+test_files[i])
            print('---------this pic is unqual')
        else:
            print('---------this pic is qual')
        if img0 is None:
            print('PIC NOT FOUND')
        pre_out = sess.run((tf.nn.softmax(out)), feed_dict={tfx: xst[i][np.newaxis, :], tfy: yst[i][np.newaxis, :]})
        if not np.argmax(pre_out) == np.argmax(yst[i]) and pre_out[0][np.argmax(pre_out)] > 0.5:
            err_count += 1
            cv2.putText(img0, "NG Answer:%s,  Pred: %s" % (yst[i], pre_out), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
            if np.argmax(pre_out) == 0:
                cv2.imwrite('../data/vggrst/i am wrong/qual/'+test_files[i], img0)
            else:
                cv2.imwrite('../data/vggrst/i am wrong/unqual/' + test_files[i], img0)
        else:
            cv2.putText(img0, "GOOD Answer:%s,  Pred: %s" % (yst[i], pre_out), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imwrite('../data/vggrst/i am right/' + test_files[i], img0)

        print("the %d test image, predict is %s,ys is %s, error count is %d" % (i, pre_out,yst[i], err_count))


if __name__ == "__main__":
    train()
    # test()
