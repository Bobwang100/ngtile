import tensorflow as tf
import os
from tensorflow.python.platform import gfile
import numpy as np
import models.vgg_train as vgg_train

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

pb_file_path = './models/pb/' #获取当前代码路径
sess = tf.Session()
with gfile.FastGFile(pb_file_path + 'vgg19_s7_1227no_drop.pb', 'rb') as f: #加载模型
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')  # 导入计算图

# 需要有一个初始化的过程
sess.run(tf.global_variables_initializer())
# 需要先复原变量
# print(sess.run('b:0'))
# 1
#下面三句，是能否复现模型的关键
# 输入
input_x = sess.graph.get_tensor_by_name('Placeholder:0')  #此处的x一定要和之前保存时输入的名称一致！
# input_x = sess.graph.get_tensor_by_name('IteratorGetNext:1')  #此处的x一定要和之前保存时输入的名称一致！
# input_y = sess.graph.get_tensor_by_name('Placeholder:1')  #此处的y一定要和之前保存时输入的名称一致！
# op = sess.graph.get_tensor_by_name('resnet_v2_152/logits/BiasAdd:0')  #此处的op_to_store一定要和之前保存时输出的名称一致！
# op = sess.graph.get_tensor_by_name('concat_9:0')  #此处的op_to_store一定要和之前保存时输出的名称一致！
op = sess.graph.get_tensor_by_name('vgg_19/fc8/squeezed:0')  #此处的op_to_store一定要和之前保存时输出的名称一致！
qual_xt, unqual_xt, qual_yt, unqual_yt, test_files = vgg_train.load_test_data4pb()
xst = np.concatenate(qual_xt + unqual_xt, axis=0)
yst = np.concatenate((qual_yt, unqual_yt), axis=0)
err_count = 0
valid_count = 0
for i in range(len(xst)):
    ret = sess.run(op, feed_dict={input_x: xst[i][np.newaxis, :]})
    # ret_S = sess.run(tf.nn.softmax(op), feed_dict={input_x: xst[i][np.newaxis, :]})
    ret_S = [np.exp(ret[0][0])/(np.exp(ret[0][0]) + np.exp(ret[0][1])),
             np.exp(ret[0][1])/(np.exp(ret[0][0]) + np.exp(ret[0][1]))]
    print('pic:', test_files[i], 'cal_result :', ret, 'soft:', ret_S)
    if ret_S[0] > 0.5:
        valid_count += 1
        print('valid_count', valid_count)
    # if not np.argmax(ret) == np.argmax(yst[i]):
    #     err_count += 1
    # print('the error count is %d' % err_count)
    # print(i, test_files[i], ret)
