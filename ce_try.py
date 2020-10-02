# import tensorflow as tf
# from models import vgg_train
# from tensorflow.contrib.slim.nets import resnet_v2 as resnet_v2
# import numpy as np
# import os
# # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# # # a = tf.placeholder(tf.float32, shape=(1,))
# # # b = tf.placeholder(tf.float32, shape=(1,))
# # a = tf.constant((1, 0, 0), shape=(1, 3), dtype=tf.float32)
# # b = tf.constant((0.0001, 1, 12), shape=(1, 3), dtype=tf.float32)
# # ce = tf.losses.softmax_cross_entropy(a, b)
# # aa = tf.nn.softmax(a)
# # bb = tf.nn.softmax(b)
# # ce_1 = tf.reduce_mean(-tf.reduce_sum(a*tf.log(bb), reduction_indices=[1]))
# # sess = tf.Session()
# # print(sess.run(ce))
# # print(sess.run((ce_1, aa, bb)))
# from tensorflow.python.client import device_lib
#
#
# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']
#
#
# print(get_available_gpus())
# import pynvml
# from pynvml import *
# pynvml.nvmlInit()
# # 这里的1是GPU id
# handle = pynvml.nvmlDeviceGetHandleByIndex(1)
# meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
# print(meminfo.total) #第二块显卡总的显存大小
# print(meminfo.used)#这里是字节bytes，所以要想得到以兆M为单位就需要除以1024**2
# print(meminfo.free) #
# deviceCount = nvmlDeviceGetCount()#几块显卡
# for i in range(deviceCount):
#     handle = nvmlDeviceGetHandleByIndex(i)
#     print ("Device", i, ":", nvmlDeviceGetName(handle)) #
