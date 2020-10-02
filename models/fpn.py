from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1 as resnet_v1
from tensorflow.contrib.slim.nets import vgg as vgg
from tensorflow.python.ops import math_ops

NUM_CLASSES = 2
WEIGHT_DECAY = 0.0001
LEVEL = ['P2', 'P3', 'P4', 'P5', "P6"]


class FPN(object):
    def __init__(self, inputs, net_name, share_head=False):
        self.inputs = inputs
        self.net_name = net_name
        self.share_head = share_head
        self.num_classes = NUM_CLASSES
        if self.net_name == 'resnet_v1_101':
            _, self.share_net = self.get_network_by_name(self.net_name,
                                                         self.inputs
                                                         )
        elif self.net_name == 'vgg_19':
            _, self.share_net = vgg.vgg_19(inputs=inputs,
                                           num_classes=self.num_classes,
                                           is_training=True,
                                           )                               # dont know why,the vggnet cant put into function
        self.level = LEVEL
        self.feature_maps_dict = self.get_feature_maps()
        self.feature_pyramid = self.build_feature_pyramid()
        self.net_scores = self.fpn_net()

    def get_network_by_name(self, net_name,
                            inputs,
                            num_classes=None,
                            is_training=True,
                            global_pool=True,
                            output_stride=None,
                            ):
        if net_name == 'resnet_v1_50':
            with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=0.0001)):
                logits, end_points = resnet_v1.resnet_v1_50(inputs=inputs,
                                                            num_classes=num_classes,
                                                            is_training=is_training,
                                                            global_pool=global_pool,
                                                            output_stride=output_stride,
                                                            )
            return logits, end_points

        if net_name == 'resnet_v1_101':
            with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=0.0001)):
                logits, end_points = resnet_v1.resnet_v1_101(inputs=inputs,
                                                             num_classes=num_classes,
                                                             is_training=is_training,
                                                             global_pool=global_pool,
                                                             output_stride=output_stride,
                                                             )
            return logits, end_points
        if net_name == 'vgg_19':
            with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=0.0001)):
                logits, end_points = vgg.vgg_19(inputs=inputs,
                                                num_classes=num_classes,
                                                is_training=is_training,
                                                )
            return logits, end_points

    def get_feature_maps(self):
        with tf.variable_scope('get_feature_maps'):
            if self.net_name == 'resnet_v1_50':
                feature_maps_dict = {
                    'C2': self.share_net['resnet_v1_50/block1/unit_2/bottleneck_v1'],  # [56,56]
                    'C3': self.share_net['resnet_v1_50/block2/unit_3/bottleneck_v1'],  # [28,28]
                    'C4': self.share_net['resnet_v1_50/block3/unit_5/bottleneck_v1'],  # [14,14]
                    'C5': self.share_net['resnet_v1_50/block4']  # [7,7]
                }
            elif self.net_name == 'resnet_v1_101':
                feature_maps_dict = {
                    'C2': self.share_net['resnet_v1_101/block1/unit_2/bottleneck_v1'],  # [56, 56]
                    'C3': self.share_net['resnet_v1_101/block2/unit_3/bottleneck_v1'],  # [28, 28]
                    'C4': self.share_net['resnet_v1_101/block3/unit_22/bottleneck_v1'],  # [14, 14]
                    'C5': self.share_net['resnet_v1_101/block4']  # [7, 7]
                }
            elif self.net_name == 'vgg_19':
                feature_maps_dict = {
                    'C2': self.share_net['vgg_19/pool2'],  # [56, 56]
                    'C3': self.share_net['vgg_19/pool3'],  # [28, 28]
                    'C4': self.share_net['vgg_19/pool4'],  # [14, 14]
                    'C5': self.share_net['vgg_19/pool5']  # [7, 7]
                }
            else:
                raise Exception('get no feature maps')
            return feature_maps_dict

    def build_feature_pyramid(self):
        '''
        build P2 P3 P4 P5
        :return: multi-scale feature map
        '''

        feature_pyramid = {}
        with tf.variable_scope('build_feature_pyramid'):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY)):
                feature_pyramid['P5'] = slim.conv2d(self.feature_maps_dict['C5'],
                                                    num_outputs=256,
                                                    kernel_size=[1, 1],
                                                    stride=1,
                                                    scope='build_P5')
                feature_pyramid['P6'] = slim.max_pool2d(feature_pyramid['P5'],
                                                        kernel_size=[2, 2], stride=2, scope='build_P6')
                for layer in range(4, 1, -1):
                    p, c = feature_pyramid['P' + str(layer + 1)], self.feature_maps_dict['C' + str(layer)]
                    up_sample_shape = tf.shape(c)
                    up_sample = tf.image.resize_nearest_neighbor(p, [up_sample_shape[1], up_sample_shape[2]],
                                                                 name='build_P%d/up_sample_nearest_neighbor' % layer)

                    c = slim.conv2d(c, num_outputs=256, kernel_size=[1, 1], stride=1,
                                    scope='build_P%d/reduce_dimension' % layer)
                    p = up_sample + c
                    p = slim.conv2d(p, 256, kernel_size=[3, 3], stride=1,
                                    padding='SAME', scope='build_P%d/avoid_aliasing' % layer)
                    feature_pyramid['P' + str(layer)] = p
        return feature_pyramid

    def fpn_net(self):
        scores_list = []
        with tf.variable_scope('fpn_net'):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY)):
                for level in self.level:
                    if self.share_head:
                        reuse_flag = None if level == 'P2' else True
                        scope_list = ['conv2d_3x3', 'rpn_classifier']
                    else:
                        reuse_flag = None
                        scope_list = ['conv2d_3x3_' + level, 'rpn_classifier_' + level]
                    rpn_conv2d_3x3 = slim.conv2d(inputs=self.feature_pyramid[level],
                                                 num_outputs=512,
                                                 kernel_size=[3, 3],
                                                 stride=1,
                                                 scope=scope_list[0],
                                                 reuse=reuse_flag)
                    rpn_box_scores = slim.conv2d(rpn_conv2d_3x3,
                                                 num_outputs=NUM_CLASSES,
                                                 kernel_size=[3, 3],
                                                 stride=1,
                                                 scope=scope_list[1],
                                                 activation_fn=None,
                                                 reuse=reuse_flag
                                                 )
                    scores_reduce = math_ops.reduce_mean(rpn_box_scores, [1, NUM_CLASSES], name='pool5', keepdims=True)
                    scores_reshape = tf.reshape(scores_reduce, [-1, NUM_CLASSES])
                    scores_list.append(scores_reshape)
                scores_mean = tf.reduce_mean(scores_list, axis=0)
                return scores_mean


if __name__ == '__main__':
    inputs = tf.placeholder(tf.float32, shape=[None, 224, 224, 1])

    # logits, end_points = vgg.vgg_19(inputs=inputs,
    #                                 num_classes=2,
    #                                 is_training=True,
    #                                 )
    fpn = FPN(inputs=inputs, net_name='vgg_19', )
    fpn.get_feature_maps()
