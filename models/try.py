# import os
# print(os.getcwd())
# print("this change is from outside, change again")
# import cv2
#
# img = cv2.imread('../data/train/stats1011/unqual4x/20191021 000498.png')
# img = cv2.resize(img, (600, 200))
# cv2.imshow('11', img)
# print(img.shape)
# # img = cv2.resize(img, (224, 224))
# # cv2.imshow('12', img)
#
# # img1 = cv2.flip(img, 1)
# # cv2.imshow('21', img1)
# #
# #
# # img2 = img[:, ::-1, :]
# # cv2.imshow('22', img2)
# cv2.waitKey(0)
import tensorflow as tf
b = tf.constant([0, 1, 2, 3, 4, 5], shape=[3, 2])
a = tf.less(b, 2)
c = tf.where(a, b, 2*b)
sess = tf.Session()
print(sess.run(a))
print(sess.run(c))