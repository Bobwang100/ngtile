import os
import cv2

qual_path = './data/工位7-1122/合格/'
unqual_path = './data/工位7-1122/NG/'


def file_clean(path):
    num1 = 0
    for pic in os.listdir(path):
        num1 += 1
        # img = cv2.imread(path + pic)
        if num1 <= 7228:
            print(num1)
            # cv2.imwrite('./data/train/qual/' + pic, img)
        else:
            print(num1)
            img = cv2.imread(path + pic)
            cv2.imwrite('./data/train/qual/' + pic, img)
            # cv2.imwrite('./data/test/qual/' + pic, img)


def file_clean1(path):
    num1 = 0
    for pic in os.listdir(path):
        num1 += 1
        # img = cv2.imread(path + pic)
        if num1 <= 6472:
            print(num1)
            # cv2.imwrite('./data/train/unqual/' + pic, img)
        else:
            print(num1)
            img = cv2.imread(path + pic)
            cv2.imwrite('./data/train/unqual/' + pic, img)
            # cv2.imwrite('./data/test/unqual/' + pic, img)

file_clean(qual_path)
file_clean1(unqual_path)

