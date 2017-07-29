# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import cv2
from keras import backend as K

IMAGE_SIZE = 64
def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    #print image.size

    #获取图像尺寸
    h, w,_= image.shape

    #对于长宽不相等的图片，找到最长的一边
    longest_edge = max(h, w)

    #计算短边需要增加多上像素宽度使其与长边等长
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass

    #RGB颜色
    BLACK = [0, 0, 0]

    #给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)

    #调整图像大小并返回
    return cv2.resize(constant, (height, width))

def read_image(image):
    image=resize_image(image)
    image = np.array(image)
    if K.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
        image = resize_image(image)                             #尺寸必须与训练集一致都应该是IMAGE_SIZE x IMAGE_SIZE
        image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))   #与模型训练不同，这次只是针对1张图片进行预测
    elif K.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
        image = resize_image(image)
        image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        #浮点并归一化
        image = image.astype('float32')
        image /= 255
    return image