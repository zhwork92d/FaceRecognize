# -*- coding: utf-8 -*-

import cv2
import sys
import gc
from keras.models import Sequential
from keras.models import *
import h5py
import numpy
import random
import numpy as np
from keras.models import Sequential
from keras.utils import np_utils
from keras import backend as K
from face_recognize_image import *

def keras_recognize():
    IMAGE_SIZE=64
    #加载模型
    model=model_from_json(open("my_facemodel_ver1.json","r").read())
    model.load_weights("my_facemodel_ver1_weights.h5")
    print("load model success!")
    #框住人脸的矩形边框颜色
    color = (0, 255, 0)

    #捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture(0)

    #人脸识别分类器本地存储路径
    cascade_path = "D:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml"


    #循环检测识别人脸
    while True:
        _, frame = cap.read()   #读取一帧视频

        #图像灰化，降低计算复杂度
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #使用人脸识别分类器，读入分类器
        cascade = cv2.CascadeClassifier(cascade_path)

        #利用分类器识别出哪个区域为人脸
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect

                #截取脸部图像提交给模型识别这是谁
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                images=read_image(image)
                #给出输入属于各个类别的概率，我们是二值类别，则该函数会给出输入图像属于0和1的概率各为多少
                result = model.predict_proba(images)
                print("result:",result)#1,0代表0 ，0代表zhou
                print('周惠:%.5f\n叶珑%.5f\n'%(result[0][0],result[0][1]))
                #给出类别预测：0或者1
                results = model.predict_classes(images)
                faceID=results[0]
                print("faceID%d:"%faceID)

                #如果是“我”
                if faceID == 0:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)

                    #文字提示是谁
                    cv2.putText(frame,'zhou',
                                (x + 30, y + 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (255,0,255),                           #颜色
                                2)                                     #字的线宽
                elif faceID==1:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)

                    #文字提示是谁
                    cv2.putText(frame,'ye',
                                (x + 30, y + 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (255,0,255),                           #颜色
                                2)                                     #字的线宽
                else:
                    pass

        cv2.imshow("face recognition", frame)

        #等待10毫秒看是否有按键输入
        k = cv2.waitKey(10)
        #如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break

    #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    keras_recognize()