# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image
import sys

def readCapture():
    #使用opencv自带的函数定义摄像头对象，参数0表示第一个摄像头
    cap = cv2.VideoCapture(0)
    #告诉opencv使用人脸识别分类器
    classfier =cv2.CascadeClassifier('D:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
    color = (0,255,0)
    while (1):
        # get a frame
        ret, frame = cap.read()#使用摄像头对象读取视频
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("grey",grey)#一般是灰度图像加快检测速度 1.2表示每次搜索窗口每次扩大1.2倍
        faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.1, minNeighbors = 3, minSize = (32, 32))
        if len(faceRects) > 0:            #大于0则检测到人脸
            for faceRect in faceRects:  #单独框出每一张人脸
                x, y, w, h = faceRect
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        # show a frame
        cv2.imshow("capture", frame)#显示
        if cv2.waitKey(1) & 0xFF == ord('q'):#等待1毫秒
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ =="__main__":
    readCapture()