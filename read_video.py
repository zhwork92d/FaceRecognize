# -*- coding: utf-8 -*-
import cv2
import numpy as np

def readCapture():
    #使用opencv自带的函数定义摄像头对象，参数0表示第一个摄像头
    cap = cv2.VideoCapture(0)
    while(1):
        # get a frame
        ret, frame = cap.read()#使用摄像头对象读取视频
        # show a frame
        cv2.imshow("capture", frame)#显示
        if cv2.waitKey(1) & 0xFF == ord('q'):#等待1毫秒
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ =="__main__":
    readCapture()