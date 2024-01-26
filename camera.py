# from pyrealsense2 import pyrealsense2 as rs
import pyrealsense2 as rs
import json
import numpy as np
from retrying import retry
import os
import cv2
import matplotlib.pyplot as plt
import time


class Camera:
    pipeline = rs.pipeline()
    config = rs.config()
    def __init__(self):

        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        # self.camera = cv2.VideoCapture(0)
        profile = self.pipeline.start(self.config)

    def capture(self):
        frame = self.pipeline.wait_for_frames()
        # reg, img = self.camera.read()
        # print(reg)
        # print(img)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        color_rs = frame.get_color_frame()
        img = np.asanyarray(color_rs.get_data())
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        return img, img


if __name__ == '__main__':
    t0 = time.time()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    prf = pipeline.start(config)
    t1 = time.time()

    frame = pipeline.wait_for_frames()
    color_rs = frame.get_color_frame()

    img = np.asanyarray(color_rs.get_data())
    print(img.shape)
    cv2.imshow('img', img)
    t2 = time.time()
    print(t2-t0)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


    # camera = Camera()
    # camera.capture()
    # reg, img = camera.read()
    # cv2.imshow('img', img)
    # cv2.waitKey(0)


