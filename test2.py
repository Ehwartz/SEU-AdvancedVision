import os
import numpy as np
import pyrealsense2 as rs
import cv2
import time

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
prf = pipeline.start(config)


def capture_img():
    frame = pipeline.wait_for_frames()
    color_rs = frame.get_color_frame()
    img = np.asanyarray(color_rs.get_data())
    return img


if __name__ == '__main__':
    time.sleep(5)
    img0 = capture_img()
    print(img0.shape)
    img0 = cv2.GaussianBlur(img0, (3, 3), 0)
    img1 = cv2.cvtColor(cv2.GaussianBlur(capture_img(), (3, 3), 0), cv2.COLOR_BGR2GRAY)
    print(1)
    time.sleep(2)
    img2 = cv2.cvtColor(cv2.GaussianBlur(capture_img(), (3, 3), 0), cv2.COLOR_BGR2GRAY)
    print(2)
    time.sleep(1)
    img1 = cv2.threshold(img1, 63, 255, cv2.THRESH_BINARY)
    img2 = cv2.threshold(img2, 63, 255, cv2.THRESH_BINARY)
    # print(img1)
    delta = img2[1]-img1[1]
    avg = np.average(delta)
    print(avg)
    print(delta)
    # print(delta.sum(axis=2))
    # print(delta.sum(axis=1))
    # print(delta.sum(axis=0))


    cv2.imshow('Image', img0)
    cv2.waitKey(0)

