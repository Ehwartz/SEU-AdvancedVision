import torch
import numpy as np
from models.common import DetectMultiBackend
import torchvision
# from PIL import Image
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator

import cv2

imgsz = (640, 640)  # inference size (height, width)
conf_thres = 0.25  # confidence threshold
iou_thres = 0.45  # NMS IOU threshold
max_det = 1000
classes = None
agnostic_nms = False
line_thickness = 3

weights = './yolov5s.pt'

model = DetectMultiBackend(weights=weights)
stride, names, pt = model.stride, model.names, model.pt
bs = 1
model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))
img = cv2.imread('./bus.jpg')
img0 = cv2.imread('./bus.jpg')
img = letterbox(img)[0]
img = img.transpose((2, 0, 1))[::-1]
img = np.ascontiguousarray(img)
img = torch.from_numpy(img)
img = img.half() if model.fp16 else img.float()
img /= 255
if __name__ == '__main__':
    print(img.shape)
    img = img[None]
    print(img.shape)
    pred = model(img)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    # print(pred[0][0])
    for i, det in enumerate(pred):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        print(det.shape)
        print(det)

