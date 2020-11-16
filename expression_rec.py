import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable
from time import perf_counter
import time

import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *

from ctypes import *
import darknet as dn
cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def process_frame():
    process_this_frame = True
    while True:
        ret, frame = video_capture.read()
        if ret == False:
            continue
        img_ori = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))   # Image
        height, width, _ = frame.shape
        scale_width = round(int(width) / 4096, 2)
        scale_height = round(int(height) / 2160, 2)
        img_IM, _ = dn.array_to_image(frame)  # im:IMAGE
        start1 = perf_counter()
        res_img = dn.detect(net2, meta, img_IM, thresh=.5, hier_thresh=.5, nms=.45, debug=False)
        end1 = perf_counter()
        print("yolo检测耗时:" + str(end1 - start1))
        print(res_img)

        # img_ori = np.array(Image.open('./images/kuaile.jpg'))    # image
        # height, width, _ = img_ori.shape
        # img_IM, _ = dn.array_to_image(img_ori)  # IMAGE
        # res_img = dn.detect(net2, meta, img_IM, thresh=.5, hier_thresh=.5, nms=.45, debug=False)
        # print(res_img)

        if len(res_img) != 0:
            for i in range(len(res_img)):
                if res_img[i][0] == b'person':
                    pos = res_img[i][2]
                    x = pos[0]
                    y = pos[1]
                    w = pos[2]
                    h = pos[3]
                    left = x - w / 2  # x0
                    right = x + w / 2   # x1
                    top = y - h / 2 # y0
                    bottom = y + h / 2  # y1
                    img_ori = np.array(img_ori)     # 切割前转换成array

                    cropped = img_ori[int(top):int(bottom), int(left):int(right)]

                    gray = rgb2gray(cropped)
                    gray = np.asarray(gray)
                    gray = cv2.resize(gray, (48, 48))
                    img_cropped = gray[:, :, np.newaxis]
                    img_cropped = np.concatenate((img_cropped, img_cropped, img_cropped), axis=2)
                    img_cropped = Image.fromarray(np.uint8(img_cropped))

                    inputs = transform_test(img_cropped)
                    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                    ncrops, c, h, w = np.shape(inputs)

                    inputs = inputs.view(-1, c, h, w)
                    inputs = inputs.cuda()
                    with torch.no_grad():
                        inputs = Variable(inputs)
                    outputs = net(inputs).to(device)

                    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
                    print("输出概率："+str(outputs_avg))

                    _, predicted = torch.max(outputs_avg.data, 0)
                    print("输出预测结果:"+str(predicted))
                    emojis_img = io.imread('images/emojis/%s.png' % str(class_names[int(predicted.to(device).cpu().numpy())]))

                    cv2.rectangle(img_ori, (int(left), int(top)), (int(right), int(bottom)), (135, 120, 28), 2)
                    cv2.rectangle(img_ori, (int(left), int(top-100*scale_height)), (int(right), int(top)), (35, 235, 185), -1)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(img_ori, class_names[int(predicted.to(device).cpu().numpy())], (int(x - 60*scale_width), int(top - 10*scale_height)), font, 3.0*scale_height, (30, 41, 61), 1)
                    cv2.imshow('emoji', emojis_img)
            img_ori = cv2.cvtColor(np.array(img_ori), cv2.COLOR_BGR2RGB)    # 从BGR转回RGB
            cv2.imshow('video', img_ori)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            img_ori = cv2.cvtColor(np.array(img_ori), cv2.COLOR_BGR2RGB)
            cv2.imshow('video', img_ori)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    #     out.write(frame)
    # cap.release()
    # out.release()
    video_capture.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    #video_capture = cv2.VideoCapture(0)
    video_capture = cv2.VideoCapture("rtsp://admin:qs123456@192.168.1.180:554/Streaming/Channels/1")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    net = VGG('VGG19')
    # net = ResNet18()
    # checkpoint = torch.load(os.path.join('CK+_VGG19', '1', 'Test_model.t7'), map_location='cuda:0') # 因为训练的时候用的双卡训练
    checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'))
    net.load_state_dict(checkpoint['net'])
    # net.cuda()
    net.to(device)
    net.eval()

    # cap = cv2.VideoCapture('./images/ph1.MP4')
    # fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #
    # out = cv2.VideoWriter('DJ1.mp4', 0x00000021, fps, size)
    # out = cv2.VideoWriter('ph1.mp4', fourcc, fps, size)
    # net2 = dn.load_net(b"/home/ubuntu/mfq/object_detection_train/yolo_trainning/data/pmb_0828/pmb_0828/yolov4.cfg",
    #                     b"/home/ubuntu/mfq/object_detection_train/yolo_trainning/data/pmb_0828/pmb_0828/yolov4_best.weights", 0)
    # meta = dn.load_meta(b"/home/ubuntu/mfq/object_detection_train/yolo_trainning/data/pmb_0828/pmb_0828/pmb_0828.data")

    net2 = dn.load_net(b"/home/ubuntu/mfq/object_detection_train/yolo_trainning/data/pmb_0828/pmb_0828/yolov4.cfg",
                       b"/home/ubuntu/mfq/object_detection_train/yolo_trainning/data/pmb_0828/pmb_0828/yolov4_best.weights", 0)
    meta = dn.load_meta(b"/home/ubuntu/mfq/object_detection_train/yolo_trainning/data/pmb_0828/pmb_0828/pmb_0828.data")
    process_frame()