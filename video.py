from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random


def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    parser=argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help=
                        "Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help=
                        "weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    parser.add_argument("--video", dest="videofile", help="Video file to run detection on",
                        default="2018-10-11_162316_724_yr_low.avi", type=str)
    
    return parser.parse_args()


args = arg_parse()
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()


num_classes = 80
classes = load_classes("data/coco.names")


# Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0 
assert inp_dim > 32

# If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()

# Set the model in evaluation mode
model.eval()


# 将方框和文字写在图片上
def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img


# Detection phase

videofile = args.videofile # or path to the video file.

cap = cv2.VideoCapture(videofile)  

# cap = cv2.VideoCapture(0)  for webcam

# 当没有打开视频时抛出错误
assert cap.isOpened(), 'Cannot capture source'

# frames用于统计图片的帧数
frames = 0  
start = time.time()

while cap.isOpened():
    # ret指示是否读入了一张图片，为true时读入了一帧图片
    ret, frame = cap.read()
    
    if ret:
        # 将图片按照比例缩放缩放，将空白部分用(128,128,128)填充，得到为416x416的图片。并且将HxWxC转换为CxHxW
        img = prep_image(frame, inp_dim)
        # cv2.imshow("a", frame)
        # 得到图片的W,H,是一个二元素tuple
        im_dim = frame.shape[1], frame.shape[0]
        # 先将im_dim变成长度为2的一维行tensor，再在1维度(列这个维度)上复制一次，变成1x4的二维tensor二维行tensor [W,H,W,H]，展开
        # 成1x4主要是在后面计算x1,y1,x2,y2各自对应的缩放系数时好对应上。
        im_dim = torch.FloatTensor(im_dim).repeat(1, 2)
                     
        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        # 只进行前向计算，不计算梯度
        with torch.no_grad():
            # 得到每个预测方框在输入网络图片(416x416)坐标系中的坐标和宽高以及目标得分以及各个类别得分(x,y,w,h,s,s_cls1,s_cls2...)
            # 并且将tensor的维度转换成(batch_size, grid_size*grid_size*num_anchors, 5+类别数量)
            output = model(Variable(img, volatile=True), CUDA)
        # 将方框属性转换成(ind,x1,y1,x2,y2,s,s_cls,index_cls)，去掉低分，NMS等操作，得到在输入网络坐标系中的最终预测结果
        output = write_results(output, confidence, num_classes, nms_conf = nms_thesh)

        # output的正常输出类型为float32,如果没有检测到目标时output元素为0，此时为int型，将会用continue进行下一次检测
        if type(output) == int:
            frames += 1
            print("FPS of the video is {:5.4f}".format(frames / (time.time() - start)))
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue

        # im_dim一行对应一个方框所在图片尺寸。在detect.py中一次测试多张图片，所以对应的im_dim_list是找到每个方框对应的图片的尺寸。
        # 而这里每次只有一张图片，每个方框所在图片的尺寸一样，只需将图片的尺寸的行数重复方框的数量次数即可
        im_dim = im_dim.repeat(output.size(0), 1)
        # 得到每个方框所在图片缩放系数
        scaling_factor = torch.min(416/im_dim, 1)[0].view(-1, 1)

        # 将方框的坐标(x1,y1,x2,y2)转换成相对于原始图片缩放后的坐标系中坐标
        output[:, [1, 3]] -= (inp_dim - scaling_factor*im_dim[:, 0].view(-1, 1))/2
        output[:, [2, 4]] -= (inp_dim - scaling_factor*im_dim[:, 1].view(-1, 1))/2

        # 将坐标映射回原始图片
        output[:, 1:5] /= scaling_factor

        # 将超过了原始图片范围的方框坐标限定在图片范围之内
        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

        # coco.names文件中保存的是所有类别的名字，load_classes()返回一个列表classes，每个元素是一个类别的名字
        classes = load_classes('data/coco.names')
        # 读入包含100个颜色的文件pallete，里面是100个三元组序列
        colors = pkl.load(open("pallete", "rb"))

        # 将每个方框的属性写在图片上
        list(map(lambda x: write(x, frame), output))

        cv2.imshow("frame", frame)
        # 等待1ms,如果有按键输入则返回按键值编码，输入q返回113
        key = cv2.waitKey(10000)
        # 如果输入q，终止程序。为什么要进行按位与操作？
        test = key & 0xFF
        if key & 0xFF == ord('q'):
            break
        # 统计已经处理过的帧数
        frames += 1
        print(time.time() - start)
        print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
    else:
        break     






