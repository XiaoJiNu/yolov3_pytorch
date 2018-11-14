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
    # 创建一个ArgumentParser对象
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    # imames 是所有测试图片所在的文件夹
    parser.add_argument("--images", dest='images', help=
                        "Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--det", dest='det', help="Image / Directory to store detections to",
                        default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile", default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    
    return parser.parse_args()


# args是一个namespace类型的变量，就像一个字典，key来索引变量的值   Namespace(bs=1, cfgfile='cfg/yolov3.cfg', confidence=0.5,
# det='det', images='imgs', nms_thresh=0.4, reso='416', weightsfile='yolov3.weights')
args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start=0
CUDA=torch.cuda.is_available()

# coco有80类
num_classes = 80
# coco.names文件中保存的是所有类别的名字，load_classes()返回一个列表classes，每个元素是一个类别的名字
classes = load_classes("data/coco.names")


# Set up the neural network
print("Loading network.....")
# Darknet类中初始化时得到了网络结构和网络的参数信息，保存在net_info，module_list中
model = Darknet(args.cfgfile)
# 将权重文件载入，并复制给对应的网络结构model中
model.load_weights(args.weightsfile)
print("Network successfully loaded")

# model类中net_info是一个字典。’’height’’是图片的宽高，因为图片缩放到416x416，所以宽高一样大
model.net_info["height"] = args.reso
inp_dim=int(model.net_info["height"])
# 如果图片的尺寸不是32的位数或者不大于32，抛出异常
assert inp_dim % 32 == 0 
assert inp_dim > 32

# If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()


# Set the model in evaluation mode 变成测试模式，这主要是对dropout和batch normalization的操作在训练和测试的时候是不一样的
model.eval()
# 开始计时
read_dir = time.time()
# Detection phase
try:
    # imlist列表保存了images文件中所有图片的完整路径，一张图片路径对应一个元素。 osp.realpath('.')得到了realpath()这个方法所在脚本
    # (detect.py)的文件夹的绝对路径，images是测试图片文件夹，listdir(images)得到了images文件夹下面所有图片的名字。通过join()得
    # 一张图片的完整路径
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    # 如果上面的路径有错，只得到images文件路径即可
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print("No file or directory with the name {}".format(images))
    exit()

# 如果保存结果的文件路径不存在，创建一个
if not os.path.exists(args.det):
    os.makedirs(args.det)

# 开始载入图片的时间。 load_batch - read_dir 得到读取所有图片路径的时间
load_batch = time.time()
# 将所有的图片读入，一张图片的数组在loaded_ims列表中保存为一个元素
loaded_ims = [cv2.imread(x) for x in imlist]

# 这里是将每张图片的opencv格式BxHxWxC用prep_image函数转换为pytorch格式BxCxHxW.BGR到RGB的转换没有看到？？lodded_ims是
# 所有图片数组的列表，[inp_dim for x in range(len(imlist))] 遍历imlist长度(即图片的数量)这么多次，每次返回值是图片的
# 尺寸inp_dim, loaded_ims和[inp_dim for x in range(len(imlist))]是两个列表，map函数将对应的元素作为参数传入prep_image
# 函数，最终的所有结果也会组成一个列表(im_batches)，是BxCxHxW
im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))
# im_dim_list用于保存原始图片的维度。一个元素对应一张图片的宽高,opencv读入的图片矩阵对应的是 HxWxC
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
# 将im_dim_list转换为floatTensor类型的tensor，此时维度为11x2，并且每个元素沿着第二维(列的方向)进行复制，最终变成11x4的tensor
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)


# Create the Batches，将所有测试图片按照batch size分成多个batch
leftover = 0
# 如果测试图片的数量不能被batch size整除，leftover=1
if (len(im_dim_list) % batch_size):
    leftover = 1
# 如果batch size 不等于1，则将一个batch的图片作为一个元素保存在im_batches中，按照if语句里面的公式计算。如果batch_size=1,
# 则每一张图片作为一个元素保存在im_batches中
if batch_size != 1:
    # 如果batch size 不等于1，则batch的数量 = 图片数量//batch_size + leftover(测试图片的数量不能被batch size整除，leftover=1，
    # 否则为0)。比如这里有11张图片，假设batch_size=2,则batch数量=6
    num_batches = len(imlist) // batch_size + leftover
    # 前面的im_batches变量将所有的图片以BxCxHxW的格式保存。而这里将一个batch的所有图片在B这个维度(0维度)上进行连接，torch.cat()
    # 默认在0维上进行连接。将这个连接后的tensor作为im_batches列表的一个元素。第i个batch在前面的im_batches变量中所对应的元素
    # 就是i*batch_size: (i + 1)*batch_size，但是最后一个batch如果用(i + 1)*batch_size可能会超过图片数量的len(im_batches)长度，
    # 所以取min((i + 1)*batch_size, len(im_batches)
    im_batches = [torch.cat((im_batches[i*batch_size: min((i + 1)*batch_size, len(im_batches))])) for i in range(num_batches)]


# The Detection Loop
write = 0

if CUDA:
    im_dim_list = im_dim_list.cuda()

# 开始计时，计算开始检测的时间。start_det_loop - load_batch 为读入所有图片并将它们分成不同batch的时间
start_det_loop = time.time()
# enumerate返回im_batches列表中每个batch在0维连接成一个元素的tensor和这个tensor在im_batches中的序号。本例子中batch只有一张图片
for i, batch in enumerate(im_batches):
    # load the image
    start = time.time()
    if CUDA:
        batch = batch.cuda()
    # 取消梯度计算
    with torch.no_grad():
        # Variable(batch)将图片生成一个可导tensor，现在已经不再支持这种写法，Autograd automatically supports Tensors with
        #  requires_grad set to True。
        # prediction是一个batch所有图片通过yolov3模型得到的预测值，维度为1x10647x85，三个scale的图片每个scale的特
        # 征图大小为13x13,26x26,52x52,一个元素看作一个格子，每个格子有3个anchor，将一个anchor保存为一行，所以prediction
        # 一共有(13x13+26x26+52x52)x3=10647行，一个anchor预测(x,y,w,h,s,s_cls1,s_cls2...s_cls_80)，一共有85个元素。所以
        # prediction的维度为Bx10647x85，加为这里batch size为1，所以prediction的维度为1x10647x85
        prediction = model(Variable(batch), CUDA)

    # 这里返回了经过NMS后剩下的方框，最终每个方框的属性为(ind,x1,y1,x2,y2,s,s_cls,index_cls) ind是这个方框所属图片在这个batch
    # 中的序号，x1,y1是在网络输入图片(416x416)坐标系中，方框左上角的坐标；x2,y2是在网络输入图片(416x416)坐标系中，方框右下角的坐标。
    # s是这个方框含有目标的得分，s_cls是这个方框中所含目标最有可能的类别的概率得分，index_cls是s_cls对应的这个类别在所有类别中所对
    # 应的序号。这里prediction维度是3x8，表示有3个框
    prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thesh)

    end = time.time()

    # 如果从write_results()返回的一个batch的结果是一个int(0)，表示没有检测到时目标，此时用continue跳过本次循环
    if type(prediction) == int:
        # 在imlist中，遍历一个batch所有的图片对应的元素(即每张图片的存储位置)，同时返回这张图片在这个batch中的序号im_num
        for im_num, image in enumerate(imlist[i*batch_size: min((i + 1)*batch_size, len(imlist))]):
            # 计算图片在imlist中所对应的序号,即在所有图片中的序号
            im_id = i*batch_size + im_num
            # 打印图片运行的时间，用一个batch的平均运行时间来表示。.3f就表示保留三位小数点的浮点
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            # 输出本次处理图片所有检测到的目标的名字
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
        continue

    # prediction[:, 0]取出了每个方框在所在图片在这个batch(第i个batch)中的序号，加上i*batch_size，就将prediction中每个框(一行)的
    # 第一个元素变成了这个框所在图片在imlist中的序号，即在所有图片中的序号
    prediction[:, 0] += i*batch_size    # transform the atribute from index in batch to index in imlist

    # 这里用一个write标志来标记是否是第一次得到输出结果，因为每次的结果要进行torch.cat()操作，而一个空的变量不能与tensor连接，
    # 所以第一次将它赋值给output，后面就直接进行cat()操作
    if not write:                      # If we have't initialised output
        output = prediction
        write = 1
    else:
        # output将每个batch的输出结果在0维进行连接，即在行维度上连接，每行表示一个检测方框的预测值。最终本例子中的11张图片检测得到
        # 的结果output维度为 34 x 8
        output = torch.cat((output, prediction))

    # 在imlist中，遍历一个batch所有的图片对应的元素(即每张图片的存储位置)，同时返回这张图片在这个batch中的序号im_num
    for im_num, image in enumerate(imlist[i*batch_size: min((i + 1)*batch_size, len(imlist))]):
        # 计算图片在imlist中所对应的序号,即在所有图片中的序号
        im_id = i*batch_size + im_num
        # objs列表包含了本次处理图片中所有检测得到的方框所包含目标的名称。每个元素对应一个检测得到的方框所包含目标的名称
        # for x in output遍历output中的每一行(即一个方框的预测值)得到x，如果这个方
        # 框所在图片在所有图片中的序号等于本次处理图片的序号，则用classes[int(x[-1])
        # 找到这个方框包含目标类别在classes中对应的类的名字。
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        # 打印本次处理图片运行的时间，用一个batch的平均运行时间来表示。.3f就表示保留三位小数点的浮点
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
        # 输出本次处理图片所有检测到的目标的名字
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

    if CUDA:
        # 保证gpu和cpu同步
        torch.cuda.synchronize()


# Drawing bounding boxes on images
try:
    #  check whether there has been a single detection has been made or not
    # 最终返回的output维度为34x8,是所有图片中剩下的方框的属性，一行对应一个方框，属性为(ind,x1,y1,x2,y2,s,s_cls,index_cls)，
    # ind 是这个方框所属图片在所有测试图片中的序号，x1,y1是在网络输入图片(416x416)坐标系中，方框左上角的坐标；x2,y2是在网络输入
    # 图片(416x416)坐标系中，方框右下角的坐标。s是这个方框含有目标的得分，s_cls是这个方框中所含目标最有可能的类别的概率得分，
    # index_cls是s_cls对应的这个类别所对应的序号
    output
except NameError:
    print("No detections were made")
    # 当所有图片都有没检测到目标时，退出程序
    exit()

# 前面im_dim_list是一个二维tensor，一行的元素为(W,H,W,H)，对应一张图片原始的宽、高，且重复了一次。(W,H,W,H)主要是在后面计算
# x1,y1,x2,y2各自对应的缩放系数时好对应上。本例中im_dim_list维度为11x4.index_select()就是在im_dim_list中查找output中每行所
# 对应方框所在图片在所有图片中的序号对应im_dim_list中的那一行，最终得到的im_dim_list的行数应该与output的行数相同。本例中此时
# im_dim_list维度34x4
im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

# min(416/im_dim_list, 1)，im_dim_list中的每个元素去除以416，然后在得到的tensor中的第1维(每行)去找到最小的元素.torch.min()返回一个
# 有两个tensor元素的tuple，第一个元素就是找到最小的元素的结果，这里没有给定 keepdim=True的标记，所以得到的最小元素的tensor会比原来减小一维，
# 另一个是每个最小值在每行中对应的序号。torch.min(416/im_dim_list, 1)[0]得到长度为34的最小元素构成的tensor，通过view(-1, 1)
# 变成了维度为34x1的tensor。这个tensor，即scaling_factor的每个元素就对应一张图片缩放成416的时候所采用的缩放系数
# ！！！ Scaling_factor在进行计算的时候用的416，如果是其它的尺寸，这里不应该固定为416，在开始检测时所用的缩放系数就是用的
# min(w/img_w, h/img_h)
scaling_factor = torch.min(416/im_dim_list, 1)[0].view(-1, 1)

# 将相对于输入网络图片(416x416)的方框坐标(x1,y1,x2,y2)变换成相对于原图按照纵横比不变进行缩放后的区域的坐标。看word文档注释
output[:, [1, 3]] -= (inp_dim - scaling_factor*im_dim_list[:, 0].view(-1, 1))/2
output[:, [2, 4]] -= (inp_dim - scaling_factor*im_dim_list[:, 1].view(-1, 1))/2

# 将方框坐标(x1,y1,x2,y2)映射到原始图片尺寸上，直接除以缩放系数即可。output[:,1:5]维度为34x4，scaling_factor维度是34x1.
# 相除时会利用广播性质将scaling_factor扩展为34x4的tensor
output[:, 1:5] /= scaling_factor

# 如果映射回原始图片中的坐标超过了原始图片的区域，则x1,x2限定在[0,img_w]内，img_w为原始图片的宽度。如果x1,x2小于0.0，
# 令x1,x2为0.0，如果x1,x2大于原始图片宽度，令x1,x2大小为图片的宽度。同理，y1,y2限定在0,img_h]内，img_h为原始图片的高度。clamp()
# 函数就是将第一个输入对数的值限定在后面两个数字的区间
for i in range(output.shape[0]):
    output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])
    
# output_recast为将输出结果经过去掉低分，NMS以及映射回原图片等操作后的时间。output_recast - start_det_loop为从开始检测到所有检测
# 结果映射回原图为止的时间。最后输出output_recast - start_det_loop想计算的只是检测的时间，不包含去掉低分，NMS以及映射回原图等操作的时间
# output_recast想计算的应该是从得到网络的输出结果output，然后开始进行去掉低分，NMS以及映射回原图片等操作的时间。
# output_recast = time.time()应该放在 try: output  except NameError:...这个结果后才对
output_recast = time.time()
# 开始载入颜色文件的时间
class_load = time.time()
# 读入包含100个颜色的文件pallete，里面是100个三元组序列
colors = pkl.load(open("pallete", "rb"))

# 开始画方框的文字的时间
draw = time.time()


# x为映射到原始图片中一个方框的属性(ind,x1,y1,x2,y2,s,s_cls,index_cls)，results列表保存了所有测试图片，一个元素对应一张图片
def write(x, results):
    # c1为方框左上角坐标x1,y1
    c1 = tuple(x[1:3].int())
    # c2为方框右下角坐标x2,y2
    c2 = tuple(x[3:5].int())
    # 在results中找到x方框所对应的图片，x[0]为方框所在图片在所有测试图片中的序号
    img = results[int(x[0])]
    # cls为方框包含目标的类别在所有类中的序号
    cls = int(x[-1])
    # 随机选择一个颜色，用于后面画方框的颜色
    color = random.choice(colors)

    # label为这个框所含目标名字的字符串
    label = "{0}".format(classes[cls])
    # 在图片上画出(x1,y1,x2,y2)矩形，即我们检测到的目标方框
    cv2.rectangle(img, c1, c2, color, 3)
    # 得到一个包含目标名字字符的方框的宽高
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    # 得到包含目标名字的方框右下角坐标c2，这里在x,y方向上分别加了3、4个像素
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    # 在图片上画一个实心方框，我们将在方框内放置目标名字
    cv2.rectangle(img, c1, c2, color, -1)
    # 在图片上写文字，(c1[0], c1[1] + t_size[1] + 4)为字符串的左下角坐标
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 2);
    return img


# 将每个框在对应图片上画出来，同时写上方框所含目标名字。map函数将output传递给map()中参数是函数的那个参数，每次传递
# 一行。而lambda中x就是output中的一行，维度为1x8。loaded_ims列表保存了所有图片数组,一个元素对应一张图片。
list(map(lambda x: write(x, loaded_ims), output))

# 将带有方框的每张测试图片重新命名。pd.Series(imlist)返回一个series对象，
test = pd.Series(imlist)
det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det, x.split("\\")[-1]))

# 保存标注了方框和目标名字的图片。det_names对应所有测试图片的保存路径，loaded_ims对应所有标注了方框和目标名字的图片数组
list(map(cv2.imwrite, det_names, loaded_ims))


end = time.time()

print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
# 读取所有图片路径的时间
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
# 读入所有图片，并将图片按照batch size分成不同batch的时间
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
# 从开始检测到所有检测结果映射回原图为止的时间。这里的output_recast时间似乎有错，output_recast - start_det_loop想计算的应该只是检测
# 的时间，不包含去掉低分，NMS以及映射回原图等操作的时间。前面output_recast变量的位置错了
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) + " images)", output_recast - start_det_loop))
# 这里计算的应该是包含去掉低分，NMS以及映射回原图等操作的时间，但是错误地将class_load，output_recast位置放在了一直，结果为0
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
# 画框和文字的时间
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
# 从开始载入图片到所有结果处理完成，平均每张图片所消耗时间
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
print("----------------------------------------------------------")


torch.cuda.empty_cache()

