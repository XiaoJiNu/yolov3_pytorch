from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    
    # Intersection area 这里没有对inter_area为负的情况进行判断，后面计算出来的IOU就可能是负的
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou


# predict_transform()利用一个scale得到的feature map预测得到的每个anchor的属性(x,y,w,h,s,s_cls1,s_cls2...),其中x,y,w,h
# 是在网络输入图片坐标系下的值,s是方框含有目标的得分，s_cls1,s_cls_2等是方框所含目标对应每类的概率。输入的feature map(prediction变量)
# 维度为(batch_size, num_anchors*bbox_attrs, grid_size, grid_size)，类似于一个batch彩色图片BxCxHxW存储方式。参数见predict_transform()
# 里面的变量。
# 并且将结果的维度变换成(batch_size, grid_size*grid_size*num_anchors, 5+类别数量)的tensor，同时得到每个方框在网络输入图片
# (416x416)坐标系下的(x,y,w,h)以及方框含有目标的得分以及每个类的得分。
def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):

    batch_size = prediction.size(0)
    # stride表示的是整个网络的步长，等于图像原始尺寸与yolo层输入的feature mapr尺寸相除，因为输入图像是正方形，所以用高相除即可
    stride = inp_dim // prediction.size(2)
    # 格子的数量，416/32=13
    grid_size = inp_dim // stride
    # 一个方框属性个数，等于5+类别数量
    bbox_attrs = 5 + num_classes
    # anchors数量
    num_anchors = len(anchors)

    # 输入的prediction维度为(batch_size, num_anchors * bbox_attrs, grid_size, grid_size)，类似于一个batch彩色图片BxCxHxW
    # 存储方式，将它的维度变换成(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    # 将prediction维度转换成(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)。不看batch_size，
    # (grid_size*grid_size*num_anchors, bbox_attrs)相当于将所有anchor按行排列，即一行对应一个anchor属性，
    # 此时的属性仍然是feature map得到的值
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    # 此时的anchors是相对于最终的feature map的尺寸
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:,:,1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:,:,4] = torch.sigmoid(prediction[:, :, 4])
    
    # Add the center offsets
    # 这里生成了每个格子的左上角坐标，生成的坐标为grid x grid的二维数组，a，b分别对应这个二维矩阵的x,y坐标的数组，a,b的维度与grid维度一样
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    # 这里的x_y_offset对应的是最终的feature map中每个格子的左上角坐标，比如有13个格子，刚x_y_offset的坐标就对应为(0,0),(0,1)…(12,12)
    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    # ???????????????
    # 这里的x_y_offset对应的是最终的feature map中每个格子的左上角相对于一个feature map左上角的坐标，比如13个格子，x_y_offset的
    # 坐标就对应为(0,0),(0,1)…(12,12)。view(-1, 2)将tensor变成两列，unsqueeze(0)在0维上添加了一维。
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    # 前面经过sigmoid变换得到的x,y坐标是相对于一个格子的左上角坐标，这里加上每个
    prediction[:, :, :2] += x_y_offset

    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    # 这里的anchors本来是一个长度为6的list(三个anchors每个2个坐标)，然后在0维上(行)进行了grid_size*grid_size个复制，在1维(列)上
    # 一次复制(没有变化)，即对每个格子都等到三个anchor。Unsqueeze(0)的作用是在数组上添加一维，这里是在第0维上添加的。
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    # 对网络预测得到的矩形框的宽高的偏差值进行指数计算，然后乘以anchors里面对应的宽高(这里的anchors里面的宽高是对应最终的feature map尺寸)，
    # 得到目标的方框的宽高，这里得到的宽高是相对于在feature map的尺寸
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4])*anchors

    # 这里得到每个anchor中每个类别的得分。将网络预测的每个得分用sigmoid()函数计算得到
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    # 将相对于最终feature map的方框坐标和尺寸映射回输入网络图片(416x416)，即将方框的坐标乘以网络的stride即可
    prediction[:, :, :4] *= stride
    
    return prediction


# write_results()首先将网络输出方框属性(x,y,w,h)转换为在网络输入图片(416x416)坐标系中，方框左上角与右下角坐标(x1,y1,x2,y2)，
# 以方便NMS操作。然后将方框含有目标得分低于阈值的方框去掉，提取得分最高的那个类的得分max_conf，同时返回这个类对应的序号max_conf_score
# 然后进行NMS操作。最终每个方框的属性为(ind,x1,y1,x2,y2,s,s_cls,index_cls)，ind 是这个方框所属图片在这个batch中的序号，x1,y1是
# 在网络输入图片(416x416)坐标系中，方框左上角的坐标；x2,y2是在网络输入图片(416x416)坐标系中，方框右下角的坐标。s是这个方框含有目标
# 的得分s_cls是这个方框中所含目标最有可能的类别的概率得分，index_cls是s_cls对应的这个类别所对应的序号
def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    # Object Confidence Thresholding

    # 这句语句是怎么操作的？对于含有目标的得分小于confidence的方框，它对应的含有目标的得分将变成0,即conf_mask中对应元素为0
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    # conf_mask中含有目标的得分小于confidence的方框所对应的含有目标的得分为0，根据numpy的广播原理，它会扩展成与prediction维度一样
    # 的tensor，所以含有目标的得分小于confidence的方框所有的属性都为0
    prediction = prediction*conf_mask

    # 创建 一个新的数组，大小与predicton的大小相同，这里是将以前每个矩形坐标与宽高(x,y,w,h)转换成矩形的左上角坐标(x1,y1)和右下角
    # 坐标(x2,y2)，得到(x1,y1,x2,y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]
    
    batch_size = prediction.size(0)

    write = False

    # 对每一张图片得分的预测值进行NMS操作，因为每张图片的目标数量不一样，所以有效得分的方框的数量不一样，没法将几张图片同时处理，所以
    # 遍历每张图片，将得分低于一定分数的去掉，对剩下的方框进行进行NMS
    for ind in range(batch_size):
        # image Tensor. image_pred 对应一张图片中所有方框的坐标(x1,y1,x2,y2)以及得分，是一个二维tensor 维度为10647x85
        image_pred = prediction[ind]

        # confidence threshholding

        # 返回每一行中所有类别的得分最高的那个类的得分max_conf，同时返回这个类对应的序号max_conf_score
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
        # 添加一个列的维度，max_conf变成二维tensor，尺寸为10647x1
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        # 将每个方框的(x1,y1,x2,y2,s)与得分最高的这个类的分数s_cls和对应类的序号index_cls在列维度上连接起来，即将10647x5,
        # 10647x1,10647x1三个tensor 在列维度进行concatenate操作，得到一个10647x7的tensor,(x1,y1,x2,y2,s,s_cls,index_cls)。
        # 这里的s代表的是方框有目标的分数
        image_pred = torch.cat(seq, 1)

        # 这里返回每个方框含有目标的得分是非0元素的下标。image_pred[:,4]是长度为10647的一维tensor,假设有15个框含有目标的得分非0，
        # 返回15x1的tensor non_zero_ind
        non_zero_ind = (torch.nonzero(image_pred[:, 4]))
        try:
            # non_zero_ind.squeeze()将15x1的non_zero_ind去掉维度为1的维度，变成长度为15的一维tensor，相当于一个列向量，
            # image_pred[non_zero_ind.squeeze(),:]是在image_pred中找到non_zero_ind中非0目标得分的行的所有元素(image_pred维度
            # 是10647x7，找到其中的15行)， 再用view(-1,7)将它变为15x7的tensor，为什么用view()？前面得到的tensor不是15x7吗
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        except:
            continue

        # 当没有检测到时目标时，进行下一次循环
        if image_pred_.shape[0] == 0:
            continue
  
        # Get the various classes detected in the imageimage_   pred_[:,-1]是一个15x7的tensor，最后一列保存的是每个框里面物体的
        # 类别，-1表示取最后一列。用unique()除去重复的元素，即一类只留下一个元素，这里最后只剩下了3个元素，即只有3类物体。
        img_classes = unique(image_pred_[:, -1])  # -1 index holds the class index

        for cls in img_classes:
            # perform NMS

            # get the detections with one particular class
            # 本句是将image_pred_中属于cls类的预测值保持不变，其余的全部变成0。image_pred_[:,-1] == cls，返回一个与image_pred_
            # 行数一样的一维tensor，这里长度为15.当image_pred_中的最后一个元素(物体类别)等于第cls类时，返回的tensor对应元素为1，
            # 否则为0. 它与image_pred_相乘时，先扩展为15x7的tensor(似乎这里还没有变成15x7的tensor)，为0元素一行全部为0，再与
            # image_pred_相乘，属于cls这类的方框对应预测元素不变，其它类的为0.unsqueeze(1)添加了列这一维，变成15x7的二维tensor。
            cls_mask = image_pred_*(image_pred_[:, -1] == cls).float().unsqueeze(1)
            # cls_mask[:, -2]为cls_mask倒数第二列，为这个类别的得分。将属于cls类的元素的下标序号提取出来，再变成一维tensor
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            # 找到image_pred_中对应cls类的所有方框的预测值，并转换为二维张量。这里这4x7
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)
            
            # sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending = True )[1]

            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)   # Number of detections
            
            for i in range(idx):
                # Get the IOUs of all boxes that come after the one we are looking at
                # in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
            
                except IndexError:
                    break
            
                # Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask       
            
                # Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                # 得到含有目标的得分非0的方框的预测值(x1, y1, x2, y2, s, s_class, index_cls)，为1x7的tensor
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            # image_pred_class.new(image_pred_class.size(0), 1)中，new()创建了一个和image_pred_class类型相同的tensor，tensor
            # 行数等于cls这个类别所有的方框经过NMS剩下的方框的个数，即image_pred_class的行数，列数为1.再将生成的这个tensor所有元素
            # 赋值为这些方框所属图片对应于batch中的序号ind(一个batch有多张图片同时测试)，用fill_(ind)实现
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)      # Repeat the batch_id for as many detections of the class cls in the image

            seq = batch_ind, image_pred_class

            if not write:
                # 将batch_ind, image_pred_class在列维度上进行连接，image_pred_class每一行存储的是(x1,y1,x2,y2,s,s_cls,index_cls)，
                # 现在在第一列增加了一个代表这个行对应方框所属图片在一个batch中的序号ind
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    try:
        # 最终返回的output是一个batch中所有图片中剩下的方框的属性，一行对应一个方框，属性为(ind,x1,y1,x2,y2,s,s_cls,index_cls)，
        # ind 是这个方框所属图片在这个batch中的序号，x1,y1是在网络输入图片(416x416)坐标系中，方框左上角的坐标；x2,y2是在网络输入
        # 图片(416x416)坐标系中，方框右下角的坐标。s是这个方框含有目标的得分s_cls是这个方框中所含目标最有可能的类别的概率得分，index_cls
        # 是s_cls对应的这个类别所对应的序号
        return output
    except:
        # 如果所有的图片都没有检测到方框，则在前面不会进行NMS等操作，不会生成output，此时将在except中返回0
        return 0


# lteerbox_image()将图片按照纵横比进行缩放，将空白部分用(128,128,128)填充
def ltterbox_image(img, inp_dim):
    ''' resize image with unchanged aspect ratio using padding '''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    # 取min(w/img_w, h/img_h)这个比例来缩放，缩放后的尺寸为new_w, new_h,即将其中一边缩放后正好对应需要的尺寸，
    # 另一边的尺寸缩放后还没有填充满.
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    # 将图片按照纵横比不变来缩放为new_w x new_h，768 x 576的图片缩放成416x312.
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 生成一个我们最终需要的图片尺寸h x w x 3的array,这里生成416x416x3的array,每个元素值为128
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    # 将w x h x 3的array中对应new_w x new_h x 3的部分(这两个部分的中心应该对齐)赋值为刚刚由原图缩放得到的数组,得到最终缩放后图片
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h, (w-new_w)//2:(w-new_w)//2 + new_w, :] = resized_image
    
    return canvas


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """
    # lteerbox_image()将图片按照纵横比进行缩放，将空白部分用(128,128,128)填充
    img = (ltterbox_image(img, (inp_dim, inp_dim)))
    # 将HxWxC格式转换为CxHxW格式，BGR格式怎么转换成RGB格式的???
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    # from_numpy(()将ndarray数据转换为tensor格式，div(255.0)将每个元素除以255.0，进行归一化，unsqueeze(0)，在0维上添加了一维，
    # 从3x416x416变成1x3x416x416，多出来的一维表示batch。这里就将图片变成了BxCxHxW的pytorch格式
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names







