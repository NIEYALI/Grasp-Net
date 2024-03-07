# -*- coding: utf-8 -*-
"""
@ Time ： 2020/3/2 11:33
@ Auth ： wangdx
@ File ：affga.py
@ IDE ：PyCharm
@ Function : 
"""

import cv2
import os
import torch
import time
from skimage.feature import peak_local_max
import numpy as np
from models.common import post_process_output
from models.loss import get_pred

# 裁剪图片中心的320,320区域的图像
def input_rgb(img):
    """
    对图像进行裁剪，保留中间(320, 320)的图像
    :param file: rgb文件
    :return: 直接输入网络的tensor, 裁剪区域的左上角坐标
    """

    out_size = 320
    assert img.shape[0] >= out_size and img.shape[1] >= out_size, '输入的深度图必须大于等于(320, 320)'

    # 裁剪中间320*320作为输入
    crop_x1 = int((img.shape[1] - out_size) / 2)
    crop_y1 = int((img.shape[0] - out_size) / 2)
    crop_x2 = crop_x1 + out_size
    crop_y2 = crop_y1 + out_size
    # 裁剪的图片中心区域的320x320的patch
    crop_rgb = img[crop_y1:crop_y2, crop_x1:crop_x2, :]

    # 归一化，都变为0-1,再求均值，再减去均值
    rgb = crop_rgb.astype(np.float32) / 255.0
    rgb -= rgb.mean()

    # 调整顺序，和网络输入一致,ndarray，将通道channel放在最后，原本是（0,1,2）现在变为(2,0,1),通道channel到第一维度
    rgb = rgb.transpose((2, 0, 1))  # (320, 320, 3) -> (3, 320, 320)
    # np转tensor
    rgb = torch.from_numpy(np.expand_dims(rgb, 0).astype(np.float32))
    # 返回处理后的图像，和裁剪框的左上角点的位置
    return rgb, crop_x1, crop_y1

def arg_thresh(array, thresh):
    """
    获取array中大于thresh的二维索引
    :param array: 二维array
    :param thresh: float阈值
    :return: array shape=(n, 2)
    """
    print(array.shape)
    res = np.where(array > thresh)
    rows = np.reshape(res[0], (-1, 1))
    cols = np.reshape(res[1], (-1, 1))
    locs = np.hstack((rows, cols))
    for i in range(locs.shape[0]):
        for j in range(locs.shape[0])[i+1:]:
            if array[locs[i, 0], locs[i, 1]] < array[locs[j, 0], locs[j, 1]]:
                locs[[i, j], :] = locs[[j, i], :]

    return locs


class AFFGA:
    def __init__(self, model, device):
        self.t = 0
        self.num = 0
        # 加载模型
        print('>> loading AFFGA')
        self.device = device
        self.net = torch.load(model, map_location=torch.device(device))
        self.net.eval()
        print('>> load done')

    def fps(self):
        return 1.0 / (self.t / self.num)

    def predict(self, img, device, mode, thresh=0.5, peak_dist=3, angle_k=120):
        """
        预测抓取模型
        :param img: 输入图像 np.array (h, w, 3)
        :param thresh: 置信度阈值
        :param peak_dist: 置信度筛选峰值
        :param angle_k: 抓取角分类数
        :return:
            pred_grasps: list([row, col, angle, width])
            crop_x1
            crop_y1
        """
        # 首先裁剪出图片中心附近的大小的320x320的图像pacth,和图像左上角
        rgb, self.crop_x1, self.crop_y1 = input_rgb(img)

        t1 = time.time()
        # 预测生成出confidence region, angle, width
        self.able_out, self.angle_out, self.width_out = get_pred(self.net, rgb.to(device))
        t2 = time.time() - t1
        print("time use ",t2)
        # 后处理，预测输出图像的后处理
        able_pred, angle_pred, width_pred = post_process_output(self.able_out, self.angle_out, self.width_out)
        if mode == 'peak':
            # 置信度峰值 抓取点置信度大于0.5，发现最大峰的间隔2 * min_distance + 1
            pred_pts = peak_local_max(able_pred, min_distance=peak_dist, threshold_abs=thresh)
        elif mode == 'all':
            # 超过阈值的所有抓取点
            pred_pts = arg_thresh(able_pred, thresh=thresh)
        elif mode == 'max':
            # 置信度最大的点
            loc = np.argmax(able_pred) # 得到当前点在图像中的序号，为一个int数
            row = loc // able_pred.shape[0] # 整除图像的列数得到所在行
            col = loc % able_pred.shape[0]  # 取余图像的列数得到所在列
            pred_pts = np.array([[row, col]]) # 确定最后的抓取点
        else:
            raise ValueError

        # 绘制预测的抓取
        pred_grasps = []
        for idx in range(pred_pts.shape[0]):
            # 找到抓取点
            row, col = pred_pts[idx]
            # 当前点的抓取角度
            angle = angle_pred[row, col] / angle_k * 2 * np.pi  # 预测的抓取角弧度
            # 当前点的抓取宽度
            width = width_pred[row, col]
            row += self.crop_y1
            col += self.crop_x1
            # 将此抓取点位置，角度，宽度保存起来。
            pred_grasps.append([row, col, angle, width])

        self.t += t2
        self.num += 1
        # 返回抓取点的集合，裁剪框的左上角
        return pred_grasps, self.crop_x1, self.crop_y1

    def maps(self, img, device):
        """绘制最终的特征图"""
        # 抓取置信度
        rgb, self.crop_x1, self.crop_y1 = input_rgb(img)
        # 评估过程中不需要梯度下降
        self.net.eval()
        with torch.no_grad():
            self.able_out, self.angle_out, self.width_out = get_pred(self.net, rgb.to(device))
        # 维度压缩，将1的维度压缩
        able_map = self.able_out.detach().numpy().squeeze()
        able_featureMap = np.zeros((able_map.shape[0], able_map.shape[1], 3), dtype=np.float)
        """
        R: 0    ->    255
        G: 0    ->    0
        B: 255  ->    0
        """
        able_featureMap[:, :, 2] = able_map * 255.0
        able_featureMap[:, :, 1] = 0
        able_featureMap[:, :, 0] = able_map * -255.0 + 255.0
        able_featureMap = able_featureMap.astype(np.uint8)

        # 抓取角
        angle_map = self.angle_out.detach().numpy().squeeze()[0]
        angle_map = angle_map.copy() * (255.0 / angle_map.max())
        angle_featureMap = np.zeros((angle_map.shape[0], angle_map.shape[1], 3), dtype=np.float)
        angle_featureMap[:, :, 2] = 100
        angle_featureMap[:, :, 0] = 100
        angle_featureMap[:, :, 1] = angle_map * -1 + 255
        angle_featureMap = angle_featureMap.astype(np.uint8)

        # 抓取宽度
        # detach()从当前计算图中分离下来的，仍指向原变量的存放位置,只是requires_grad为false，不需要计算其梯度，不具有grad。
        width_map = self.width_out.detach().numpy().squeeze()
        width_featureMap = np.zeros((width_map.shape[0], width_map.shape[1], 3), dtype=np.float)
        width_featureMap[:, :, 2] = width_map * 800.0
        width_featureMap[:, :, 1] = 255
        width_featureMap[:, :, 0] = 255
        width_featureMap = width_featureMap.astype(np.uint8)

        return able_featureMap, angle_featureMap, width_featureMap
