# -*- coding: UTF-8 -*-
"""
生成ggcnn抓取检测标签
抓取点、抓取角、抓取宽度 合成一个mat  (3, 480, 640)
抓取点: 1通道
抓取角：1通道。单位弧度
抓取宽度：1通道。单位像素
"""

import os
import glob
import scipy.io as scio
import numpy as np

label_path = 'label'
save_path = label_path


def run():
    label_files = glob.glob(label_path + '/*Label.txt')

    max_w = 0
    for label_file in label_files:
        print('processing ', label_file)
        label_mat = np.zeros((3, 480, 640), dtype=float)

        with open(label_file) as f:
            labels = f.readlines()
            for label in labels:
                label = label.strip().split(' ')
                row = int(float(label[0]))
                col = int(float(label[1]))
                label_mat[0, row, col] = 1. # 设置抓取点
                
                # 设置抓取角
                if len(label) == 3: # 无约束抓取
                    label_mat[1, row, col] = 0.0
                else:   # 单向抓取 或 对称抓取
                    label_mat[1, row, col] = float(label[2])

                label_mat[2, row, col] = float(label[-1]) / 200.  # 设置抓取宽度
                if float(label[-1]) > max_w:
                    max_w = float(label[-1])

            # 保存 mat
            save_name = os.path.join(save_path, os.path.basename(label_file).replace('Label.txt', 'grasp.mat'))
            scio.savemat(save_name, {'A': label_mat})

    print('max_w = ', max_w)


if __name__ == '__main__':
    run()