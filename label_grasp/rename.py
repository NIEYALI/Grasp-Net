# -*- coding: utf-8 -*-
"""
将拍摄的图像文件名修改为pcd...r.png
"""

import os
import glob
import shutil 

path = './test'

rgb_files = glob.glob(os.path.join(path, '*rgb.png'))
rgb_files.sort()

i = 0
for rgb_file in rgb_files:
    depth_file = rgb_file.replace('_rgb', '_dep')

    new_rgb_name = 'pcd2{:03d}r.png'.format(i)
    new_rgb_file = os.path.join(path, new_rgb_name)
    shutil.copyfile(rgb_file, new_rgb_file)

    new_dep_name = 'pcd2{:03d}d.png'.format(i)
    new_dep_file = os.path.join(path, new_dep_name)
    shutil.copyfile(depth_file, new_dep_file)

    i += 1

    print(new_rgb_file)

