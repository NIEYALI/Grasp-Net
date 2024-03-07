import glob
import os
import numpy as np
from imageio import imsave
import argparse
from utils.dataset_processing.image import DepthImage


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Generate depth images from Cornell PCD files.')
    # parser.add_argument('--path',type=str, default='C:/Users/dell/Desktop/philchen/myRobot/robot_grasp/cornell_dataset/depth_image', help='Path to Cornell Grasping Dataset')
    # args = parser.parse_args()
    path = 'C:/Users/dell/Desktop/philchen/myRobot/robot_grasp/cornell_dataset/depth_image/'
    pcds = glob.glob(os.path.join(path, 'pcd*[0-9].txt'))
    pcds.sort()

    for pcd in pcds:
        di = DepthImage.from_pcd(pcd, (480, 640))
        di.inpaint()

        of_name = pcd.replace('.txt', 'd.tiff')
        print(of_name)
        imsave(of_name, di.img.astype(np.float32))