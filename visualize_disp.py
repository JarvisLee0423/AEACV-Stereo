import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os
from PIL import Image
import cv2
import argparse


def visual_depth(img_path, out_path):
    img_list = sorted(glob(os.path.join(img_path, '*.png')))
    # img_list = ['/mnt/bn/shin-depth/Adaptable-Cost-Volume-for-Stereo-Matching/sub_kitti/test/000067_10.png',
    # '/mnt/bn/shin-depth/A_dap-igev/Adaptable-Cost-Volume-for-Stereo-Matching/sub_kitti/test/000067_10.png']
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # import pdb; pdb.set_trace()
    for img_pth in img_list:
        depth = np.array(Image.open(img_pth))
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255.0/depth.max(), beta=0), cv2.COLORMAP_JET)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=1/108., beta=0), cv2.COLORMAP_JET)
        name = os.path.join(out_path, os.path.basename(img_pth))
        cv2.imwrite(name, depth_colormap)
        old_depth = depth.copy()


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--image_path', help='Path of your depth images')
    parse.add_argument('--out_path', help='Path for the storation of visualization results')
    args = parse.parse_args()
    
    visual_depth(args.image_path, args.out_path)
