import numpy as np
import sympy as sp
import cv2
import imageio.v2 as imageio
from scipy.io import loadmat, savemat

#
# # img = imageio.imread('/home/e/桌面/image_09996.png')
# img = imageio.imread('/home/e/桌面/DMDTC-main/Code/prior_learning/config/deblurring/datasets_mask/val_GT/1.png')
#
# img = img / (img.max() + 5e-16)
# img = cv2.resize(img, (512, 512))
#
# fshift= np.fft.fft2(img)
# img=np.fft.fftshift(fshift)
#
# img=np.abs(img)
# # img=np.fft.ifftshift(img)
# img = cv2.resize(img, (512, 512))
#
# cv2.imwrite('/home/e/桌面/mnist_512_qiangdu/image_6.png', img)


# weight = loadmat('/home/e/桌面/CDI_2/FDTC/FDTC_Stage2/weight1.mat')['weight']
# for i in range(len(weight)):
#     for j in range(len(weight)):
#         print(weight[i][j],end=' ')
#     print()


# img = imageio.imread('/home/e/桌面/CDI_2/FDTC/FDTC_Stage2/measurement/result_0.png')
#
#
# def weight(l=64, r=0.02, p=0.05):
#     center = (l - 1) / 2.0
#     w = np.zeros((l, l), dtype=np.float32)
#     for i in range(l):
#         for j in range(l):
#             w[i][j] = (r * (i - center) ** 2 + r * (j - center) ** 2) ** p
#     return w
# weight=weight()
# savemat('/home/e/桌面/CDI_2/FDTC/FDTC_Stage2/weight3.mat',{'weight':weight})

# from PIL import Image
# import os
#
# def resize_png_images_in_folder(folder_path, new_size):
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             if file.endswith(".png"):
#                 file_path = os.path.join(root, file)
#                 img = Image.open(file_path)
#                 img = img.resize(new_size, Image.ANTIALIAS)
#                 img.save(file_path)
#
# # 使用示例
# dest_folder_path = '/home/e/桌面/DMDTC-main/Code/prior_learning/config/deblurring/Train_datasets/Spatial_domain/256X256/val_GTjiazaosheng'
# new_image_size = (128,128)
# resize_png_images_in_folder(dest_folder_path, new_image_size)


# import numpy as np
# import cv2
# import imageio.v2 as imageio
#
# # Load and normalize the image
# img = imageio.imread('/home/e/桌面/DMDTC-main/Code/prior_learning/config/deblurring/datasets_mask/val_GT/1.png')
# img = img / (img.max() + 5e-16)
#
# # Apply Fourier Transform
# fshift = np.fft.fft2(img)
# fshift = np.fft.fftshift(fshift)
#
# # Compute magnitude spectrum
# img = np.abs(fshift)
#
# # Convert magnitude spectrum to 8-bit format
# img = np.uint8(255 * img / np.max(img))
#
# # Ensure the saved image size matches the original
# img = cv2.resize(img, (512, 512))  # Adjust if necessary
#
# # Save the processed image
# cv2.imwrite('/home/e/桌面/mnist_512_qiangdu/image_7.png', img)

# img=imageio.imread('/home/e/桌面/mnist_512/image_1.png')
# print(img.shape)

import copy
import functools

import imageio
import torch
import numpy as np
import abc
import pywt

from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from models import utils as mutils
# from skimage.measure import compare_psnr,compare_ssim
import cv2
import os.path as osp
import matplotlib.pyplot as plt
import scipy.io as io
import math
from packages.ffdnet.models import FFDNet
import scipy.io as sio
from packages.ffdnet.test_ffdnet_ipol import ffdnet_vdenoiser
from tvdenoise import tvdenoise
import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.io import loadmat
import argparse
import logging
import os.path
import sys
import time
from collections import OrderedDict
import torchvision.utils as tvutils

import numpy as np

combinations = [
    [0.1, 0.7], [0.1, 0.8], [0.1, 0.9], [0.1, 1],
    [0.2, 0.1], [0.2, 0.3], [0.2, 0.4], [0.2, 0.5], [0.2, 0.6], [0.2, 0.7], [0.2, 0.8], [0.2, 0.9], [0.2, 1],
    [0.3, 0.1], [0.3, 0.2], [0.3, 0.4], [0.3, 0.5], [0.3, 0.6], [0.3, 0.7], [0.3, 0.8], [0.3, 0.9], [0.3, 1],
    [0.4, 0.6], [0.4, 0.7], [0.4, 0.8], [0.4, 0.9], [0.4, 1],
    [0.5, 0.1], [0.5, 0.2], [0.5, 0.3], [0.5, 0.4], [0.5, 0.6], [0.5, 0.7], [0.5, 0.8], [0.5, 0.9], [0.5, 1],
    [0.6, 0.1], [0.6, 0.2], [0.6, 0.3], [0.6, 0.4], [0.6, 0.5], [0.6, 0.7], [0.6, 0.8], [0.6, 0.9], [0.6, 1]
    # [0.7, 0.1], [0.7, 0.2], [0.7, 0.3], [0.7, 0.4], [0.7, 0.5], [0.7, 0.6], [0.7, 0.8], [0.7, 0.9], [0.7, 1],
    # [0.8, 0.1], [0.8, 0.2], [0.8, 0.3], [0.8, 0.4], [0.8, 0.5], [0.8, 0.6], [0.8, 0.7], [0.8, 0.9], [0.8, 1],
    # [0.9, 0.1], [0.9, 0.2], [0.9, 0.3], [0.9, 0.4], [0.9, 0.5], [0.9, 0.6], [0.9, 0.7], [0.9, 0.8], [0.9, 1],
    # [1, 0.1], [1, 0.2], [1, 0.3], [1, 0.4], [1, 0.5], [1, 0.6], [1, 0.7], [1, 0.8], [1, 0.9]
]

img_a = imageio.imread(r'/home/e/桌面/CDI_3_2/FDTC/FDTC_Stage2/save_imges_test_result/w_a=0.1,w_b=0.2/GT_0.png')
img_b = imageio.imread(r'/home/e/桌面/CDI_3_2/FDTC/FDTC_Stage2/save_imges_test_result/w_a=0.1,w_b=0.2/GT_5.png')
img_c = imageio.imread(r'/home/e/桌面/CDI_3_2/FDTC/FDTC_Stage2/save_imges_test_result/w_a=0.1,w_b=0.2/GT_9.png')

l1 = [img_a, img_b, img_c]
l1_name = ["GT_0", "GT_5", "GT_9"]
l2_name = ["LQ_5", "LQ_0", "LQ_9"]

for iii in range(len(combinations)):
    weight_a = combinations[iii][0]
    weight_b = combinations[iii][1]
    dir = '/FDTC/FDTC_Stage2/数据汇总/save_imges_test_result'
    dir += '/w_a={},w_b={}'.format(weight_a, weight_b)
    os.makedirs(dir)
    for jjj in range(len(l1)):
        cv2.imwrite(r'{}/{}.png'.format(dir, l1_name[jjj]), l1[jjj])

    dir_2 = '/FDTC/FDTC_Stage2/数据汇总/save_imges_test'
    dir_2 += '/w_a={},w_b={}'.format(weight_a, weight_b)
    for jjj in range(len(l1)):
        img = imageio.imread('{}/{}.png'.format(dir_2, jjj))
        cv2.imwrite(r'{}/{}.png'.format(dir, l2_name[jjj]), img)
