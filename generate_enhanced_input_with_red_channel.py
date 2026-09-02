# increase the pixel intensity of the nuclei regions, which is indicated by the areas in the mask image with pixel intensity value 255.

import os

import cv2
import numpy as np
import tifffile
import PIL.Image as Image

dataset_name = "240629_01"

#dataset_type = "AF - train"
dataset_type = "AF - test"

input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/test_data/" + dataset_name + "/01_AF_tile"
output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/test_data/" + dataset_name + "/testA"
mask_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/test_data/" + dataset_name + "/01_AF_mask_tile"
seg_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/test_data/" + dataset_name + "/01_AF_seg_tile"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(seg_dir):
    os.makedirs(seg_dir)

input_im_list = [im_name for im_name in os.listdir(input_dir) if im_name.endswith('.png')]
mask_im_list = [im_name for im_name in os.listdir(mask_dir) if im_name.endswith('.png') or im_name.endswith('.tiff') or im_name.endswith('.tif')]
input_im_list.sort()
mask_im_list.sort()

for input_im_name, mask_im_name in zip(input_im_list, mask_im_list):
    input_im = Image.open(os.path.join(input_dir, input_im_name))
    if mask_im_name.endswith('.tiff') or mask_im_name.endswith('.tif'):
        mask_im = tifffile.imread(os.path.join(mask_dir, mask_im_name))
    elif mask_im_name.endswith('.png'):
        # open as 8-bit image
        print("Reading png image...")
        mask_im = Image.open(os.path.join(mask_dir, mask_im_name)).convert('L')

    input_im = np.array(input_im)
    mask_im = np.array(mask_im)
    print("mask_im max: ", np.max(mask_im)) # 255
    
    # decrease all pixel intensity without overflowing
    input_im = np.where(input_im > 30, input_im - 20, 0)
    # increase the pixel intensity in input_im where mask_im is 255, and decrease the pixel intensity where mask_im is 0, while consider the overflow
    for i in range(input_im.shape[0]):
        for j in range(input_im.shape[1]):
            if mask_im[i][j] == 255:
                if input_im[i][j] + 150 > 255:
                    input_im[i][j] = 255
                else:
                    input_im[i][j] += 150
    
    output_im = Image.fromarray(input_im)
    output_im.save(os.path.join(output_dir, input_im_name))
    
    # save as 8-bit image
    mask_im = Image.fromarray(mask_im)
    mask_im.save(os.path.join(seg_dir, mask_im_name))