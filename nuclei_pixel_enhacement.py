# increase the pixel intensity of the nuclei regions, which is indicated by the areas in the mask image with pixel intensity value 255.

import os

import cv2
import numpy as np
import tifffile
import PIL.Image as Image

dataset_name = "nuc_enhance_test2"

#dataset_type = "AF - train"
dataset_type = "AF - test"

if dataset_type == "AF - train":
    input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/train_data/" + dataset_name + "/01_AF_png_aug"
    output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/train_data/" + dataset_name + "/01_AF_png_aug_enhanced"
    mask_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/train_data/" + dataset_name + "/01_AF_png_aug_mask"

elif dataset_type == "AF - test":
    input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/" + dataset_name + "/og"
    output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/" + dataset_name + "/nuc_enhanced"
    mask_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/" + dataset_name + "/mask"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

input_im_list = [im_name for im_name in os.listdir(input_dir) if im_name.endswith('.png')]
mask_im_list = [im_name for im_name in os.listdir(mask_dir) if im_name.endswith('.png') or im_name.endswith('.tiff') or im_name.endswith('.tif')]
input_im_list.sort()
mask_im_list.sort()

for input_im_name, mask_im_name in zip(input_im_list, mask_im_list):
    input_im = Image.open(os.path.join(input_dir, input_im_name))
    if mask_im_name.endswith('.tiff') or mask_im_name.endswith('.tif'):
        mask_im = tifffile.imread(os.path.join(mask_dir, mask_im_name))
    elif mask_im_name.endswith('.png'):
        mask_im = Image.open(os.path.join(mask_dir, mask_im_name))

    input_im = np.array(input_im)
    mask_im = np.array(mask_im)
    
    # invert mask_im using opencv
    mask_im = cv2.bitwise_not(mask_im)
    # smooth the edges of mask_im
    mask_im = cv2.GaussianBlur(mask_im, (11, 11), 0)
    # reduce the global intensity by 128
    mask_im = np.where(mask_im > 128, mask_im - 128, 0)
    # save mask_im in png format
    mask_im = Image.fromarray(mask_im)
    #mask_im.save(os.path.join(output_dir, mask_im_name.replace('.tiff', '_inter.png')))
    
    # decrease all pixel intensity without overflowing
    #input_im = np.where(input_im > 50, input_im - 30, 0)

    # increase the pixel intensity of the nuclei regions
    #input_im = np.where(mask_im == 255, input_im + 120, input_im)
    input_im = input_im + mask_im

    output_im = Image.fromarray(input_im)
    output_im.save(os.path.join(output_dir, input_im_name))