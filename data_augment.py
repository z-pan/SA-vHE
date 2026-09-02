# Data augmentation to increase the number of training data

import os

import cv2
import numpy as np
import tifffile
import PIL.Image as Image

data_type = "AF - train"
#data_type = "AF - test"
#data_type = "HE - train"

dataset_name = "set30-32"

if data_type == "AF - train":
    input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/train_data/" + dataset_name + "/01_AF_png"
    output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/train_data/" + dataset_name + "/01_AF_png_aug"
    if_convert_to_grayscale = False
    if_smooth = False
    if_enhance_contrast = False

elif data_type == "HE - train":
    input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/train_data/" + dataset_name + "/01_HE_png"
    output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/train_data/" + dataset_name + "/02_HE_png_aug"
    if_convert_to_grayscale = False
    if_smooth = False
    if_enhance_contrast = False

elif data_type == "AF - test":
    input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/test_data/" + dataset_name + "/original_AF_RGB_tif"
    output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/test_data/" + dataset_name + "/original_AF_png_inverted"
    if_convert_to_grayscale = True
    if_smooth = True
    if_enhance_contrast = True

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_im_name_prefix_count = 1

input_im_list = [im_name for im_name in os.listdir(input_dir) if im_name.endswith('.png') or im_name.endswith('.tif')]
input_im_list.sort()

for input_im_name in input_im_list:
    input_im = Image.open(os.path.join(input_dir, input_im_name))

    # convert to grayscale
    if if_convert_to_grayscale:
        print("Converting to grayscale...")
        input_im = input_im.convert('L')
    
    # image smoothing with opencv
    if if_smooth:
        print("Smoothing the image...")
        input_im = cv2.GaussianBlur(np.array(input_im), (3, 3), 0)
        input_im = Image.fromarray(input_im)
    
    # enhance contrast with CLAHE
    if if_enhance_contrast:
        print("Enhancing the contrast...")
        input_im = cv2.equalizeHist(np.array(input_im))
        input_im = Image.fromarray(input_im)
    
    # divide into smaller images according to the specified input and output x and y, 
    # with input image size [input_x, input_y] and output images of size [output_x, output_y]
    print("Data augmentation...")
    input_im_horz_flip = input_im.transpose(Image.FLIP_LEFT_RIGHT)
    output_im_horz_flip_name = input_im_name[:-4] + '_horz_flip.png'
    input_im_horz_flip.save(os.path.join(output_dir, output_im_horz_flip_name))

    input_im_vert_flip = input_im.transpose(Image.FLIP_TOP_BOTTOM)
    output_im_vert_flip_name = input_im_name[:-4] + '_vert_flip.png'
    input_im_vert_flip.save(os.path.join(output_dir, output_im_vert_flip_name))

    input_im_rot_90 = input_im.transpose(Image.ROTATE_90)
    output_im_rot_90_name = input_im_name[:-4] + '_rot_90.png'
    input_im_rot_90.save(os.path.join(output_dir, output_im_rot_90_name))

    input_im_rot_180 = input_im.transpose(Image.ROTATE_180)
    output_im_rot_180_name = input_im_name[:-4] + '_rot_180.png'
    input_im_rot_180.save(os.path.join(output_dir, output_im_rot_180_name))

    input_im_rot_270 = input_im.transpose(Image.ROTATE_270)
    output_im_rot_270_name = input_im_name[:-4] + '_rot_270.png'
    input_im_rot_270.save(os.path.join(output_dir, output_im_rot_270_name))

    output_im_name = input_im_name[:-4] + '.png'
    input_im.save(os.path.join(output_dir, output_im_name))

    output_im_name_prefix_count += 1