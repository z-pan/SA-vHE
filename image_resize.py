# resize a given image

import cv2
import numpy as np
import os
import sys
import tifffile
from PIL import Image

def resize_image(input_im_path, output_im_path, new_size):
    input_im = Image.open(input_im_path)
    input_im = input_im.resize(new_size)
    input_im.save(output_im_path)

input_im_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/test_data/240629_01/075_12_1.2_RGB_saliency_loss_output_160/full_im"
output_im_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/test_data/240629_01/075_12_1.2_RGB_saliency_loss_output_160/full_im"
if not os.path.exists(output_im_dir):
    os.makedirs(output_im_dir)

input_im_list = [im_name for im_name in os.listdir(input_im_dir) if im_name.endswith('.png')]
input_im_list.sort()
input_im_list = ["GT.png"]


new_size = (2560, 2560)

for input_im_name in input_im_list:
    input_im_path = os.path.join(input_im_dir, input_im_name)

    output_im_name = input_im_name.replace(".png", "_resized.png")
    output_im_path = os.path.join(output_im_dir, output_im_name)
    resize_image(input_im_path, output_im_path, new_size)
    print("Resized: ", input_im_name)