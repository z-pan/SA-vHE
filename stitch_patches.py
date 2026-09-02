# Crop the test images into tiles with overlap
import os

import numpy as np
import cv2
from PIL import Image
import tifffile as tiff

data_type = "AF - test"

dataset_name = "set37_human_ovarian"
im_folder_name = "Line11/im_0006"
netout_folder_name = "/netout_gray_saliency_A50_B210"
full_im_folder_name = "/full_im_gray_saliency_A50_B210"

model_output_name = "095_07_1.3.3_output"

is_grayscale = False

input_im_x = 512
input_im_y = 512
overlap = 0
num_input_x = 5
num_input_y = 5

#input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/" + dataset_name + "/" + im_folder_name + netout_folder_name
#output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/" + dataset_name + "/" + im_folder_name + full_im_folder_name
input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/250328_bulk-tissue/test_patch_03"
output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/250328_bulk-tissue/test_patch_03"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

input_im_list = [im_name for im_name in os.listdir(input_dir) if (im_name.endswith('.png') or im_name.endswith('.tif'))]
input_im_list.sort()
#input_im_list.reverse()

print("Number of input images: ", len(input_im_list))

full_im_x = int(input_im_x * (num_input_x))
full_im_y = int(input_im_y * (num_input_y))
if is_grayscale:
    full_im = np.zeros((full_im_x, full_im_y), dtype=np.uint8)
else:
    full_im = np.zeros((full_im_x, full_im_y, 3), dtype=np.uint8)

# iterate through the full image, and fill in the tile with each input image
for i in range(num_input_x):
    for j in range(num_input_y):
        input_im_name = input_im_list[num_input_x*j + i]
        input_im_path = os.path.join(input_dir, input_im_name)
        if ".png" in input_im_name:
            input_im = cv2.imread(input_im_path, cv2.IMREAD_UNCHANGED)
            # convert from BGR to RGB
            input_im = cv2.cvtColor(input_im, cv2.COLOR_BGR2RGB)
        else:
            input_im = tiff.imread(input_im_path)
            # squeeze the tiff image
            input_im = np.squeeze(input_im)
        
        print("input_im shape: ", input_im.shape)
        print("Processing: ", input_im_name)
        print("location at full_im: ", i*input_im_x, (i+1)*input_im_x, j*input_im_y, (j+1)*input_im_y)
        full_im[i*input_im_x:(i+1)*input_im_x, j*input_im_y:(j+1)*input_im_y, :] = input_im

# save the full image with tifffile
full_im_name = model_output_name + '_full.png'
tiff.imwrite(os.path.join(output_dir, full_im_name), full_im)
