# move images whose mean intensity is below a certain threshold from a directory to another directory

import os
import cv2
import numpy as np
import tifffile as tiff

#data_type = "AF - train"
#data_type = "AF - test"
data_type = "HE - train"

dataset_name = "set37"

if data_type == "AF - train":
    input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/train_data/" + dataset_name + "/01_AF_RGB_png"
    output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/train_data/" + dataset_name + "/01_AF_RGB_png_selected"
    mean_threshold = 6.5

elif data_type == "HE - train":
    input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/train_data/" + dataset_name + "/01_HE_png"
    output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/train_data/" + dataset_name + "/01_HE_png_selected"
    mean_threshold = 15.0

elif data_type == "AF - test":
    input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/test_data/" + dataset_name + "/testA"
    output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/test_data/" + dataset_name + "/testA_selected"
    mean_threshold = 6.5

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

input_im_list = [im_name for im_name in os.listdir(input_dir) if im_name.endswith('.png')]
input_im_list.sort()

for input_im_name in input_im_list:
    input_im_path = os.path.join(input_dir, input_im_name)
    input_im = cv2.imread(input_im_path, cv2.IMREAD_UNCHANGED)
    
    if data_type == "AF - train" or data_type == "AF - test":
        if np.mean(input_im) > mean_threshold:
            output_im_path = os.path.join(output_dir, input_im_name)
            
            # move the image
            os.rename(input_im_path, output_im_path)
            print("Moved: ", input_im_name)
            print("Mean intensity: ", np.mean(input_im))
    
    elif data_type == "HE - train" or data_type == "HE - test":
        if np.mean(input_im) < (255 - mean_threshold):
            output_im_path = os.path.join(output_dir, input_im_name)
            
            # move the image
            os.rename(input_im_path, output_im_path)
            print("Moved: ", input_im_name)
            print("Mean intensity: ", np.mean(input_im))
