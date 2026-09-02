# Convert RGB 3-channel images to 2-channel (red & green) images or 1-channel (grayscale) images.

import os
import cv2
import numpy as np
import tifffile as tiff
import PIL.Image as Image

#data_type = "AF - train"
data_type = "AF - test"
#data_type = "bulk tissue"

dataset_name = "250711_slides"
#dataset_name = "set37_human_ovarian/Line11"

output_num_channels = 1 # 1 for grayscale, 2 for red and green channels, 3 for RGB

if data_type == "AF - train":
    input_path = "C:/Users/zpanp/projects/UTOM-master/datasets/train_data/" + dataset_name + "/00_og_RGB"
    #output_path = "C:/Users/zpanp/projects/UTOM-master/datasets/train_data/" + dataset_name + "/trainA_RG"
    output_path = "C:/Users/zpanp/projects/UTOM-master/datasets/train_data/" + dataset_name + "/00_og_gray"
elif data_type == "AF - test":
    input_path = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/" + dataset_name + "/00_og_TPAF_RGB_full"
    output_path = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/" + dataset_name + "/01_og_TPAF_gray_full"
elif data_type == "bulk tissue":
    input_path = "C:/Users/zpanp/projects/UTOM-master/datasets/train_data/250407_bulk/00_og_RGB"
    output_path = "C:/Users/zpanp/projects/UTOM-master/datasets/train_data/250407_bulk/00_og_gray"

if not os.path.exists(output_path):
    os.makedirs(output_path)

input_im_list = [im_name for im_name in os.listdir(input_path) if (im_name.endswith('.png') or im_name.endswith('.tif') or im_name.endswith('.tiff'))]
input_im_list.sort()

print("Number of input images: ", len(input_im_list))

for input_im_name in input_im_list:
    
    if (input_im_name.endswith('.tif') or input_im_name.endswith('.tiff')): # for image in tiff format
        print("Reading tiff image...")
        input_im = tiff.imread(os.path.join(input_path, input_im_name))
        input_im = input_im.astype(np.uint8) # numpy array
        input_im = Image.fromarray(input_im)
    elif (input_im_name.endswith('.png')): # for image in png format
        print("Reading png image...")
        input_im = Image.open(os.path.join(input_path, input_im_name))

    if output_num_channels == 1:
        output_im = input_im.convert('L')
    elif output_num_channels == 2:
        # only keep the red and green channels
        red_channel, green_channel, blue_channel = input_im.split()
        # create a two channel image with red and green channels
        output_im = Image.merge("RG", (red_channel, green_channel))
    
    output_im_path = os.path.join(output_path, input_im_name)
    output_im.save(output_im_path)
    print("Processed: ", input_im_name)