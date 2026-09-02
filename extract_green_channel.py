# Extract the green channel from the RGB TPAF images

import os
import cv2
import numpy as np
import tifffile as tiff

input_path = "C:/Users/zpanp/projects/UTOM-master/datasets/train_data/250407_bulk/00_og_RGB"
output_path = "C:/Users/zpanp/projects/UTOM-master/datasets/train_data/250407_bulk/00_og_green"
if not os.path.exists(output_path):
    os.makedirs(output_path)

input_im_list = [im_name for im_name in os.listdir(input_path) if (im_name.endswith('.png') or im_name.endswith('.tif') or im_name.endswith('.tiff'))]
input_im_list.sort()
print("Number of input images: ", len(input_im_list))

for input_im_name in input_im_list:
    input_im_path = os.path.join(input_path, input_im_name)
    input_im = tiff.imread(input_im_path)

    # Extract the green channel
    green_channel = input_im[:, :, 1]

    # Save the green channel image
    output_im_name = f"{os.path.splitext(input_im_name)[0]}_green.tif"
    output_im_path = os.path.join(output_path, output_im_name)
    tiff.imwrite(output_im_path, green_channel)
    print(f"Processed: {output_im_name}")
