# Reverse color deconvolution 
# Input are hematoxylin and eosin channels, output is the original RGB H&E image.

import os
import numpy as np
import cv2
import tifffile as tiff
from skimage.color import hed2rgb

input_H_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/250711_slides/06_results_vH_patches_background_remove"
input_E_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/250711_slides/05_results_vE_patches"
output_HE_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/250711_slides/06_results_vHE_patches_color_rev_deconv"
if not os.path.exists(output_HE_dir):
    os.makedirs(output_HE_dir)

input_H_list = [im_name for im_name in os.listdir(input_H_dir) if (im_name.endswith(".tif") or im_name.endswith(".tiff") or im_name.endswith(".png"))]
input_H_list.sort()
print("Number of input H channels: ", len(input_H_list))

input_E_list = [im_name for im_name in os.listdir(input_E_dir) if (im_name.endswith(".tif") or im_name.endswith(".tiff") or im_name.endswith(".png"))]
input_E_list.sort()
print("Number of input E channels: ", len(input_E_list))

assert len(input_H_list) == len(input_E_list), "Number of H and E channels must match."

for i in range(len(input_H_list)):
    input_H_path = os.path.join(input_H_dir, input_H_list[i])
    input_E_path = os.path.join(input_E_dir, input_E_list[i])

    # Load the H&E channels
    H_channel = cv2.imread(input_H_path, cv2.IMREAD_UNCHANGED)
    E_channel = cv2.imread(input_E_path, cv2.IMREAD_UNCHANGED)
    DAB_channel = np.zeros_like(H_channel, dtype=np.uint8)  # Assuming DAB channel is not used in this case

    # Stack the channels
    HE_stack = np.stack([H_channel, E_channel, DAB_channel], axis=-1)

    # Convert H&E to RGB
    RGB_image = hed2rgb(HE_stack)
    RGB_image_uint8 = (np.clip(RGB_image, 0, 1) * 255).astype(np.uint8)

    # Save the reconstructed RGB image
    output_RGB_path = os.path.join(output_HE_dir, input_H_list[i])
    cv2.imwrite(output_RGB_path, RGB_image_uint8)

    print(f"Processed: {input_H_list[i]}")