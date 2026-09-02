# Enhance TPAF nuclei with instance mask

import os
import cv2
import numpy as np
import tifffile as tiff
from PIL import Image

input_suffix = ".png"
mask_suffix = ".png"
nuclei_gain = 1.5
other_gain = 0.8

nuc_hi_channel = "RG" # B/G/R for single channel only, RG for both red and green channels, RGB for all channels

input_image_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/250529_slides/01_og_RGB_patches"
nuclei_mask_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/250529_slides/04_gray_local_CLAHE_patches_nuc_mask"
output_image_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/250529_slides/02_RGB_RG_nuc_hi_patches"
if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)

input_image_list = [im_name for im_name in os.listdir(input_image_dir) if (im_name.endswith(input_suffix))]
input_image_list.sort()
print("Number of input images: ", len(input_image_list))

nuclei_mask_list = [im_name for im_name in os.listdir(nuclei_mask_dir) if (im_name.endswith(input_suffix))]
nuclei_mask_list.sort()
print("Number of nuclei masks: ", len(nuclei_mask_list))

for i in range(len(input_image_list)):
    input_image_name = input_image_list[i]
    nuclei_mask_name = nuclei_mask_list[i]
    
    input_image_path = os.path.join(input_image_dir, input_image_name)
    nuclei_mask_path = os.path.join(nuclei_mask_dir, nuclei_mask_name)
    
    # Read the input image and nuclei mask
    #input_image = tiff.imread(input_image_path)
    #nuclei_mask = tiff.imread(nuclei_mask_path)
    input_image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    nuclei_mask = cv2.imread(nuclei_mask_path, cv2.IMREAD_UNCHANGED)
    
    # Convert instance segmentation mask to binary mask
    nuclei_mask_binary = np.zeros_like(nuclei_mask, dtype=np.uint8)
    nuclei_mask_binary[nuclei_mask > 0] = 255

    # smooth the edges of the binary mask
    nuclei_mask_binary = cv2.GaussianBlur(nuclei_mask_binary, (11, 11), 0)
    # reduce the global intensity by 128
    nuclei_mask_binary = np.where(nuclei_mask_binary > 158, nuclei_mask_binary - 158, 0)
    #nuclei_mask_binary = Image.fromarray(nuclei_mask_binary)
    

    # Add nuclei_mask_binary to the specified channel of the input image
    if nuc_hi_channel == "R":
        input_image[:, :, 2] = input_image[:, :, 2] + nuclei_mask_binary
    elif nuc_hi_channel == "G":
        input_image[:, :, 1] = input_image[:, :, 1] + nuclei_mask_binary
    elif nuc_hi_channel == "B":
        input_image[:, :, 0] = input_image[:, :, 0] + nuclei_mask_binary
    elif nuc_hi_channel == "RG":
        input_image[:, :, 2] = input_image[:, :, 2] + nuclei_mask_binary
        input_image[:, :, 1] = input_image[:, :, 1] + nuclei_mask_binary
    elif nuc_hi_channel == "RGB":
        input_image[:, :, 2] = input_image[:, :, 2] + nuclei_mask_binary
        input_image[:, :, 1] = input_image[:, :, 1] + nuclei_mask_binary
        input_image[:, :, 0] = input_image[:, :, 0] + nuclei_mask_binary
    
    output_image = input_image.copy()
    #output_image = Image.fromarray(output_image)
    output_image_name = input_image_name.replace(input_suffix, "_nuc_enhanced" + input_suffix)
    output_image_path = os.path.join(output_image_dir, output_image_name)
    
    #tiff.imwrite(output_image_path, output_image)
    cv2.imwrite(output_image_path, output_image)
    print(f"Processed: {output_image_name}")
