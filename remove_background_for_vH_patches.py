# Read in results_vH_patches and og_TPAF_gray_patches_nuc_mask_bin_uint8 (as binary masks), 
# only preserve the nuclei regions (255 in og_TPAF_gray_patches_nuc_mask_bin_uint8) in the results_vH_patches images.

import os
import cv2
import tifffile as tiff
import numpy as np

input_vH_patches_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/250711_slides/05_results_vH_patches"
input_mask_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/250711_slides/04_og_TPAF_gray_patches_nuc_mask_bin_uint8"
output_vH_patches_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/250711_slides/06_results_vH_patches_background_remove"
if not os.path.exists(output_vH_patches_dir):
    os.makedirs(output_vH_patches_dir)

input_vH_patches_list = [im_name for im_name in os.listdir(input_vH_patches_dir) if (im_name.endswith('.tif') or im_name.endswith('.tiff') or im_name.endswith('.png'))]
input_vH_patches_list.sort()
input_mask_list = [im_name for im_name in os.listdir(input_mask_dir) if (im_name.endswith('.tif') or im_name.endswith('.tiff') or im_name.endswith('.png'))]
input_mask_list.sort()
print("Number of input vH patches: ", len(input_vH_patches_list))
print("Number of input masks: ", len(input_mask_list))

for i in range(len(input_vH_patches_list)):
    vH_patch = cv2.imread(os.path.join(input_vH_patches_dir, input_vH_patches_list[i]), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(os.path.join(input_mask_dir, input_mask_list[i]), cv2.IMREAD_UNCHANGED)
    
    # if vH_patch is 3 channels, convert to 1 channel
    if vH_patch.ndim == 3:
        vH_patch = cv2.cvtColor(vH_patch, cv2.COLOR_BGR2GRAY)

    # Apply the mask to the vH patch
    vH_patch[mask == 0] = 0

    # Save the result
    cv2.imwrite(os.path.join(output_vH_patches_dir, input_vH_patches_list[i]), vH_patch)
    print(f"Processed: {input_vH_patches_list[i]}")