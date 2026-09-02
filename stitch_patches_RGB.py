# Crop the test images into tiles with overlap
import os

import numpy as np
import cv2
from PIL import Image
import tifffile as tiff

data_type = "AF - test"

checkpoint_name = "set37_train_unpaired+paired_saliency_A70_B220_lambda25"
data_name = "RGB_RGB_nuc_hi_vHE"

patch_width_x = 512
patch_width_y = 512
overlap = 0
num_patches_x = 5
num_patches_y = 5

#input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/" + dataset_name + "/" + im_folder_name + netout_folder_name
#output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/" + dataset_name + "/" + im_folder_name + full_im_folder_name
patch_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/250529_slides/results/" + checkpoint_name + "/" + data_name
full_im_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/250529_slides/results/"+ checkpoint_name + "/" + data_name + "/full_images"

if not os.path.exists(full_im_dir):
    os.makedirs(full_im_dir)

patches_list = [im_name for im_name in os.listdir(patch_dir) if (im_name.endswith('_fake_B.png') or im_name.endswith('_fake_B.tif'))]
patches_list.sort()

print("Number of patches: ", len(patches_list))

num_patches_processed = 0

full_image = np.zeros((patch_width_y * num_patches_y, patch_width_x * num_patches_x, 3), dtype=np.uint8)
print("Full image shape: ", full_image.shape)

# iterate through all images in patches_list
for patch_idx in range(len(patches_list)):
    patch_name = patches_list[patch_idx]
    print("Processing image: ", patch_name)
    
    if patch_name.endswith('_fake_B.png'):
        input_im = cv2.imread(os.path.join(patch_dir, patch_name), cv2.IMREAD_UNCHANGED)
    elif patch_name.endswith('_fake_B.tif'):
        input_im = tiff.imread(os.path.join(patch_dir, patch_name))

    normalized_patch_idx = patch_idx % (num_patches_x * num_patches_y)
    
    full_image[normalized_patch_idx % num_patches_y * patch_width_x:(normalized_patch_idx % num_patches_y + 1) * patch_width_x,
            normalized_patch_idx // num_patches_y * patch_width_x:(normalized_patch_idx // num_patches_y + 1) * patch_width_x] = input_im

    num_patches_processed += 1

    if num_patches_processed % (num_patches_x * num_patches_y) == 0:
        full_image_resized = cv2.resize(full_image, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        output_file_name = patch_name.split('_')
        output_file_name = '_'.join(output_file_name[:2] + output_file_name[4:]) + '_full_image.png'
        cv2.imwrite(os.path.join(full_im_dir, output_file_name), full_image_resized)
        print(f"Saved stitched image to: {output_file_name}")
        full_image = np.zeros((patch_width_y * num_patches_y, patch_width_x * num_patches_x, 3), dtype=input_im.dtype)
        print("Full image shape reset: ", full_image.shape)

