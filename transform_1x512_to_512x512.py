# Convert .tif image of shape [3, 512, 512, 1] to .tif image of shape [512, 512, 3]
# Apply this script to output of virtual staining model

import os
import numpy as np
import cv2
from PIL import Image
import tifffile as tiff

data_type = "AF - test"

model_name = "set37_paired_512_saliency_A65_B220"
epoch_name = "test_latest"

input_dir = "C:/Users/zpanp/projects/UTOM-master/results/" + model_name + "/" + epoch_name + "/images"
output_dir = "C:/Users/zpanp/projects/UTOM-master/results/" + model_name + "/" + epoch_name + "/images"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

input_im_list = [im_name for im_name in os.listdir(input_dir) if ("_rec_A" in im_name) and (im_name.endswith('.png') or im_name.endswith('.tif'))]
input_im_list.sort()

print("Number of input images: ", len(input_im_list))

for i in range(len(input_im_list)):
    input_im_name = input_im_list[i]
    input_im_path = os.path.join(input_dir, input_im_name)
    if ".png" in input_im_name:
        input_im = cv2.imread(input_im_path, cv2.IMREAD_UNCHANGED)
    else:
        input_im = tiff.imread(input_im_path)
        # squeeze the tiff image
        input_im = np.squeeze(input_im)
    
    print("Processing: ", input_im_name)
    print("location at full_im: ", i)
    # save the full image with tifffile
    output_im_name = input_im_name.replace("_fake_B", "_fake_B_3channel")
    output_im_path = os.path.join(output_dir, output_im_name)
    tiff.imsave(output_im_path, input_im)
    print("Saved: ", output_im_name)