# Invert TPAF images

import os
import cv2
import numpy as np
import tifffile as tiff

input_path = "C:/Users/zpanp/projects/UTOM-master/datasets/train_data/set37/00_og_gray_contrast_enhanced"
output_path = "C:/Users/zpanp/projects/UTOM-master/datasets/train_data/set37/00_og_gray_inverted"
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
    elif (input_im_name.endswith('.png')): # for image in png format
        print("Reading png image...")
        input_im = cv2.imread(os.path.join(input_path, input_im_name), cv2.IMREAD_UNCHANGED)

    # if the image is RGB, convert it to grayscale
    if len(input_im.shape) == 3 and input_im.shape[2] == 3:
        # Convert RGB to grayscale using the luminosity method
        input_im = cv2.cvtColor(input_im, cv2.COLOR_BGR2GRAY)
    
    # invert the grayscale image
    inverted_im = 255 - input_im

    output_im_path = os.path.join(output_path, input_im_name)
    if input_im_name.endswith('.png'):
        cv2.imwrite(output_im_path, inverted_im)
    elif input_im_name.endswith('.tif') or input_im_name.endswith('.tiff'):
        tiff.imwrite(output_im_path, inverted_im)
    print("Processed: ", input_im_name)