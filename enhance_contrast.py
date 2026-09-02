# enhance contrast of .tif grayscale images

import os
import cv2
import numpy as np
import tifffile as tiff

dataset_name = "250529_slides"

contrast_enhancement_method = "CLAHE"
#contrast_enhancement_method = "saturation"

input_path = "C:/Users/zpanp/projects/denoising-fluorescence-master/denoising/datasets/my_data/test_data/0614_test/03_og_gray_patches_VST_TV_out"
output_path = "C:/Users/zpanp/projects/denoising-fluorescence-master/denoising/datasets/my_data/test_data/0614_test/03_og_gray_patches_VST_TV_out_CLAHE"
#input_path = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/" + dataset_name + "/02_og_gray_patches"
#output_path = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/" + dataset_name + "/03_gray_local_CLAHE_patches"
if not os.path.exists(output_path):
    os.makedirs(output_path)

input_im_list = [im_name for im_name in os.listdir(input_path) if (im_name.endswith('.png') or im_name.endswith('.tif') or im_name.endswith('.tiff'))]
input_im_list.sort()
print("Number of input images: ", len(input_im_list))

def saturate_contrast(img, lower_percentile=1, upper_percentile=99):
    if img.ndim == 2:  # Grayscale
        flat = img.flatten()
        low = np.percentile(flat, lower_percentile)
        high = np.percentile(flat, upper_percentile)
        img = np.clip((img - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    elif img.ndim == 3:  # RGB or multi-channel
        img_out = np.zeros_like(img, dtype=np.uint8)
        for c in range(img.shape[2]):
            channel = img[:, :, c]
            low = np.percentile(channel, lower_percentile)
            high = np.percentile(channel, upper_percentile)

            # avoid division by 0 warning: if high==low, do not perform intensity stretching, output the original channel
            if high - low < 1e-5:
                img_out[:,:,c] = channel.astype(np.uint8)
            else:
                stretched = (channel - low) * 255.0 / (high - low)
                img_out[:,:,c] = np.clip(stretched, 0, 255).astype(np.uint8)
            #img_out[:, :, c] = np.clip((channel - low) * 255.0 / (high - low), 0, 255)
        img = img_out
    return img

def CLAHE_contrast(img, clipLimit, tileGridSize):
    if img.ndim == 2: # Grayscale
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        img_clahe = clahe.apply(img)
    
    # if image is RGB, convert cv2 BGR format to LAB format
    elif img.ndim == 3: # RGB or multi-channel
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
        # Split LAB channels
        l, a, b = cv2.split(img_lab)
    
        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        l_clahe = clahe.apply(l)

        # Merge back and convert to BGR
        img_clahe = cv2.merge((l_clahe, a, b))
        img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2BGR)
    
    return img_clahe

for input_im_name in input_im_list:
    if (input_im_name.endswith('.tif') or input_im_name.endswith('.tiff')): # for image in tiff format
        print("Reading tiff image...")
        input_im = tiff.imread(os.path.join(input_path, input_im_name))
        input_im = input_im.astype(np.uint8) # numpy array
    elif (input_im_name.endswith('.png')): # for image in png format
        print("Reading png image...")
        input_im = cv2.imread(os.path.join(input_path, input_im_name), cv2.IMREAD_UNCHANGED)

    if contrast_enhancement_method == "CLAHE":
        enhanced_im = CLAHE_contrast(input_im, 2.0, (8,8))
    elif contrast_enhancement_method == "saturation":
        enhanced_im = saturate_contrast(input_im, 1, 99)

    # Option#1: enhance contrast of the grayscale image by CLAHE
    #enhanced_im = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(input_im)
    # Option#2: enhance contrast of the grayscale image by histogram equalization
    #enhanced_im = cv2.equalizeHist(input_im)

    output_im_path = os.path.join(output_path, input_im_name)
    if input_im_name.endswith('.png'):
        cv2.imwrite(output_im_path, enhanced_im)
    elif input_im_name.endswith('.tif') or input_im_name.endswith('.tiff'):
        tiff.imwrite(output_im_path, enhanced_im)
    print("Processed: ", input_im_name)