# Color deconvolution for H&E images

import os

import numpy as np
import cv2
import tifffile as tiff
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2hed
import histomicstk as htk

# color deconv options: 'macenko_pca', 'xu_snmf', 'supervised', 'skimage_rgb2hed', 'skimage_rgb2hed'
color_deconv_method = 'skimage_rgb2hed'

I_0 = 255  # 背景亮度基准

#input_HE_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/train_data/set37/01_HE_png"
#output_deconv_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/train_data/set37/02_HE_color_deconv_png"
#output_H_dir = os.path.join(output_deconv_dir, "hematoxylin_channel")
#output_E_dir = os.path.join(output_deconv_dir, "eosin_channel")
#output_stain_matrix_dir = os.path.join(output_deconv_dir, "stain_matrix")
input_HE_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/250711_slides/00_og_real_HE_full"
output_H_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/250711_slides/01_og_real_HE_deconv_H_full"
output_E_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/250711_slides/01_og_real_HE_deconv_E_full"
output_stain_matrix_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/250711_slides/01_og_real_HE_deconv_stain_matrix"
if not os.path.exists(output_H_dir):
    os.makedirs(output_H_dir)
if not os.path.exists(output_E_dir):
    os.makedirs(output_E_dir)
if not os.path.exists(output_stain_matrix_dir):
    os.makedirs(output_stain_matrix_dir)

input_HE_list = [im_name for im_name in os.listdir(input_HE_dir) if (im_name.endswith(".tif") or im_name.endswith(".tiff") or im_name.endswith(".png"))]
input_HE_list.sort()
print("Number of input H&E images: ", len(input_HE_list))

for input_HE_name in input_HE_list:
    input_HE_path = os.path.join(input_HE_dir, input_HE_name)
    
    # Load the H&E image
    if input_HE_name.endswith('.tif') or input_HE_name.endswith('.tiff'):
        #input_HE_RGB = tiff.imread(input_HE_path)
        input_HE_RGB = tiff.imread(input_HE_path, key=0)  # sometimes tiff file has multiple pages, read the page needed
    elif input_HE_name.endswith('.png'):
        input_HE_RGB = cv2.imread(input_HE_path, cv2.IMREAD_UNCHANGED)

    if color_deconv_method == 'skimage_rgb2hed':
        print("Using skimage's rgb2hed for color deconvolution.")
        img_hed = rgb2hed(input_HE_RGB)
        hematoxylin = img_hed[:, :, 0]
        eosin = img_hed[:, :, 1]
        print("hematoxylin shape: ", hematoxylin.shape)
        print("eosin shape: ", eosin.shape)
        # Normalize the channels to 0-255
        hematoxylin = (hematoxylin - np.min(hematoxylin)) / (np.max(hematoxylin) - np.min(hematoxylin)) * 255
        eosin = (eosin - np.min(eosin)) / (np.max(eosin) - np.min(eosin)) * 255

    elif color_deconv_method == 'macenko_pca':
        print("Using Macenko PCA method for color deconvolution.")
        color_matrix = htk.preprocessing.color_deconvolution.rgb_separate_stains_macenko_pca(input_HE_RGB, I_0=255)
        hematoxylin, eosin = htk.preprocessing.color_deconvolution.color_deconvolution(input_HE_RGB, color_matrix)
    
    elif color_deconv_method == 'xu_snmf':
        print("Using Xu SNMF method for color deconvolution.")
        im_sda = htk.preprocessing.color_conversion.rgb_to_sda(input_HE_RGB, I_0=I_0)
        stain_colors = np.array([
            htk.preprocessing.color_deconvolution.stain_color_map['hematoxylin'],
            htk.preprocessing.color_deconvolution.stain_color_map['eosin']
        ]).T  # shape: [3,2]
        W2 = htk.preprocessing.color_deconvolution.separate_stains_xu_snmf(im_sda, w_init=stain_colors, beta=0.2)
        from htk.preprocessing.color_deconvolution import complement_stain_matrix
        W = complement_stain_matrix(W2)
        hematoxylin, eosin = htk.preprocessing.color_deconvolution.separate_stains(input_HE_RGB, W)
    
    elif color_deconv_method == 'supervised':
        print("Using Supervised method for color deconvolution.")
        from htk.preprocessing.color_deconvolution import stain_color_map, complement_stain_matrix
        W_init = np.array([
            stain_color_map['hematoxylin'],
            stain_color_map['eosin'],
            stain_color_map.get('null', [0, 0, 0])
        ]).T
        from htk.preprocessing.color_deconvolution import complement_stain_matrix
        W = complement_stain_matrix(W_init)
        hematoxylin, eosin = htk.preprocessing.color_deconvolution.separate_stains(input_HE_RGB, W)
    
    # Save stain matrix
    #stain_matrix_filename = input_HE_name + "_" + color_deconv_method + "_stain_matrix.npy"
    #print(f"Saving stain matrix to {stain_matrix_filename}")
    #np.save(os.path.join(output_stain_matrix_dir, stain_matrix_filename), W)

    # Save the separated channels
    output_H_path = os.path.join(output_H_dir, input_HE_name)
    output_E_path = os.path.join(output_E_dir, input_HE_name)
    
    io.imsave(output_H_path, hematoxylin.astype(np.uint8))
    io.imsave(output_E_path, eosin.astype(np.uint8))
    
    print(f"Processed: {input_HE_name}")