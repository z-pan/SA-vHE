# cell segmentation by multi-stage merging
# reference: Automatic segmentation of skin cells in multiphoton data using multi-stage merging (2021, Nat SR)
# Pre-processing: top-hat transformation, CLAHE, anisotropic diffusion
# Segmentation: watershed, multi-stage merging
# Semantic assignment: feature extraction, segmentation

import os

import numpy as np
import cv2
from PIL import Image
import tifffile as tiff
import torch
from skimage import io, color, filters, measure, morphology

#data_type = "AF - train"
data_type = "AF - test"
#data_type = "HE - train"

dataset_name = "nuc_enhance_test"

if data_type == "AF - train":
    input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/train_data/" + dataset_name + "/01_AF_png"
    output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/train_data/" + dataset_name + "/01_AF_png_mask"

elif data_type == "HE - train":
    input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/train_data/" + dataset_name + "/original_HE_png_selected_temp"
    output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/train_data/" + dataset_name + "/original_HE_png_selected_mask"

elif data_type == "AF - test":
    #input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/test_data/" + dataset_name + "/testA"
    #output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/test_data/" + dataset_name + "/testA_mask"
    input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/test_data/" + dataset_name + "/og"
    output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/test_data/" + dataset_name + "/mask"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

input_im_list = [im_name for im_name in os.listdir(input_dir) if im_name.endswith('.png')]
input_im_list.sort()

for input_im_name in input_im_list:
    print("Processing: ", input_im_name)

    input_im_path = os.path.join(input_dir, input_im_name)
    input_im = cv2.imread(input_im_path, cv2.IMREAD_GRAYSCALE)

    # top-hat transformation
    # kernel: a circular structuring element, diameter 6~9 produce accepted results
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    top_hat = cv2.morphologyEx(input_im, cv2.MORPH_TOPHAT, kernel)
    black_hat = cv2.morphologyEx(input_im, cv2.MORPH_BLACKHAT, kernel)
    hat_transformed_im = input_im + top_hat + black_hat
    # save hat_transformed_im in png format
    #hat_transformed_im_name = input_im_name.split('.')[0] + '_hat_transformed.png'
    #cv2.imwrite(os.path.join(output_dir, hat_transformed_im_name), hat_transformed_im)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_im = clahe.apply(hat_transformed_im)
    # save clahe_im in png format
    #clahe_im_name = input_im_name.split('.')[0] + '_clahe.png'
    #cv2.imwrite(os.path.join(output_dir, clahe_im_name), clahe_im)

    # anisotropic diffusion for denoising by smoothing, with structure preservation
    # if clahe_im is grayscale, convert it to color image
    clahe_im_BGR = cv2.cvtColor(clahe_im, cv2.COLOR_GRAY2BGR)
    anisotropic_diffusion_im = cv2.ximgproc.anisotropicDiffusion(clahe_im_BGR, alpha=0.1, K=0.02, niters=10)
    # convert anisotropic_diffusion_im to grayscale
    anisotropic_diffusion_im = cv2.cvtColor(anisotropic_diffusion_im, cv2.COLOR_BGR2GRAY)
    # save anisotropic_diffusion_im in png format
    #anisotropic_diffusion_im_name = input_im_name.split('.')[0] + '_anisotropic_diffusion.png'
    #cv2.imwrite(os.path.join(output_dir, anisotropic_diffusion_im_name), anisotropic_diffusion_im)

    # thresholding
    #_, binary_im = cv2.threshold(anisotropic_diffusion_im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # watershed
    watershed_im = anisotropic_diffusion_im.copy()
    _, markers = cv2.connectedComponents(watershed_im)
    markers = markers + 10
    # convert watershed_im to RGB
    watershed_im = cv2.cvtColor(watershed_im, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(watershed_im, markers)
    watershed_im[markers == -1] = 255
    # save watershed im in png format
    watershed_im_name = input_im_name.split('.')[0] + '_watershed.png'
    #cv2.imwrite(os.path.join(output_dir, watershed_im_name), watershed_im)

    # multi-stage merging


    # model-based semantic assignement of nucleus, cytoplasm, intracellular matrix
    # feature extraction
    # segmentation


    # save mask
