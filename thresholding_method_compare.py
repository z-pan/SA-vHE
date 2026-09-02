# Enhance the contrast of nuclei regions in the image

import os

import numpy as np
import cv2
from PIL import Image
import tifffile as tiff
import torch
from skimage import io, color, filters, measure, morphology

data_type = "AF - train"
#data_type = "AF - test"
#data_type = "HE - train"

dataset_name = "set10"
#segmentation_method = "adaptive-mean"
segmentation_method = "adaptive-gaussian"
#segmentation_method = "adaptive-range"
#segmentation_method = "global-threshold"
#segmentation_method = "global-range"
#segmentation_method = "adaptive-subtract-global"
#segmentation_method = "global-subtract-adaptive"
#segmentation_method = "adaptive-subtract-adaptive"
#segmentation_method = "adaptive-then-fill-holes"
#segmentation_method = "utom-sc"

# parameter for adaptive thresholding
block_size = 37 # window size for adaptive thresholding, must be an odd number
contrast_diff = 8 # contrast threshold for adaptive thresholding

# parameter for global thresholding
global_threshold = 120

if data_type == "AF - train":
    input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/train_data/" + dataset_name + "/01_AF_png"
    output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/train_data/" + dataset_name + "/01_AF_png_mask"

elif data_type == "HE - train":
    input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/train_data/" + dataset_name + "/original_HE_png_selected_temp"
    output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/train_data/" + dataset_name + "/original_HE_png_selected_mask"

elif data_type == "AF - test":
    input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/test_data/" + dataset_name + "/testA"
    output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/test_data/" + dataset_name + "/testA_mask"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

input_im_list = [im_name for im_name in os.listdir(input_dir) if im_name.endswith('.png')]
input_im_list.sort()

for input_im_name in input_im_list:
    # read image
    input_im = Image.open(os.path.join(input_dir, input_im_name))
    input_im = np.array(input_im)
    #print("input_im_shape: ", input_im.shape)

    # convert RGB to grayscale for HE images
    if "HE" in data_type:
        input_im = np.mean(input_im, axis=2)
        input_im = input_im.round().astype(np.uint8)
        print("input_im shape: ", input_im.shape)
        print("input_im max: ", input_im.max())
        print("input_im min: ", input_im.min())

    # cv2 adaptive mean thresholding
    if segmentation_method == "adaptive-mean":
        input_im = cv2.adaptiveThreshold(input_im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                         cv2.THRESH_BINARY_INV, block_size, contrast_diff)
        output_im = Image.fromarray(input_im)
    
    elif segmentation_method == "adaptive-gaussian":
        input_im = cv2.adaptiveThreshold(input_im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                         cv2.THRESH_BINARY_INV, block_size, contrast_diff)
        
        # find and preserve the remaining connected white pixel areas with shape of round or ellipse
        labeled_regions, num_regions = measure.label(input_im, return_num=True)
        region_props = measure.regionprops(labeled_regions)
        for region in region_props:
            if (region.bbox[2] - region.bbox[0] > 47 or region.bbox[3] - region.bbox[1] > 47):
                input_im[labeled_regions == region.label] = 0
            else:
                if region.area > 500:
                    input_im[labeled_regions == region.label] = 0
                else:
                    if region.eccentricity > 0.8: # 0 is a circle, 1 is a line
                        input_im[labeled_regions == region.label] = 0
                    else:
                        input_im[labeled_regions == region.label] = 255
        
        # make the connected white pixel areas more round
        kernel = np.ones((5, 5), np.uint8)
        input_im = cv2.morphologyEx(input_im, cv2.MORPH_CLOSE, kernel)

        output_im = Image.fromarray(input_im)

    elif segmentation_method == "adaptive-range":
        input_im1 = cv2.adaptiveThreshold(input_im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                         cv2.THRESH_BINARY, block_size1, contrast_diff1)
        input_im2 = cv2.adaptiveThreshold(input_im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                         cv2.THRESH_BINARY_INV, block_size2, contrast_diff2)
        input_im = input_im1 & input_im2
        output_im = Image.fromarray(input_im)

    elif segmentation_method == "global-threshold":
        input_im_mean = np.mean(input_im)
        print("input_im_mean: ", input_im_mean)
        global_threshold = int(0.45 * input_im_mean)
        (T, segmented_im) = cv2.threshold(input_im, global_threshold, 255, cv2.THRESH_BINARY)
        output_im = Image.fromarray(segmented_im)

    elif segmentation_method == "global-range":
        enhanced_image = filters.rank.enhance_contrast(input_im, morphology.disk(5))
        thresholded_image = np.logical_and(enhanced_image >= 30, enhanced_image <= 75)
        labeled_regions, num_regions = measure.label(thresholded_image, return_num=True)
        print("num_regions: ", num_regions)
        region_props = measure.regionprops(labeled_regions)
        segmented_image = np.zeros_like(input_im, dtype=np.uint8)
        for region in region_props:
            if 30 <= region.equivalent_diameter <= 50:
                segmented_image[labeled_regions == region.label] = 255
        output_im = Image.fromarray(segmented_image)

    elif segmentation_method == "adaptive-subtract-global":
        # (nucleus content + background) - (background)
        # parameter for adaptively thresholding a range
        block_size = 11
        contrast_diff = 10

        enhanced_image = filters.rank.enhance_contrast(input_im, morphology.disk(5))
        input_im_mean = np.mean(input_im)
        global_threshold = int(0.45 * input_im_mean)
        print("input_im_mean: ", input_im_mean)
        print("global_threshold: ", global_threshold)
        adaptive_thresholded_im = cv2.adaptiveThreshold(input_im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                            cv2.THRESH_BINARY_INV, block_size, contrast_diff)
        (T, global_thresholded_im) = cv2.threshold(input_im, global_threshold , 255, cv2.THRESH_BINARY_INV)
        #segmented_im = global_thresholded_im - adaptive_thresholded_im
        segmented_im = adaptive_thresholded_im - global_thresholded_im
        output_im = Image.fromarray(segmented_im)

    elif segmentation_method == "adaptive-subtract-adaptive":
        # parameter for adaptively thresholding a range
        block_size1 = 41
        contrast_diff1 = 3
        block_size2 = 41
        contrast_diff2 = 10

        enhanced_image = filters.rank.enhance_contrast(input_im, morphology.disk(5))
        input_im_mean = np.mean(input_im)
        adaptive_thresholded_im1 = cv2.adaptiveThreshold(input_im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                            cv2.THRESH_BINARY_INV, block_size1, contrast_diff1)
        adaptive_thresholded_im2 = cv2.adaptiveThreshold(input_im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                            cv2.THRESH_BINARY_INV, block_size2, contrast_diff2)
        segmented_im = adaptive_thresholded_im1 - adaptive_thresholded_im2
        output_im = Image.fromarray(segmented_im)

    elif segmentation_method == "adaptive-then-fill-holes":
        # parameter for adaptively thresholding
        block_size = 41
        contrast_diff = 10

        enhanced_image = filters.rank.enhance_contrast(input_im, morphology.disk(5))
        adaptive_thresholded_mask = cv2.adaptiveThreshold(input_im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                            cv2.THRESH_BINARY, block_size, contrast_diff)
        # fill holes
        hole_filled_mask = morphology.remove_small_holes(adaptive_thresholded_mask, area_threshold=64) * 255
        final_mask = hole_filled_mask - adaptive_thresholded_mask
        segmented_im = input_im & (~final_mask)
        output_im = Image.fromarray(final_mask)

        adaptive_thresholded_mask_im = Image.fromarray(adaptive_thresholded_mask)
        hole_filled_mask_im = Image.fromarray(hole_filled_mask)
        final_mask_im = Image.fromarray(final_mask)
        adaptive_thresholded_mask_suffix = "_" + segmentation_method + "_ws" + str(block_size) + "_c" + str(contrast_diff) + "_adaptive_mask.png"
        hole_filled_mask_suffix = "_" + segmentation_method + "_ws" + str(block_size) + "_c" + str(contrast_diff) + "_hole_filled_mask.png"
        final_mask_suffix = "_" + segmentation_method + "_ws" + str(block_size) + "_c" + str(contrast_diff) + "_final_mask.png"
        output_im_suffix = "_" + segmentation_method + "_ws" + str(block_size) + "_c" + str(contrast_diff) + ".png"
        adaptive_thresholded_mask_name = input_im_name.replace(".png", adaptive_thresholded_mask_suffix)
        hole_filled_mask_name = input_im_name.replace(".png", hole_filled_mask_suffix)
        final_mask_name = input_im_name.replace(".png", final_mask_suffix)
        output_im_name = input_im_name.replace(".png", output_im_suffix)
        adaptive_thresholded_mask_im.save(os.path.join(output_dir, adaptive_thresholded_mask_name))
        hole_filled_mask_im.save(os.path.join(output_dir, hole_filled_mask_name))
        final_mask_im.save(os.path.join(output_dir, final_mask_name))
        output_im.save(os.path.join(output_dir, output_im_name))

    elif segmentation_method == "utom-sc":
        input_mean = np.mean(input_im)
        input_mean_norm = (input_mean - 127.5) / 127.5
        input_norm = (input_mean_norm - (70/127.5-1))*100
        input_sigmoid = 1 / (1 + np.exp(-input_norm))
        print("input_sigmoid_shape: ", input_sigmoid.shape)
        print("input_sigmoid_max: ", input_sigmoid.max())
        print("input_sigmoid_min: ", input_sigmoid.min())
        input_im = input_sigmoid * 255
        input_im = input_im.astype(np.uint8)
        print("input_im type: ", type(input_im))
        output = Image.fromarray(input_im)

    # save the segmented image
    output_im_suffix = "_" + segmentation_method + "_ws" + str(block_size) + "_c" + str(contrast_diff) + ".png"
    output_im_name = input_im_name.replace(".png", output_im_suffix)
    output_im.save(os.path.join(output_dir, output_im_name))

"""
def content_loss(self):
    # print('self.rec_A -----> ',self.rec_A.shape)
    # print('self.real_A -----> ',self.real_A.shape)
    # print('self.rec_B -----> ',self.rec_B.shape)
    # print('self.real_B -----> ',self.real_B.shape)
    # print('self.fake_A -----> ',self.fake_A.shape)
    # print('self.fake_B -----> ',self.fake_B.shape)

    L1_function = torch.nn.L1Loss()
    real_A_mean = torch.mean(self.real_A,dim=1,keepdim=True)
    real_B_mean = torch.mean(self.real_B,dim=1,keepdim=True)
    fake_A_mean = torch.mean(self.fake_A,dim=1,keepdim=True)
    fake_B_mean = torch.mean(self.fake_B,dim=1,keepdim=True)

    real_A_normal = (real_A_mean - (self.opt.threshold_A/127.5-1))*100
    real_B_normal = (real_B_mean - (self.opt.threshold_B/127.5-1))*100

    fake_A_normal = (fake_A_mean - (self.opt.threshold_B/127.5-1))*100
    fake_B_normal = (fake_B_mean - (self.opt.threshold_A/127.5-1))*100

    real_A_sigmoid = torch.sigmoid(real_A_normal)
    real_B_sigmoid = 1 - torch.sigmoid(real_B_normal)

    fake_A_sigmoid = torch.sigmoid(fake_A_normal)
    fake_B_sigmoid = 1 - torch.sigmoid(fake_B_normal)

    content_loss_A = L1_function( real_A_sigmoid , fake_B_sigmoid )
    content_loss_B = L1_function( fake_A_sigmoid , real_B_sigmoid )

    content_loss_rate = 50*np.exp(-(self.opt.counter/self.opt.data_size))
    content_loss = (content_loss_A + content_loss_B)*content_loss_rate
    return content_loss
"""