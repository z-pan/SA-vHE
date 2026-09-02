# Generate saliency maps for AF and HE images

import os
import numpy as np
import cv2
import tifffile as tiff

thresholdA = 70
thresholdB = 220

input_A_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/saliency_threshold_test/trainA"
input_B_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/saliency_threshold_test/trainB"
output_A_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/saliency_threshold_test/trainA_saliency"
output_B_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/saliency_threshold_test/trainB_saliency"
if not os.path.exists(output_A_dir):
    os.makedirs(output_A_dir)
if not os.path.exists(output_B_dir):
    os.makedirs(output_B_dir)

input_A_list = [im_name for im_name in os.listdir(input_A_dir) if (im_name.endswith('.png') or im_name.endswith('.tif'))]
input_A_list.sort()
input_B_list = [im_name for im_name in os.listdir(input_B_dir) if (im_name.endswith('.png') or im_name.endswith('.tif'))]
input_B_list.sort()

for input_A_name in input_A_list:
    input_A_path = os.path.join(input_A_dir, input_A_name)
    if ".png" in input_A_name:
        input_A = cv2.imread(input_A_path, cv2.IMREAD_UNCHANGED)
    else:
        input_A = tiff.imread(input_A_path)
    
    # enhance the contrast of each channel of the image and merge them back
    input_A_R = input_A[:,:,0]
    input_A_G = input_A[:,:,1]
    input_A_B = input_A[:,:,2]
    input_A_R = cv2.equalizeHist(input_A_R)
    #input_A_G = cv2.equalizeHist(input_A_G)
    #input_A_B = cv2.equalizeHist(input_A_B)
    #input_A = cv2.merge((input_A_R, input_A_G, input_A_B))

    # assign pixel in input_A_R higher than 45 and lower than 95 to 255, and the rest to 0
    input_A_R = np.where((input_A_R > 40) & (input_A_R < 60), 255, 0)

    # calculate mean value along the channel dimension
    input_A_mean = np.mean(input_A, axis=2)
    input_A_R = np.where((input_A_mean > 40) & (input_A_mean < 90), 255, 0)

    #real_A_normal = (real_A_mean - (self.opt.threshold_A/127.5-1))*100
    input_A_normal = (input_A_mean - (thresholdA/127.5-1)) * 100
    input_A_sigmoid = 1 / (1 + np.exp(-input_A_normal)) # normalize to 0-1 using sigmoid function
    print("A sigmoid shape: ", input_A_sigmoid.shape)
    print("A sigmoid: ", input_A_sigmoid)
    
    input_A_sigmoid = input_A_sigmoid.astype(np.uint8)
    input_A_sigmoid = input_A_sigmoid * 255
    #print("A sigmoid: ", input_A_sigmoid)
    
    output_A_name = input_A_name.replace(".tif", "_saliency_map.tif")
    output_A_path = os.path.join(output_A_dir, output_A_name)
    tiff.imsave(output_A_path, input_A_R)

for input_B_name in input_B_list:
    input_B_path = os.path.join(input_B_dir, input_B_name)
    if ".png" in input_B_name:
        input_B = cv2.imread(input_B_path, cv2.IMREAD_UNCHANGED)
    else:
        input_B = tiff.imread(input_B_path)

    # calculate mean value along the channel dimension
    input_B_mean = np.mean(input_B, axis=2)
    input_B_normal = (input_B_mean - (thresholdB/127.5-1)) * 100
    input_B_sigmoid = 1 - (1 / (1 + np.exp(-input_B_normal))) # normalize to 0-1 using sigmoid function
    input_B_sigmoid = input_B_sigmoid.astype(np.uint8)
    input_B_sigmoid = input_B_sigmoid * 255

    output_B_name = input_B_name.replace(".tif", "_saliency_map.tif")
    output_B_path = os.path.join(output_B_dir, output_B_name)
    tiff.imsave(output_B_path, input_B_mean)