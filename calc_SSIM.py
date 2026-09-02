# Calculate SSIM between the paired and aligned virtual stained image and real stain image

import os

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import tifffile as tiff
import PIL.Image as Image


data_type = "AF - test"

dataset_name = "set30_23FJP095-1.1"
model_output_name = "compare_set30-32_ssim_loss"

is_grayscale = False

#input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/test_data/" + dataset_name + "/" + model_output_name
#output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/test_data/" + dataset_name + "/" + model_output_name + "/full_im"
gt_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/test_data/set30_23FJP095-1.3.4/same_crop/01_real"
input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/test_data/set30_23FJP095-1.3.4/same_crop/01_virtual"
output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/test_data/set30_23FJP095-1.3.4/same_crop/01_virtual"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

input_list = [file for file in os.listdir(input_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
gt_list = [file for file in os.listdir(gt_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
input_list.sort()

# Initialize lists to store the metrics
ssim_values = []

# Calculate metrics for each pair of images
for input_fn, gt_fn in zip(input_list, gt_list):
    # Load images
    input_im = Image.open(os.path.join(input_dir, input_fn))
    gt_im = Image.open(os.path.join(gt_dir, gt_fn))

    # Convert images to numpy arrays
    input_arr = np.array(input_im, dtype=np.uint8)
    gt_arr = np.array(gt_im, dtype=np.uint8)
    #input_arr = cv2.equalizeHist(input_arr)
    #gt_arr = cv2.equalizeHist(gt_arr)

    # Calculate PSNR and SSIM

    # check if the image is RGB or grayscale
    if len(input_arr.shape) == 3:
        #ssim_value = ssim(input_arr[:,:,0], gt_arr[:,:,0], multichannel=False, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255)
        ssim_value = ssim(input_arr, gt_arr, multichannel=True)
    else:
        ssim_value = ssim(input_arr, gt_arr, multichannel=False, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255)

    # Append to the lists
    ssim_values.append(ssim_value)

# Calculate the average PSNR and SSIM
avg_ssim = np.mean(ssim_values)

# Save results to the text file
PSNR_SSIM_out_path = os.path.join(output_dir, "PSNR_SSIM_results.txt")
with open(PSNR_SSIM_out_path, 'w') as f:
    f.write(f"Average SSIM: {avg_ssim:.4f}\n")
    f.write("PSNR and SSIM for each image:\n")
    for i, (input_file, output_file) in enumerate(zip(input_list, gt_list)):
        f.write(f"Image {i+1}: {input_file} - SSIM: {ssim_values[i]:.4f}\n")

print("Metrics calculation and saving completed.")