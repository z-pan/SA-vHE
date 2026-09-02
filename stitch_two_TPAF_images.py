# Stitch two two-photon autofluorescence images, without overlapping regions
# before stitching, the brightness of the two images are balanced

import os
import sys

import cv2
import tifffile
import numpy as np
from tqdm import tqdm

def balance_brightness(img1, img2, output_dir):
    # Balance the brightness of two images of the corresponding channels
    # img1 and img2 are two images with the same size
    # return the balanced images
    img1 = img1.astype(np.float32) # img shape = [1024, 1024, 3]
    img2 = img2.astype(np.float32)
    for i in range(3):
        # calculate the mean pixel value of each channel
        img1_mean = np.mean(img1[:,:,i])
        img2_mean = np.mean(img2[:,:,i])
        img1[:,:,i] = img1[:,:,i] * img2_mean / img1_mean
    # normalizae the pixel values to [0, 255]
    img1 = np.clip(img1, 0, 255).astype(np.uint8)
    img2 = np.clip(img2, 0, 255).astype(np.uint8)
    # save the balanced images with tifffile.imsave
    print("Saving brightness balanced images: ")
    tifffile.imsave(output_dir + "/" + "balanced_img1.tif", img1)
    tifffile.imsave(output_dir + "/" + "balanced_img2.tif", img2)
    
    # see if img1 and img2 are empty
    if img1 is None or img2 is None:
        print("No images loaded, please check the image paths.")

    return img1, img2

# Load your images
images_dir = "C:/Users/zpanp/projects/datasets/virtual_staining/20241214_human_ovarian_primary site/20241114 slice 240703HOCIST240717-1/test3/input"
output_dir = "C:/Users/zpanp/projects/datasets/virtual_staining/20241214_human_ovarian_primary site/20241114 slice 240703HOCIST240717-1/test3/output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
image_names = os.listdir(images_dir)
image1_name = image_names[0]
image2_name = image_names[1]

image1_path = images_dir + '/' + image1_name
image2_path = images_dir + '/' + image2_name
images = [tifffile.imread(image1_path), tifffile.imread(image2_path)]
# check if images is empty
if len(images) == 0:
    print("No images loaded, please check the image paths.")
    sys.exit()

# Balance the brightness of the two images
images[0], images[1] = balance_brightness(images[0], images[1], output_dir)

# Step 1: Feature Detection
sift = cv2.SIFT_create()
keypoints = []
descriptors = []

for img in images:
    kp, des = sift.detectAndCompute(img, None)
    keypoints.append(kp)
    descriptors.append(des)

# Step 2: Feature Matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = []
for i in range(len(images) - 1):
    matches.append(bf.match(descriptors[i], descriptors[i + 1]))

# Step 3: Homography Estimation and Image Registration
# Use RANSAC for robust estimation
homographies = []
for i in range(len(matches)):
    src_pts = np.float32([keypoints[i][m.queryIdx].pt for m in matches[i]]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints[i + 1][m.trainIdx].pt for m in matches[i]]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    homographies.append(H)

# Step 4: Warp and Stitch
result = images[0]
for i in range(1, len(images)):
    result = cv2.warpPerspective(result, homographies[i-1], (result.shape[1] + images[i].shape[1], result.shape[0]))
    result[0:images[i].shape[0], 0:images[i].shape[1]] = images[i]

# Final result
output_name = "stitched_{}_{}.tif".format(image1_name, image2_name)
output_path = output_dir + '/' + output_name
tifffile.imsave(output_path, result)
print("Stitched image saved at {}".format(output_path))