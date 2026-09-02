import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

def register_images(target_image, source_image):
    
    # extract red channel
    target_red = target_image[:,:,2]
    source_red = source_image[:,:,2]

    # Detect keypoints and descriptors using SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(target_red, None)
    kp2, des2 = sift.detectAndCompute(source_red, None)

    # Match keypoints using FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Estimate the transformation (homography) between images
    print("number of good match points: ", len(good_matches))
    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        #M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) # M is the transformation matrix
        M, _ = cv2.estimateAffine2D(src_pts, dst_pts)
        #aligned_image = cv2.warpPerspective(source_image, M, (target_image.shape[1], target_image.shape[0]))

        #M = M.astype(np.float64)
        aligned_image = cv2.warpAffine(source_image, M, (target_image.shape[1], target_image.shape[0]))
    else:
        aligned_image =  None
    
    return aligned_image


if __name__ == '__main__':
    
    source_im_path = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/test_data/set30_23FJP095-1.3.4/same_crop/00_real"
    target_im_path = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/test_data/set30_23FJP095-1.3.4/same_crop/00_virtual"
    output_im_path = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/test_data/set30_23FJP095-1.3.4/same_crop/01_reg" # source images are aligned to target images, and save to output_path
    if not os.path.exists(output_im_path):
        os.makedirs(output_im_path)

    source_im_list = [im_name for im_name in os.listdir(source_im_path)]
    source_im_list.sort()
    target_im_list = [im_name for im_name in os.listdir(target_im_path)]
    target_im_list.sort()

    total_im_num = len(source_im_list)

    processed_im_num = 1

    # Initialize the tqdm progress bar
    progress_bar = tqdm(total=total_im_num, desc="Processing images", unit="image", position=0, leave=True)
    
    for range_idx in range(total_im_num):
        source_im_name = source_im_list[range_idx]
        target_im_name = target_im_list[range_idx]
        print("source_im_name: ", source_im_name)
        print("target_im_name: ", target_im_name)

        source_im = cv2.imread(source_im_path + '/' + source_im_name, cv2.IMREAD_UNCHANGED)
        target_im = cv2.imread(target_im_path + '/' + target_im_name, cv2.IMREAD_UNCHANGED)
        output_im = register_images(target_im, source_im)

        output_im_name = output_im_path + '/' + source_im_name.replace('.jpg', '_reg.png')
        output_source_im_name = output_im_path + '/' + source_im_name.replace('.jpg', '.png')
        output_target_im_name = output_im_path + '/' + target_im_name.replace('.jpg', '.png')
        
        if output_im is not None:
            cv2.imwrite(output_im_name, output_im)
        cv2.imwrite(output_source_im_name, source_im)
        cv2.imwrite(output_target_im_name, target_im)

        processed_im_num += 1
        progress_bar.update(1)
        progress_bar.set_postfix_str(f"Processed {processed_im_num}/{total_im_num} images")
