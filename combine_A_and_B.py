import os
import numpy as np
import cv2
import tifffile as tiff
import argparse
from multiprocessing import Pool

pathA = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/train_data/set37_paired/A/test"
pathB = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/train_data/set37_paired/B/test"
pathAB = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/train_data/set37_paired_test"
if not os.path.exists(pathAB):
    os.makedirs(pathAB)

def check_same_filename(pathA, pathB):
    path_list_A = os.listdir(pathA)
    path_list_B = os.listdir(pathB)
    path_list_A.sort()
    path_list_B.sort()
    for i in range(len(path_list_A)):
        if path_list_A[i] != path_list_B[i]:
            print("Different: ", path_list_A[i], path_list_B[i])
            return False
    print("Same")
    return True

def image_write(path_A, path_B, path_AB):
    #im_A = tiff.imread(path_A)
    #im_B = tiff.imread(path_B)
    #im_AB = np.concatenate([im_A, im_B], 1)
    #tiff.imsave(path_AB, im_AB)

    im_A = cv2.imread(path_A, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_B = cv2.imread(path_B, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_AB = np.concatenate([im_A, im_B], 1)
    cv2.imwrite(path_AB, im_AB)

im_list_A = [im_name for im_name in os.listdir(pathA) if im_name.endswith('.png') or im_name.endswith('.tif') or im_name.endswith('.tiff')]
im_list_B = [im_name for im_name in os.listdir(pathB) if im_name.endswith('.png') or im_name.endswith('.tif') or im_name.endswith('.tiff')]
im_list_A.sort()
im_list_B.sort()

check_same_filename(pathA, pathB)

for i in range(len(im_list_A)):
    name_A = im_list_A[i]
    name_B = im_list_B[i]
    path_A = os.path.join(pathA, name_A)
    path_B = os.path.join(pathB, name_B)
    name_AB = name_A
    path_AB = os.path.join(pathAB, name_AB)
    if os.path.isfile(path_A) and os.path.isfile(path_B):
        if not os.path.isfile(path_AB):
            image_write(path_A, path_B, path_AB)
        else:
            print("File exists: ", path_AB)
