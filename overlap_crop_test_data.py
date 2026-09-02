# Crop the test images into tiles with overlap
import os

import numpy as np
import cv2
from PIL import Image
import tifffile as tiff

data_type = "AF - test"

dataset_name = "set30_23FJP095-1.3.3"

input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/test_data/" + dataset_name + "/01_AF_png"
output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/test_data/" + dataset_name + "/01_AF_png_mask"

