# Generate the optical redox ratio (ORR) images from AF images
# Formula of calculating ORR from FAD and NADH AF signals:
# ORR = FAD / (FAD + NADH)

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tifffile

input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/train_data/mouse_in-vivo_liver/00_og_RGB"
output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/train_data/mouse_in-vivo_liver/01_ORR"
heatmap_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/train_data/mouse_in-vivo_liver/02_heatmap"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(heatmap_dir):
    os.makedirs(heatmap_dir)

AF_im_list = [im_name for im_name in os.listdir(input_dir) if (im_name.endswith('.tif') or im_name.endswith('.png'))]
AF_im_list.sort()

print("Number of AF images: ", len(AF_im_list))

for AF_im_name in AF_im_list:
    print("Processing: ", AF_im_name)
    # read the tiff image
    if (AF_im_name.endswith('.tif')): # for image in tiff format
        AF_input_im = tifffile.imread(os.path.join(input_dir, AF_im_name))
        AF_input_im = AF_input_im.astype(np.uint8)  # numpy array
        print(AF_input_im.shape)
        #AF_input_im = Image.fromarray(AF_input_im)
    elif (AF_im_name.endswith('.png')): # for image in png format
        AF_input_im = Image.open(os.path.join(input_dir, AF_im_name))

    # FAD signal is the red channel, NADH signal is the green channel
    FAD_im = AF_input_im[:,:,0].astype(np.float32)
    NADH_im = AF_input_im[:,:,1].astype(np.float32)
    #orr_im = Image.eval(FAD_im, lambda x: x / (x + NADH_im.getpixel((x, x))))
    epsilon = 1e-10
    orr_im = FAD_im / (NADH_im + FAD_im + epsilon)
    
    # save the ORR image
    orr_save_path = os.path.join(output_dir, AF_im_name.replace('.tif', '_ORR.png'))

    """
    orr_im = (orr_im * 255).astype(np.uint8)  # convert to uint8
    orr_im = Image.fromarray(orr_im)
    orr_im.save(os.path.join(output_dir, AF_im_name.replace('.tif', '_ORR.tif')))
    """
    plt.imsave(orr_save_path, orr_im, cmap='hot')

    print("Saved: ", AF_im_name.replace('.tif', '_ORR.png'))

    # create a heatmap of the ORR image, with high value assigned to and save to the heatmap directory