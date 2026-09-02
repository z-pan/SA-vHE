# change black color to white

import os

import numpy as np
import tifffile
import PIL.Image as Image

input_dir = "C:/Users/zpanp/projects/datasets/virtual_staining/20240817_human-ovarian/HE-240817HOV240827-4/temp"
output_dir = "C:/Users/zpanp/projects/datasets/virtual_staining/20240817_human-ovarian/HE-240817HOV240827-4/temp_out"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

im_list = [im_name for im_name in os.listdir(input_dir) if (im_name.endswith('.tif') or im_name.endswith('.png'))]
im_list.sort()

for im_name in im_list:
    print("Processing: ", im_name)
    # read the tiff image
    if (im_name.endswith('.tif')): # for image in tiff format
        input_im = tifffile.imread(os.path.join(input_dir, im_name))
        input_im = input_im.astype(np.uint8)  # numpy array
        print(input_im.shape)
        input_im = Image.fromarray(input_im)
    elif (im_name.endswith('.png')): # for image in png format
        input_im = Image.open(os.path.join(input_dir, im_name))

    # change black color to white, if red channel is less than 10
    input_im_nparray = np.array(input_im)
    red_channel = input_im_nparray[:,:,0]
    green_channel = input_im_nparray[:,:,1]
    blue_channel = input_im_nparray[:,:,2]
    for i in range(input_im_nparray.shape[0]):
        for j in range(input_im_nparray.shape[1]):
            if red_channel[i,j] < 80:
                red_channel[i,j] = 255
                green_channel[i,j] = 255
                blue_channel[i,j] = 255
    input_im_nparray[:,:,0] = red_channel
    input_im_nparray[:,:,1] = green_channel
    input_im_nparray[:,:,2] = blue_channel
    input_im = Image.fromarray(input_im_nparray)

    # save the image
    input_im.save(os.path.join(output_dir, im_name))
    print("Saved: ", im_name)