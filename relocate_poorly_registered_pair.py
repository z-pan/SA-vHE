# Convert paired RGB images to grayscale images, calculate SSIM, and move pairs with low SSIM to another folder

import os

import numpy as np
import tifffile
import PIL.Image as Image

input_dir = "C:/Users/zpanp/projects/datasets/virtual_staining/20240817_human-ovarian/HE-240817HOV240827-4/registered_patch"
output_dir = "C:/Users/zpanp/projects/datasets/virtual_staining/20240817_human-ovarian/HE-240817HOV240827-4/poorly_registered_patch"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

AF_im_list = [im_name for im_name in os.listdir(input_dir) if (im_name.endswith('_AF.tif') or im_name.endswith('_AF.png'))]
AF_im_list.sort()
HE_im_list = [im_name for im_name in os.listdir(input_dir) if (im_name.endswith('_HE_reg.tif') or im_name.endswith('_HE_reg.png'))]
HE_im_list.sort()

print("Number of AF images: ", len(AF_im_list))
print("Number of HE images: ", len(HE_im_list))

for AF_im_name in AF_im_list:
    HE_im_name = AF_im_name.replace('_AF.tif', '_HE_reg.tif')
    print("Processing: ", AF_im_name, " and ", HE_im_name)
    # read the tiff image
    if (AF_im_name.endswith('.tif')): # for image in tiff format
        AF_input_im = tifffile.imread(os.path.join(input_dir, AF_im_name))
        HE_input_im = tifffile.imread(os.path.join(input_dir, HE_im_name))
        AF_input_im = AF_input_im.astype(np.uint8)  # numpy array
        HE_input_im = HE_input_im.astype(np.uint8)  # numpy array
        print(AF_input_im.shape)
        print(HE_input_im.shape)
        AF_input_im = Image.fromarray(AF_input_im)
        HE_input_im = Image.fromarray(HE_input_im)
    elif (HE_im_name.endswith('.png')): # for image in png format
        AF_input_im = Image.open(os.path.join(input_dir, AF_im_name))
        HE_input_im = Image.open(os.path.join(input_dir, HE_im_name))

    # convert to grayscale and invert the color
    AF_input_im = AF_input_im.convert('L')
    AF_input_im = Image.eval(AF_input_im, lambda x: 255 - x)
    HE_input_im = HE_input_im.convert('L')

    # calculate SSIM
    from skimage.metrics import structural_similarity as ssim
    AF_input_im_nparray = np.array(AF_input_im)
    HE_input_im_nparray = np.array(HE_input_im)
    ssim_val = ssim(AF_input_im_nparray, HE_input_im_nparray)
    print("SSIM: ", ssim_val)

    if ssim_val < 0.5:
        os.rename(os.path.join(input_dir, AF_im_name), os.path.join(output_dir, AF_im_name))
        os.rename(os.path.join(input_dir, HE_im_name), os.path.join(output_dir, HE_im_name))
        print("Moved to poorly_registered_patch: ", AF_im_name, " and ", HE_im_name)
    else:
        print("SSIM is greater than 0.5, keeping: ", AF_im_name, " and ", HE_im_name)
        continue