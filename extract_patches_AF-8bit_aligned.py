# Preprocessing of training and testing data: cropping, grayscale conversion, and color inversion

import os

import numpy as np
import tifffile
import PIL.Image as Image

if_convert_to_grayscale = False
if_color_inversion = False
if_data_augment = False

patch_size = 128

input_dir = "C:/Users/zpanp/projects/datasets/virtual_staining/20240817_human-ovarian/HE-240817HOV240827-4/registered"
output_dir = "C:/Users/zpanp/projects/datasets/virtual_staining/20240817_human-ovarian/HE-240817HOV240827-4/registered_patch"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

AF_im_list = [im_name for im_name in os.listdir(input_dir) if (im_name.endswith('_AF.tif') or im_name.endswith('_AF.png'))]
AF_im_list.sort()
HE_im_list = [im_name for im_name in os.listdir(input_dir) if (im_name.endswith('_HE_reg.tif') or im_name.endswith('_HE_reg.png'))]
HE_im_list.sort()

print("Number of AF images: ", len(AF_im_list))
print("Number of HE images: ", len(HE_im_list))

def data_augment(im):
    im_horz_flip = im.transpose(Image.FLIP_LEFT_RIGHT)
    im_vert_flip = im.transpose(Image.FLIP_TOP_BOTTOM)
    im_rot_90 = im.transpose(Image.ROTATE_90)
    im_rot_180 = im.transpose(Image.ROTATE_180)
    im_rot_270 = im.transpose(Image.ROTATE_270)
    return [im, im_horz_flip, im_vert_flip, im_rot_90, im_rot_180, im_rot_270]

for AF_im_name in AF_im_list:
    HE_im_name = AF_im_name.replace('_AF.tif', '_HE_reg.tif')
    print("Processing: ", AF_im_name, " and ", HE_im_name)
    # read the tiff image
    if (AF_im_name.endswith('.tif')): # for image in tiff format
        AF_input_im = tifffile.imread(os.path.join(input_dir, AF_im_name))
        HE_input_im = tifffile.imread(os.path.join(input_dir, HE_im_name))
        HE_input_im = HE_input_im[0,:,:,:]
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
    if if_convert_to_grayscale:
        AF_input_im = AF_input_im.convert('L')

    if if_color_inversion:
        AF_input_im = Image.eval(AF_input_im, lambda x: 255 - x)

    # image height and width
    AF_im_height, AF_im_width = AF_input_im.size
    HE_im_height, HE_im_width = HE_input_im.size

    assert AF_im_height == HE_im_height and AF_im_width == HE_im_width, "AF and HE images have different sizes!"
    
    # divide into smaller images according to the specified input and output x and y, 
    # with input image size [input_x, input_y] and output images of size [output_x, output_y]
    for i in range(0, AF_im_height-patch_size, patch_size):
        for j in range(0, AF_im_width-patch_size, patch_size):
            AF_im_crop = AF_input_im.crop((i, j, i + patch_size, j + patch_size))
            HE_im_crop = HE_input_im.crop((i, j, i + patch_size, j + patch_size))

            if if_data_augment:
                AF_im_augment_list = data_augment(AF_im_crop)
                HE_im_augment_list = data_augment(HE_im_crop)
            
            im_prefix = AF_im_name[:-len('_AF.tif')] + '_x' + str(i) + '_y' + str(j)
            AF_output_im_name = im_prefix + '_AF.tif'
            HE_output_im_name = im_prefix + '_HE_reg.tif'
            AF_im_crop.save(os.path.join(output_dir, AF_output_im_name))
            HE_im_crop.save(os.path.join(output_dir, HE_output_im_name))