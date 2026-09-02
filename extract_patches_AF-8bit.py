# Preprocessing of training and testing data: cropping from 1024*1024 tp 512*512, grayscale conversion, and image scaling

import os

import numpy as np
import tifffile
import PIL.Image as Image
import cv2
import imagecodecs

#data_type = "AF - train"
data_type = "AF - test"
#data_type = "AF - test mask"
#data_type = "HE - train"
#data_type = "HE - test"

#dataset_name = "set37" # training data
dataset_name = "250711_slides" # testing data
if_data_augment = False

if data_type == "AF - train":
    print("Processing AF - train images in " + dataset_name)
    input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/train_data/" + dataset_name + "/original_AF_RGB_tif"
    #input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/train_data/" + dataset_name + "/00_og_AF"
    output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/train_data/" + dataset_name + "/01_AF_RGB_scale1.0"
    #output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/train_data/" + dataset_name + "/A"
    input_x = 1024
    input_y = 1024
    output_x = 512
    output_y = 512
    scaling_rate = 1.0 # meaning each [output size * scaling rate] size of area will be resized to output size of output
    # ovarian data scaling 0.4
    if_convert_to_grayscale = False
    if_smooth = True
    if_enhance_contrast = False

elif data_type == "HE - train":
    print("Processing HE - train images in " + dataset_name)
    input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/train_data/" + dataset_name + "/original_HE_tif"
    output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/train_data/" + dataset_name + "/01_HE_scale1.0"
    input_x = 5440
    input_y = 3648
    output_x = 512
    output_y = 512
    scaling_rate = 2.5
    if_convert_to_grayscale = False
    if_smooth = False
    if_enhance_contrast = False

elif data_type == "AF - test":
    print("Processing AF - test images in " + dataset_name)
    input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/" + dataset_name + "/01_og_real_HE_deconv_H_full"
    output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/" + dataset_name + "/02_og_real_HE_deconv_H_patches"
    #input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/250523_slides_with_nuclei_enhance/02_nuclei_enhance"
    #output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/250523_slides_with_nuclei_enhance/testA"
    input_x = 1024
    input_y = 1024
    output_x = 512
    output_y = 512
    scaling_rate = 0.4 # Human ovarian data 0.25, mouse liver data 0.4
    #scaling_rate = 0.9 # Human ovarian data 0.9, for output_x = 128
    if_convert_to_grayscale = False
    if_smooth = False
    if_enhance_contrast = False

elif data_type == "HE - test":
    print("Processing HE - train images in " + dataset_name)
    input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/" + dataset_name + "/00_og_HE"
    output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/" + dataset_name + "/testB"
    input_x = 5440
    input_y = 3648
    output_x = 512
    output_y = 512
    scaling_rate = 1.0
    if_convert_to_grayscale = False
    if_smooth = False
    if_enhance_contrast = False

elif data_type == "AF - test mask":
    print("Processing AF - test mask images in " + dataset_name)
    input_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/" + dataset_name + "/00_AF_mask_og"
    output_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/" + dataset_name + "/01_AF_mask_tile"
    input_x = 1024
    input_y = 1024
    output_x = 512
    output_y = 512
    scaling_rate = 0.4
    if_convert_to_grayscale = False
    if_smooth = True
    if_enhance_contrast = True

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

crop_x = int(output_x * scaling_rate) # e.g. 512*0.5=256
crop_y = int(output_y * scaling_rate)

output_im_name_prefix_count = 1

input_im_list = [im_name for im_name in os.listdir(input_dir) if im_name.endswith('.tif') or im_name.endswith('.tiff') or im_name.endswith('.png')]
input_im_list.sort()

for input_im_name in input_im_list:
    # read tiff image
    if (input_im_name.endswith('.tif') or input_im_name.endswith('.tiff')): # for image in tiff format
        print("Reading tiff image...")
        input_im = tifffile.imread(os.path.join(input_dir, input_im_name))
        #input_im = tifffile.imread(os.path.join(input_dir, input_im_name), key=0) # sometimes tiff file has multiple pages, read the page needed
        input_im = input_im.astype(np.uint8) # numpy array
        input_im = Image.fromarray(input_im)
    # read png image
    elif (input_im_name.endswith('.png')): # for image in png format
        print("Reading png image...")
        input_im = Image.open(os.path.join(input_dir, input_im_name))

    # convert to grayscale
    if if_convert_to_grayscale:
        print("Converting to grayscale...")
        input_im = input_im.convert('L')

    # extract only the red channel of the "AF - test mask" images
    if data_type == "AF - test mask":
        print("Extracting red channel...")
        input_im = input_im.split()[0]
    
    # divide into smaller images according to the specified input and output x and y, 
    # with input image size [input_x, input_y] and output images of size [output_x, output_y]
    input_x = input_im.size[0]
    input_y = input_im.size[1]
    if scaling_rate == 1.0:
        for i in range(0, input_x-output_x, output_x):
            for j in range(0, input_y-output_y, output_y):
                input_im_crop = input_im.crop((i, j, i + output_x, j + output_y))

                if if_data_augment:
                    input_im_crop_horz_flip = input_im_crop.transpose(Image.FLIP_LEFT_RIGHT)
                    output_im_horz_flip_name = input_im_name[:-4] + '_x' + str(i) + '_y' + str(j) + '_horz_flip.png'
                    input_im_crop_horz_flip.save(os.path.join(output_dir, output_im_horz_flip_name))

                    input_im_crop_vert_flip = input_im_crop.transpose(Image.FLIP_TOP_BOTTOM)
                    output_im_vert_flip_name = input_im_name[:-4] + '_x' + str(i) + '_y' + str(j) + '_vert_flip.png'
                    input_im_crop_vert_flip.save(os.path.join(output_dir, output_im_vert_flip_name))
                    
                    """
                    input_im_crop_rot_90 = input_im_crop.transpose(Image.ROTATE_90)
                    output_im_rot_90_name = input_im_name[:-4] + '_x' + str(i) + '_y' + str(j) + '_rot_90.png'
                    input_im_crop_rot_90.save(os.path.join(output_dir, output_im_rot_90_name))

                    input_im_crop_rot_180 = input_im_crop.transpose(Image.ROTATE_180)
                    output_im_rot_180_name = input_im_name[:-4] + '_x' + str(i) + '_y' + str(j) + '_rot_180.png'
                    input_im_crop_rot_180.save(os.path.join(output_dir, output_im_rot_180_name))

                    input_im_crop_rot_270 = input_im_crop.transpose(Image.ROTATE_270)
                    output_im_rot_270_name = input_im_name[:-4] + '_x' + str(i) + '_y' + str(j) + '_rot_270.png'
                    input_im_crop_rot_270.save(os.path.join(output_dir, output_im_rot_270_name))
                    """

                output_im_name = input_im_name[:-4] + '_x' + str(i) + '_y' + str(j) + '.png'
                input_im_crop.save(os.path.join(output_dir, output_im_name))
    
    else: # resize the image to the specified output size
        for i in range(0, input_x - crop_x + 1, crop_x):
            for j in range(0, input_y - crop_y + 1, crop_y):
                input_im_crop = input_im.crop((i, j, i + crop_x, j + crop_y))
                input_im_crop = input_im_crop.resize((output_x, output_y), Image.BICUBIC)

                if if_data_augment:
                    input_im_crop_horz_flip = input_im_crop.transpose(Image.FLIP_LEFT_RIGHT)
                    output_im_horz_flip_name = input_im_name[:-4] + '_x' + str(i) + '_y' + str(j) + '_horz_flip.png'
                    input_im_crop_horz_flip.save(os.path.join(output_dir, output_im_horz_flip_name))

                    input_im_crop_vert_flip = input_im_crop.transpose(Image.FLIP_TOP_BOTTOM)
                    output_im_vert_flip_name = input_im_name[:-4] + '_x' + str(i) + '_y' + str(j) + '_vert_flip.png'
                    input_im_crop_vert_flip.save(os.path.join(output_dir, output_im_vert_flip_name))

                    """
                    input_im_crop_rot_90 = input_im_crop.transpose(Image.ROTATE_90)
                    output_im_rot_90_name = input_im_name[:-4] + '_x' + str(i) + '_y' + str(j) + '_rot_90.png'
                    input_im_crop_rot_90.save(os.path.join(output_dir, output_im_rot_90_name))
                    
                    input_im_crop_rot_180 = input_im_crop.transpose(Image.ROTATE_180)
                    output_im_rot_180_name = input_im_name[:-4] + '_x' + str(i) + '_y' + str(j) + '_rot_180.png'
                    input_im_crop_rot_180.save(os.path.join(output_dir, output_im_rot_180_name))

                    input_im_crop_rot_270 = input_im_crop.transpose(Image.ROTATE_270)
                    output_im_rot_270_name = input_im_name[:-4] + '_x' + str(i) + '_y' + str(j) + '_rot_270.png'
                    input_im_crop_rot_270.save(os.path.join(output_dir, output_im_rot_270_name))
                    """

                output_im_name = input_im_name[:-4] + '_x' + str(i) + '_y' + str(j) + '.png'
                input_im_crop.save(os.path.join(output_dir, output_im_name))

    output_im_name_prefix_count += 1