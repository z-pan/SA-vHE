# Read WSI files with openSlide, and divide them into patches of specified size

import os

import numpy as np
import cv2
import tifffile as tiff
from openslide import open_slide
import openslide
from matplotlib import pyplot as plt

patch_width = 512
WSI_id = "1427159G-Y"

input_dir = "C:/Users/zpanp/projects/datasets/TOC/e1"
output_dir = "C:/Users/zpanp/projects/datasets/TOC/e1_patches"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the WSI file
try:
    slide = openslide.OpenSlide(os.path.join(input_dir, WSI_id + ".svs"))
    print("Successfully opened the WSI file.")

    # Display basic properties of the slide
    print("Dimensions of the highest resolution level:", slide.dimensions)
    print("Number of pyramid levels:", slide.level_count)
    print("Dimensions of each level:")
    for level in range(slide.level_count):
        print(f"Level {level}: {slide.level_dimensions[level]}")

    # Read a specific region (e.g., top-left 500x500 region of level 0)
    region = slide.read_region((0, 0), level=0, size=(512, 512))
    region.show()  # Display the region (Pillow Image object)

    # Save all regions as 512*512 .png images
    for i in range(0, slide.level_dimensions[0][0], patch_width):
        for j in range(0, slide.level_dimensions[0][1], patch_width):
            region = slide.read_region((i, j), level=0, size=(patch_width, patch_width))
            region.save(os.path.join(output_dir, f"patch_{i}_{j}.png"))

except openslide.OpenSlideError as e:
    print(f"Error opening the WSI file: {e}")
except FileNotFoundError:
    print("The specified file was not found.")

slide = open_slide(os.path.join(input_dir, WSI_id + ".svs"))

slide_props = slide.properties
print(slide_props)

# get slide dimensions for the level 0 - max resolution level
slide_dims = slide.dimensions
print(slide_dims)

"""
#Get a thumbnail of the image and visualize
slide_thumb_600 = slide.get_thumbnail(size=(600, 600))
slide_thumb_600.show()

#Convert thumbnail to numpy array

slide_thumb_600_np = np.array(slide_thumb_600)
plt.figure(figsize=(8,8))
plt.imshow(slide_thumb_600_np)    
"""

#Get slide dims at each level. Remember that whole slide images store information
#as pyramid at various levels
dims = slide.level_dimensions

num_levels = len(dims)
print("Number of levels in this image are:", num_levels)

print("Dimensions of various levels in this image are:", dims)

#By how much are levels downsampled from the original image?
factors = slide.level_downsamples
print("Each level is downsampled by an amount of: ", factors)

#Copy an image from a level
level3_dim = dims[2]
#Give pixel coordinates (top left pixel in the original large image)
#Also give the level number (for level 3 we are providing a valueof 2)
#Size of your output image
#Remember that the output would be a RGBA image (Not, RGB)
level3_img = slide.read_region((0,0), 2, level3_dim) #Pillow object, mode=RGBA

#Convert the image to RGB
level3_img_RGB = level3_img.convert('RGB')
level3_img_RGB.show()

#Convert the image into numpy array for processing
level3_img_np = np.array(level3_img_RGB)
plt.imshow(level3_img_np)


#Return the best level for displaying the given downsample.
SCALE_FACTOR = 32
best_level = slide.get_best_level_for_downsample(SCALE_FACTOR)
#Here it returns the best level to be 2 (third level)
#If you change the scale factor to 2, it will suggest the best level to be 0 (our 1st level)
#################################

#Generating tiles for deep learning training or other processing purposes
#We can use read_region function and slide over the large image to extract tiles
#but an easier approach would be to use DeepZoom based generator.
# https://openslide.org/api/python/

from openslide.deepzoom import DeepZoomGenerator

#Generate object for tiles using the DeepZoomGenerator
tiles = DeepZoomGenerator(slide, tile_size=512, overlap=0, limit_bounds=False)
#Here, we have divided our svs into tiles of size 256 with no overlap. 

#The tiles object also contains data at many levels. 
#To check the number of levels
print("The number of levels in the tiles object are: ", tiles.level_count)

print("The dimensions of data in each level are: ", tiles.level_dimensions)

#Total number of tiles in the tiles object
print("Total number of tiles = : ", tiles.tile_count)

#How many tiles at a specific level?
level_num = 11
print("Tiles shape at level ", level_num, " is: ", tiles.level_tiles[level_num])
print("This means there are ", tiles.level_tiles[level_num][0]*tiles.level_tiles[level_num][1], " total tiles in this level")

#Dimensions of the tile (tile size) for a specific tile from a specific layer
tile_dims = tiles.get_tile_dimensions(11, (3,4)) #Provide deep zoom level and address (column, row)


#Tile count at the highest resolution level (level 16 in our tiles)
tile_count_in_large_image = tiles.level_tiles[16] #126 x 151 (32001/256 = 126 with no overlap pixels)
#Check tile size for some random tile
tile_dims = tiles.get_tile_dimensions(16, (120,140))
#Last tiles may not have full 256x256 dimensions as our large image is not exactly divisible by 256
tile_dims = tiles.get_tile_dimensions(16, (125,150))


single_tile = tiles.get_tile(16, (62, 70)) #Provide deep zoom level and address (column, row)
single_tile_RGB = single_tile.convert('RGB')
single_tile_RGB.show()

###### Saving each tile to local directory
cols, rows = tiles.level_tiles[16]

tile_dir = output_dir
for row in range(rows):
    for col in range(cols):
        tile_name = os.path.join(tile_dir, '%d_%d' % (col, row))
        print("Now saving tile with title: ", tile_name)
        temp_tile = tiles.get_tile(16, (col, row))
        temp_tile_RGB = temp_tile.convert('RGB')
        temp_tile_np = np.array(temp_tile_RGB)
        plt.imsave(tile_name + ".png", temp_tile_np)