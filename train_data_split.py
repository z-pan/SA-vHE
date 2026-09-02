# Split an specified percentage of the data into training data

import os
import random
import shutil
from pathlib import Path

#data_type = "AF - train"
data_type = "HE - train"

dataset_name = "set37"

if data_type == "AF - train":
    input_path = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/train_data/" + dataset_name + "/01_AF_RGB_png"
    output_path = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/train_data/" + dataset_name + "/trainA"
    percentage = 70  # Specify the percentage of images to move

elif data_type == "HE - train":
    input_path = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/train_data/" + dataset_name + "/01_HE_png"
    output_path = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/train_data/" + dataset_name + "/trainB"
    percentage = 40  # Specify the percentage of images to move

if not os.path.exists(output_path):
    os.makedirs(output_path)

def split_images(source_dir, train_dir, percentage):

    print(f"Splitting images from {source_dir} into training data at {train_dir}...")

    # Get list of all image files from the source directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    all_images = [file for file in os.listdir(source_dir) if file.lower().endswith(tuple(image_extensions))]
    
    # Calculate the number of images to move based on the percentage
    num_train_images = int(len(all_images) * (percentage / 100))
    
    # Randomly select images to move
    train_images = random.sample(all_images, num_train_images)
    
    # Copy the selected images to the training directory
    for image in train_images:
        src_path = os.path.join(source_dir, image)
        dest_path = os.path.join(train_dir, image)
        #shutil.copy(src_path, dest_path)
        shutil.move(src_path, dest_path)
    
    print(f"Copied {num_train_images} images to the training directory: {train_dir}")


split_images(input_path, output_path, percentage)