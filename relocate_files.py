# Relocate files with specific string segment in their filenames to another directory

import os
import shutil

# Define the source and destination directories
source_dir = "C:/Users/zpanp/projects/datasets/virtual_staining/20240510_human-ovarian/卵巢癌腹膜切片AF/24FJP006-13 TPAF"
destination_dir = "C:/Users/zpanp/projects/datasets/virtual_staining/20240510_human-ovarian/卵巢癌腹膜切片AF/24FJP006-13 TPAF - not used"
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Define the string segment to search for in the filenames
string_segment = "900nm"

# List all files in the source directory
files = os.listdir(source_dir)

print("Relocating files from {} to {}...".format(source_dir, destination_dir))
print("Files containing '{}' in their filenames will be moved.".format(string_segment))
# Iterate over each file in the source directory
for file in files:
    # Check if the string segment is present in the filename
    if string_segment in file:
        # Construct the source and destination paths
        source_path = os.path.join(source_dir, file)
        destination_path = os.path.join(destination_dir, file)
        # Move the file to the destination directory
        shutil.move(source_path, destination_path)

print("Files relocated successfully.")