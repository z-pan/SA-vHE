# Convert uint16 instance segmentation mask to uint8 binary segmentation mask
import os
import numpy as np
import cv2

#input_uint16_mask_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/train_data/set37/trainA_1channel_nuc_mask"
#output_uint8_mask_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/train_data/set37/trainA_1channel_nuc_mask_bin_uint8"
input_uint16_mask_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/250711_slides/03_og_TPAF_gray_patches_nuc_mask"
output_uint8_mask_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/250711_slides/04_og_TPAF_gray_patches_nuc_mask_bin_uint8"
if not os.path.exists(output_uint8_mask_dir):
    os.makedirs(output_uint8_mask_dir)

input_mask_list = [im_name for im_name in os.listdir(input_uint16_mask_dir) if (im_name.endswith(".tif") or im_name.endswith(".tiff") or im_name.endswith(".png"))]
input_mask_list.sort()
print("Number of input masks: ", len(input_mask_list))

for input_mask_name in input_mask_list:
    input_mask_path = os.path.join(input_uint16_mask_dir, input_mask_name)
    
    # Read the input mask
    input_mask = cv2.imread(input_mask_path, cv2.IMREAD_UNCHANGED)
    
    # skip if the number of instances is less than 10
    if np.max(input_mask) < 15:
        print(f"Skipping {input_mask_name} due to insufficient instances.")
        continue
    
    # Convert uint16 mask to uint8 binary mask
    output_mask = np.zeros_like(input_mask, dtype=np.uint8)
    output_mask[input_mask > 0] = 255
    
    # Smooth the edges of the binary mask
    output_mask = cv2.GaussianBlur(output_mask, (11, 11), 0)
    
    # Save the output mask
    output_mask_path = os.path.join(output_uint8_mask_dir, input_mask_name)
    cv2.imwrite(output_mask_path, output_mask)
    
    print(f"Processed: {input_mask_name}")

