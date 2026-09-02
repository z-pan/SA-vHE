import os

from skimage import io

#============================= PARAMS SETUP STARTS ==============================
input_path = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/train_data/set37_paired/B/train"
output_path = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/train_data/set37_paired/B/train"
#=============================  PARAMS SETUP ENDS  ==============================
path_list = os.listdir(input_path)
input_im_list = [im_name for im_name in os.listdir(input_path) if im_name.endswith('.png') or im_name.endswith('.tif') or im_name.endswith('.tiff')]

im_num = 0

for input_im_name in input_im_list:
    
    if ("_HE" not in input_im_name):
        output_im_name = input_im_name.replace("_HE_reg.tif", ".tif")
        input_im_path = input_path + '/' + input_im_name
        output_im_path = output_path + '/' + output_im_name
        os.rename(input_im_path, output_im_path)
        print("Processed: ", input_im_name, " -> ", output_im_name)
    else:
        print("Skipped: ", input_im_name)
        continue

    im_num += 1


def check_same_filename(pathA, pathB):
    path_list_A = os.listdir(pathA)
    path_list_B = os.listdir(pathB)
    path_list_A.sort()
    path_list_B.sort()
    for i in range(len(path_list_A)):
        if path_list_A[i] != path_list_B[i]:
            print("Different: ", path_list_A[i], path_list_B[i])
            return False
    print("Same")
    return True

pathA = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/train_data/set37_paired/A/train"
pathB = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/train_data/set37_paired/B/train"
check_same_filename(pathA, pathB)