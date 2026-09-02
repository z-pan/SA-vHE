# Calculate perception metrics for evaluating virtual staining model: FID, KID, LPIPS, PCC
# Note: there should be multiple real and virtual H&E image pairs in the respective directories

import os
import tempfile
import shutil

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Fix for AttributeError: Can't pickle local object 'make_resizer.<locals>.func'
os.environ["CLEANFID_USE_GPU"] = "0"
os.environ["CLEANFID_USE_MULTIPROCESSING"] = "0"
from cleanfid import fid

def compute_fid_kid(real_dir, fake_dir):
    fid_score = fid.compute_fid(real_dir, fake_dir, num_workers=0)
    kid_score = fid.compute_kid(real_dir, fake_dir, num_workers=0)
    return fid_score, kid_score

def compute_pcc_singlepair(real_img, fake_img):
    # flatten grayscale channels mean over RGB
    real_gray = real_img.mean(axis=2).ravel()
    fake_gray = fake_img.mean(axis=2).ravel()
    r, p = pearsonr(real_gray, fake_gray)
    return r

def perception_metrics_main(real_dir, fake_dir, output_dir):
    # 1. FID & KID via clean-fid (folder-level)
    fid_score, kid_score = compute_fid_kid(real_dir, fake_dir)
    print(f"FID: {fid_score:.4f}, KID: {kid_score:.4f}")

    # 2. PCC per-image
    real_HE_list = [im_name for im_name in os.listdir(real_dir) if (im_name.endswith(".tif") or im_name.endswith(".tiff") or im_name.endswith(".png"))]
    fake_HE_list = [im_name for im_name in os.listdir(fake_dir) if (im_name.endswith(".tif") or im_name.endswith(".tiff") or im_name.endswith(".png"))]
    real_HE_list.sort()
    fake_HE_list.sort()
    assert len(real_HE_list) == len(fake_HE_list), "The number of real and virtual H&E images must be the same."
    print(f"Number of real H&E images: {len(real_HE_list)}, Number of virtual H&E images: {len(fake_HE_list)}")
    pcc_list = []
    for i in range(len(real_HE_list)):
        real = cv2.imread(os.path.join(real_dir, real_HE_list[i]), cv2.IMREAD_UNCHANGED)
        fake = cv2.imread(os.path.join(fake_dir, fake_HE_list[i]), cv2.IMREAD_UNCHANGED)
        pcc = compute_pcc_singlepair(real, fake)
        pcc_list.append(pcc)
    print(f"PCC (mean ± std): {np.mean(pcc_list):.4f} ± {np.std(pcc_list):.4f}")

    # save FID, KID, and PCC numeric results in a text file
    with open(os.path.join(output_dir, 'perception_metrics_results.txt'), 'w') as f:
        f.write(f"FID: {fid_score:.4f}\n")
        f.write(f"KID: {kid_score:.4f}\n")
        f.write(f"PCC (mean ± std): {np.mean(pcc_list):.4f} ± {np.std(pcc_list):.4f}\n")

    # 3. Plotting
    fig, ax = plt.subplots(1, 3, figsize=(18,5))

    # Bar plot for FID & KID
    ax[0].bar(['FID','KID'], [fid_score, kid_score], color=['tab:blue','tab:orange'])
    ax[0].set_title('FID & KID between real & fake')
    ax[0].set_ylabel('Score')

    # Histogram + KDE for PCC distribution
    sns.histplot(pcc_list, bins=20, kde=True, ax=ax[1], color='purple')
    ax[1].set_title('Distribution of Pearson r per-image')
    ax[1].set_xlabel('Pearson r')

    # Scatter plot of PCC vs image index
    ax[2].scatter(range(len(pcc_list)), pcc_list, color='green')
    ax[2].axhline(np.mean(pcc_list), color='red', linestyle='--', label='Mean PCC')
    ax[2].set_title('PCC per image')
    ax[2].set_xlabel('Image index')
    ax[2].set_ylabel('Pearson r')
    ax[2].legend()

    plt.tight_layout()
    output_plot_path = os.path.join(output_dir, 'perception_metrics_plots.png')
    plt.savefig(output_plot_path)
    plt.show()

if __name__ == '__main__':
    
    root_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/250711_slides"
    input_TPAF_dir = os.path.join(root_dir, "01_og_TPAF_RGB_patches")
    real_HE_dir = os.path.join(root_dir, "01_og_real_HE_patches")
    virtual_HE_dir = os.path.join(root_dir, "05_results_vHE_patches_nuc_hi")

    perception_metrics_main(real_HE_dir, virtual_HE_dir, root_dir)