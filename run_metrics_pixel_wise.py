# Calculate pixel-wise metrics for evaluating virtual staining model
# Note: there should be multiple real and virtual H&E image pairs in the respective directories
import os

import cv2
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.metrics import (
    mean_squared_error,
    normalized_root_mse,
    peak_signal_noise_ratio,
    structural_similarity
)
from pytorch_msssim import ms_ssim
import torch

def compute_metrics(real_HE_im, virtual_HE_im):
    # Assume uint8 or float [0,1]
    data_range = 255 if real_HE_im.dtype == np.uint8 else 1.0

    mse = mean_squared_error(real_HE_im, virtual_HE_im)
    rmse = np.sqrt(mse)
    nrmse = normalized_root_mse(real_HE_im, virtual_HE_im, normalization='euclidean')
    psnr = peak_signal_noise_ratio(real_HE_im, virtual_HE_im, data_range=data_range)
    min_dim = min(real_HE_im.shape[0], real_HE_im.shape[1])
    win_size = min(7, min_dim) if min_dim >= 3 else min_dim  # Ensure win_size is odd and <= min_dim
    if win_size % 2 == 0:
        win_size -= 1
    ssim = structural_similarity(
        real_HE_im, virtual_HE_im, data_range=data_range, channel_axis=-1, win_size=win_size
    )

    # MS-SSIM via pytorch, convert to tensor
    gt_t = torch.from_numpy(real_HE_im.transpose(2,0,1)[None,...]).float()
    fake_t = torch.from_numpy(virtual_HE_im.transpose(2,0,1)[None,...]).float()
    ms_ssim_val = ms_ssim(gt_t, fake_t, data_range=data_range, size_average=True).item()

    return mse, rmse, nrmse, psnr, ssim, ms_ssim_val

def pixel_wise_metrics_main(real_HE_dir, fake_HE_dir, output_dir):
    metrics = {k: [] for k in ['MSE','RMSE','NRMSE','PSNR','SSIM','MS-SSIM']} # Initialize metrics dictionary
    
    real_HE_list = [im_name for im_name in os.listdir(real_HE_dir) if (im_name.endswith(".tif") or im_name.endswith(".tiff") or im_name.endswith(".png"))]
    fake_HE_list = [im_name for im_name in os.listdir(fake_HE_dir) if (im_name.endswith(".tif") or im_name.endswith(".tiff") or im_name.endswith(".png"))]
    real_HE_list.sort()
    fake_HE_list.sort()
    assert len(real_HE_list) == len(fake_HE_list), "The number of real and virtual H&E images must be the same."
    print(f"Number of real H&E images: {len(real_HE_list)}, Number of virtual H&E images: {len(fake_HE_list)}")

    for i in range(len(real_HE_list)):
        real_HE_im = imread(os.path.join(real_HE_dir, real_HE_list[i]))
        virtual_HE_im = imread(os.path.join(fake_HE_dir, fake_HE_list[i]))
        # convert to float [0,1] for ss
        if real_HE_im.dtype == np.uint8:
            real_HE_im = real_HE_im.astype(np.float32) #/ 255.0
            virtual_HE_im = virtual_HE_im.astype(np.float32) #/ 255.0
        vals = compute_metrics(real_HE_im, virtual_HE_im)
        for k,v in zip(metrics.keys(), vals): # calculate every metric
            metrics[k].append(v)

    # Print mean and std for each metric, and store in a txt file
    metrics_txt_path = os.path.join(output_dir, "pixel_wise_metrics_results.txt")
    with open(metrics_txt_path, "w") as f:
        f.write("Metrics Summary:\n")
        for k in metrics:
            mean_val = np.mean(metrics[k])
            std_val = np.std(metrics[k])
            f.write(f"{k}: mean={mean_val:.4f}, std={std_val:.4f}\n")
            print(f"{k}: mean={mean_val:.4f}, std={std_val:.4f}")

    # Draw boxplots for each metric
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()
    for ax, k in zip(axs, metrics.keys()):
        ax.boxplot(metrics[k])
        ax.set_title(k)
        ax.set_ylabel(k)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pixel_wise_metrics_boxplot.png"))
    plt.show()

if __name__ == "__main__":
    
    root_dir = "C:/Users/zpanp/projects/UTOM-master/datasets/test_data/250711_slides"
    input_TPAF_dir = os.path.join(root_dir, "01_og_TPAF_RGB_patches")
    real_HE_dir = os.path.join(root_dir, "01_og_real_HE_patches")
    virtual_HE_dir = os.path.join(root_dir, "05_results_vHE_patches_nuc_hi")

    pixel_wise_metrics_main(real_HE_dir, virtual_HE_dir, root_dir)