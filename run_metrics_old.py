import os
import sys
import math

import numpy as np
import cv2
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# CHOOSE METRICS TO CALCULATE
#output_metrics = ["PSNR & SSIM", "PSNR & MS-SSIM", "FWHM", "BRISQUE", "ORB similarity", "LPIPS", "FRC"]
#output_metrics = ["PSNR & MS-SSIM"]
#output_metrics = ["PSNR & SSIM"]
output_metrics = ["FID"]
#output_metrics = ["FWHM"]

input_path = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/test_data/set30_23FJP095-1.3.3/same_crop/01_virtual"
gt_path = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/test_data/set30_23FJP095-1.3.3/same_crop/01_real"
output_path = "C:/Users/zpanp/projects/UTOM-master/datasets/AF2HE_datasets/test_data/set30_23FJP095-1.3.3/same_crop"


# Adjust dynamic range (min-max scaling)
def adjust_dynamic_range(image, new_min=0, new_max=255):
    old_min = np.min(image)
    old_max = np.max(image)
    scaled_image = (image - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    return scaled_image.astype(np.uint8)


def PSNR_01(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))


def PSNR_neuroclear(pred, gt):
    # input pred and gt are single channel grayscale np.array dtype=np.uint8
    mse = np.mean((gt - pred)**2)
    if mse > 0:
        return 20 * math.log(255, 10) - 10 * math.log(mse,10)
    else:
        return 20 * math.log(255, 10) - 10 * math.log(0.001,10)


def SSIM_01(pred, gt):
    from skimage.measure import compare_ssim
    return compare_ssim(pred, gt, multichannel=False)


def BRISQUE(pred):
    from image_quality.imquality.brisque import Brisque
    from skimage import io, img_as_float
    pred = img_as_float(pred)
    print("pred dtype: ", pred.dtype)
    print("pred.shape: ", pred.shape)

    # add empty green and blue channels to make it RGB
    green = np.zeros((pred.shape[0], pred.shape[1]))
    blue = np.zeros((pred.shape[0], pred.shape[1]))
    pred = np.stack((pred, green, blue), axis=2)
    print("pred.shape: ", pred.shape)

    return Brisque.score(pred)


def orb_sim(pred, gt):
    # SIFT is no longer available in cv2 so using ORB
    orb = cv2.ORB_create()

    # detect keypoints and descriptors
    kp_a, desc_a = orb.detectAndCompute(pred, None)
    kp_b, desc_b = orb.detectAndCompute(gt, None)

    # define the bruteforce matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # perform matches
    matches = bf.match(desc_a, desc_b)
    # Look for similar regions with distance < 50. Goes from 0 to 100 so pick a number between.
    similar_regions = [i for i in matches if i.distance < 50]  
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)


def LPIPS(pred, gt):
    import lpips
    loss_fn = lpips.LPIPS(net='alex')
    return loss_fn.forward(pred, gt).item()


def FID(pred, gt):
    import fid
    return fid.calculate_fid_given_paths([pred, gt], 50, True, 2048)


def fourier_ring_correlation(image1_fn, image1, image2, pixel_size_nm, output_path):
    FRC_fig_path = os.path.join(output_path, "FRC_fig")
    if not os.path.exists(FRC_fig_path):
        os.makedirs(FRC_fig_path)
    
    # Compute Fourier transforms of the images
    fft1 = np.fft.fftshift(np.fft.fft2(image1))
    fft2 = np.fft.fftshift(np.fft.fft2(image2))

    # Compute cross-power spectrum
    cross_power_spectrum = np.conj(fft1) * fft2

    # Compute radial average of cross-power spectrum
    frc_values = []
    height = image1.shape[0]
    width = image1.shape[1]
    for radius in range(1, min(height, width) // 2):
        y, x = np.ogrid[-height // 2:height // 2, -width // 2:width // 2]
        mask = x**2 + y**2 <= radius**2
        frc_values.append(np.sum(np.abs(cross_power_spectrum[mask])**2))
    
    # Convert spatial frequency to spatial resolution
    radii_nm = np.arange(1, min(height, width) // 2) * pixel_size_nm
    
    # Plot the FRC curve
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    FRC_fig_name = image1_fn.replace(".png", "_FRC.png")
    #plt.plot(frc_values)
    #plt.xlabel("Spatial Frequency")
    plt.plot(radii_nm)
    plt.xlabel("Spatial Resolution (nm)")
    plt.ylabel("FRC Value")
    plt.title("Fourier Ring Correlation (FRC)")
    plt.grid()
    #plt.show()
    plt.savefig(os.path.join(FRC_fig_path, FRC_fig_name))
    plt.close()

    return radii_nm


def calculate_PSNR_MSSSIM(input_path, gt_path, output_path):
    from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
    # use PSNR and MSSSIM functions in skimage library
    print("Calculating PSNR and MSSSIM of:")
    print("input path: ", input_path)
    print("Ground truth path: ", gt_path)
    print("Saving to: ", output_path)

    # Get a list of image files in both input_path and output_path
    input_list = [file for file in os.listdir(input_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    gt_list = [file for file in os.listdir(gt_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print("number of im in input_list: ", len(input_list))
    print("number of im in gt_list: ", len(gt_list))

    # Make sure the lists are of the same length
    if len(input_list) != len(gt_list):
        raise ValueError("Number of input images and output images must be the same.")

    # Initialize lists to store the metrics
    psnr_values = []
    msssim_values = []

    # Calculate metrics for each pair of images
    for input_fn, gt_fn in zip(input_list, gt_list):
        # Load images
        input_im = Image.open(os.path.join(input_path, input_fn))
        gt_im = Image.open(os.path.join(gt_path, gt_fn))

        # Convert images to numpy arrays
        input_arr = np.array(input_im, dtype=np.uint8)
        gt_arr = np.array(gt_im, dtype=np.uint8)
        #input_arr = cv2.equalizeHist(input_arr)
        #gt_arr = cv2.equalizeHist(gt_arr)

        # Calculate PSNR and MSSSIM
        # if the input image is RGB, convert it to grayscale
        if len(input_arr.shape) == 3: # for RGB images
            psnr_value = psnr(input_arr[:,:,0], gt_arr[:,:,0])
        else: # for grayscale images
            psnr_value = psnr(input_arr, gt_arr)
        #psnr_value = PSNR_neuroclear(gt_arr, gt_arr)
        #msssim_value = ssim(input_arr, gt_arr, multichannel=True)  # for RGB images
        #msssim_value = ssim(input_arr[:,:,0], gt_arr[:,:,0], multichannel=False)  # for grayscale images
        # check if the image is RGB or grayscale
        if len(input_arr.shape) == 3:
            msssim_value = ms_ssim(input_arr, gt_arr, data_range=255, size_average=False)
        else:
            msssim_value = ms_ssim(input_arr, gt_arr, data_range=255, size_average=False)

        # Append to the lists
        psnr_values.append(psnr_value)
        msssim_values.append(msssim_value)

    # Calculate the average PSNR and MS-SSIM
    avg_psnr = np.mean(psnr_values)
    avg_msssim = np.mean(msssim_values)

    # Save results to the text file
    PSNR_MSSSIM_out_path = os.path.join(output_path, "PSNR_MSSSIM_results.txt")
    with open(PSNR_MSSSIM_out_path, 'w') as f:
        f.write(f"Average PSNR: {avg_psnr:.2f}\n")
        f.write(f"Average MSSSIM: {avg_msssim:.4f}\n")
        f.write("PSNR and MSSSIM for each image:\n")
        for i, (input_file, output_file) in enumerate(zip(input_list, gt_list)):
            f.write(f"Image {i+1}: {input_file} - PSNR: {psnr_values[i]:.2f}, MSSSIM: {msssim_values[i]:.4f}\n")

    print("Metrics calculation and saving completed.")


def calculate_PSNR_SSIM(input_path, gt_path, output_path):
    # use PSNR and SSIM functions in skimage library
    print("Calculating PSNR and SSIM of:")
    print("input path: ", input_path)
    print("Ground truth path: ", gt_path)
    print("Saving to: ", output_path)

    # Get a list of image files in both input_path and output_path
    input_list = [file for file in os.listdir(input_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    gt_list = [file for file in os.listdir(gt_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print("number of im in input_list: ", len(input_list))
    print("number of im in gt_list: ", len(gt_list))

    # Make sure the lists are of the same length
    if len(input_list) != len(gt_list):
        raise ValueError("Number of input images and output images must be the same.")

    # Initialize lists to store the metrics
    psnr_values = []
    ssim_values = []

    # Calculate metrics for each pair of images
    for input_fn, gt_fn in zip(input_list, gt_list):
        # Load images
        input_im = Image.open(os.path.join(input_path, input_fn))
        gt_im = Image.open(os.path.join(gt_path, gt_fn))

        # Convert images to numpy arrays
        input_arr = np.array(input_im, dtype=np.uint8)
        gt_arr = np.array(gt_im, dtype=np.uint8)
        #input_arr = cv2.equalizeHist(input_arr)
        #gt_arr = cv2.equalizeHist(gt_arr)

        # Calculate PSNR and SSIM
        # if the input image is RGB, convert it to grayscale
        if len(input_arr.shape) == 3: # for RGB images
            psnr_value = psnr(input_arr[:,:,0], gt_arr[:,:,0])
        else: # for grayscale images
            psnr_value = psnr(input_arr, gt_arr)
        #psnr_value = PSNR_neuroclear(gt_arr, gt_arr)
        #ssim_value = ssim(input_arr, gt_arr, multichannel=True)  # for RGB images
        #ssim_value = ssim(input_arr[:,:,0], gt_arr[:,:,0], multichannel=False)  # for grayscale images
        # check if the image is RGB or grayscale
        if len(input_arr.shape) == 3:
            #ssim_value = ssim(input_arr[:,:,0], gt_arr[:,:,0], multichannel=False, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255)
            print("input_arr.shape: ", input_arr.shape)
            print("gt_arr.shape: ", gt_arr.shape)
            ssim_value = ssim(input_arr, gt_arr, channel_axis=2)
        else:
            ssim_value = ssim(input_arr, gt_arr, multichannel=False, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255)
        
        print("psnr_value: ", psnr_value)
        print("ssim_value: ", ssim_value)
        # Append to the lists
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)

    # Calculate the average PSNR and SSIM
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    # Save results to the text file
    PSNR_SSIM_out_path = os.path.join(output_path, "PSNR_SSIM_results.txt")
    with open(PSNR_SSIM_out_path, 'w') as f:
        f.write(f"Average PSNR: {avg_psnr:.2f}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f.write("PSNR and SSIM for each image:\n")
        for i, (input_file, output_file) in enumerate(zip(input_list, gt_list)):
            f.write(f"Image {i+1}: {input_file} - PSNR: {psnr_values[i]:.2f}, SSIM: {ssim_values[i]:.4f}\n")

    print("Metrics calculation and saving completed.")


def calculate_BRISQUE(input_path, gt_path, output_path):
    # use PSNR and SSIM functions in skimage library
    print("Calculating BRISQUE of:")
    print("input path: ", input_path)
    print("Ground truth path: ", gt_path)
    print("Saving to: ", output_path)

    # Get a list of image files in both input_path and output_path
    input_list = [file for file in os.listdir(input_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    gt_list = [file for file in os.listdir(gt_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    print("number of im in input_list: ", len(input_list))
    print("number of im in gt_list: ", len(gt_list))

    # Make sure the lists are of the same length
    if len(input_list) != len(gt_list):
        raise ValueError("Number of input images and output images must be the same.")

    # Initialize lists to store the metrics
    input_brisque_value_list = []
    gt_brisque_value_list = []

    # Calculate metrics for each pair of images
    for input_fn, gt_fn in zip(input_list, gt_list):
        # Load images
        print("input_fn: ", input_fn)
        print("gt_fn: ", gt_fn)
        input_im = Image.open(os.path.join(input_path, input_fn))
        gt_im = Image.open(os.path.join(gt_path, gt_fn))

        # Convert images to numpy arrays
        input_arr = np.array(input_im, dtype=np.uint8)
        gt_arr = np.array(gt_im, dtype=np.uint8)

        # calculate BRISQUE
        input_brisque_value = BRISQUE(input_arr)
        gt_brisque_value = BRISQUE(gt_arr)

        # Append to the lists
        input_brisque_value_list.append(input_brisque_value)
        gt_brisque_value_list.append(gt_brisque_value)

    # Calculate the average PSNR and SSIM
    avg_input_brisque_value = np.mean(input_brisque_value_list)
    avg_gt_brisque_value = np.mean(gt_brisque_value_list)

    # Save results to the text file
    PSNR_SSIM_out_path = os.path.join(output_path, "BRISQUE_results.txt")
    with open(PSNR_SSIM_out_path, 'w') as f:
        f.write(f"Average input BRISQUE score: {avg_input_brisque_value:.2f}\n")
        f.write(f"Average gt BRISQUE score: {avg_gt_brisque_value:.4f}\n")
        f.write("BRISQUE score for each image:\n")
        for i, (input_file, output_file) in enumerate(zip(input_list, gt_list)):
            f.write(f"input image {i+1}: {input_file} - BRISQUE score: {input_brisque_value_list[i]:.2f}\n")
            f.write(f"Ground truth image {i+1}: {output_file} - BRISQUE score: {gt_brisque_value_list[i]:.2f}\n")

    print("Metrics calculation and saving completed.")


def calculate_ORB_similarity(input_path, gt_path, output_path):
    # use orb_sim function
    print("Calculating ORB similarity of:")
    print("input path: ", input_path)
    print("Ground truth path: ", gt_path)
    print("Saving to: ", output_path)

    # Get a list of image files in both input_path and output_path
    input_list = [file for file in os.listdir(input_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    gt_list = [file for file in os.listdir(gt_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    print("number of im in input_list: ", len(input_list))
    print("number of im in gt_list: ", len(gt_list))

    # Make sure the lists are of the same length
    if len(input_list) != len(gt_list):
        raise ValueError("Number of input images and output images must be the same.")
    
    # Initialize lists to store the metrics
    orb_sim_values = []

    # Calculate metrics for each pair of images
    for input_fn, gt_fn in zip(input_list, gt_list):
        # Load images
        print("input_fn: ", input_fn)
        print("gt_fn: ", gt_fn)
        input_im = Image.open(os.path.join(input_path, input_fn))
        gt_im = Image.open(os.path.join(gt_path, gt_fn))

        # Convert images to numpy arrays
        input_arr = np.array(input_im, dtype=np.uint8)
        gt_arr = np.array(gt_im, dtype=np.uint8)

        # calculate ORB similarity
        orb_sim_value = orb_sim(input_arr, gt_arr)

        # Append to the lists
        orb_sim_values.append(orb_sim_value)

    # Calculate the average ORB similarity
    avg_orb_sim = np.mean(orb_sim_values)

    # Save results to the text file
    ORB_sim_out_path = os.path.join(output_path, "ORB_sim_results.txt")

    with open(ORB_sim_out_path, 'w') as f:
        f.write(f"Average ORB similarity: {avg_orb_sim:.4f}\n")
        f.write("ORB similarity for each image:\n")
        for i, (input_file, output_file) in enumerate(zip(input_list, gt_list)):
            f.write(f"Image {i+1}: {input_file} - ORB similarity: {orb_sim_values[i]:.4f}\n")

    print("Metrics calculation and saving completed.")


def calculate_FRC(input_path, gt_path, output_path, pixel_size_nm):

    print("Calculating FRC of:")
    print("input path: ", input_path)
    print("Ground truth path: ", gt_path)
    print("Saving to: ", output_path)

    # Get a list of image files in both input_path and output_path
    input_list = [file for file in os.listdir(input_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    gt_list = [file for file in os.listdir(gt_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Make sure the lists are of the same length
    if len(input_list) != len(gt_list):
        raise ValueError("Number of input images and output images must be the same.")

    # Initialize lists to store the metrics
    frc_values = []

    # Calculate metrics for each pair of images
    for input_fn, gt_fn in zip(input_list, gt_list):
        # Load images
        input_im = Image.open(os.path.join(input_path, input_fn))
        gt_im = Image.open(os.path.join(gt_path, gt_fn))

        # Convert images to numpy arrays
        input_arr = np.array(input_im)
        gt_arr = np.array(gt_im)

        # Calculate FRC
        frc_value = fourier_ring_correlation(input_fn, input_arr, gt_arr, pixel_size_nm, output_path)

        # Append to the lists
        frc_values.append(frc_value)

    # Calculate the average FRC
    avg_frc = np.mean(frc_values)

    # Save results to the text file
    FRC_out_path = os.path.join(output_path, "FRC_results.txt")
    with open(FRC_out_path, 'w') as f:
        f.write(f"Average FRC: {avg_frc:.4f}\n")
        f.write("FRC for each image:\n")
        for i, (input_file, output_file) in enumerate(zip(input_list, gt_list)):
            #f.write(f"Image {i+1}: {input_file} - FRC: {frc_values[i]:.4f}\n")
            f.write(f"Image {i+1}: {input_file} - FRC: {frc_values[i]}\n")

    print("Metrics calculation and saving completed.")

if "PSNR & SSIM" in output_metrics:
    print("Calculating PSNR and SSIM...")
    calculate_PSNR_SSIM(input_path, gt_path, input_path)

if "PSNR & MS-SSIM" in output_metrics:
    print("Calculating PSNR and MS-SSIM...")
    calculate_PSNR_MSSSIM(input_path, gt_path, input_path)

if "FWHM" in output_metrics:
    print("Calculating FWHM...")
    #calculate_FWHM(input_path, gt_path, input_path)

if "BRISQUE" in output_metrics:
    print("Calculating BRISQUE...")
    calculate_BRISQUE(input_path, gt_path, input_path)

if "ORB similarity" in output_metrics:
    print("Calculating ORB similarity...")
    calculate_ORB_similarity(input_path, gt_path, input_path)

if "LPIPS" in output_metrics:
    print("Calculating LPIPS...")
    #calculate_LPIPS(input_path, gt_path, input_path)

if "FRC" in output_metrics:
    print("Calculating FRC...")
    pixel_size_nm = 5.0
    calculate_FRC(input_path, gt_path, input_path, pixel_size_nm)

if "FID" in output_metrics:
    print("Calculating FID...")
    calculate_FID(input_path, gt_path, input_path)

