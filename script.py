import os
#os.system('source activate pytorch')

# Training
# Dataset: Human colon section, input_nc=1
# Model: CycleGAN, saliency loss
#os.system('python train.py --dataroot ./datasets/AF2HE_datasets --name human_colon --model cycle_gan --input_nc 1 --output_nc 3 \
#    --lambda_identity 0.0 --gpu_ids 0 --load_size 512 --crop_size 512 --display_winsize 512')

# Dataset: In vivo mouse liver, input_nc=1
# Model: CycleGAN, saliency loss
#os.system('python train.py --dataroot ./datasets/set05_train --name mouse_liver --model cycle_gan --input_nc 1 --output_nc 3 \
#    --trainA_normalize 255 --lambda_identity 0.0 --gpu_ids 0 --load_size 512 --crop_size 512 --display_winsize 512')

# Dataset: Human ovarian cancer slide, input_nc=3
# Model: CycleGAN, saliency loss
#os.system('python train.py --dataroot ./datasets/train_data/set37 --name set37_temp --model cycle_gan --input_nc 3 --output_nc 3 \
#    --trainA_normalize 255 --lambda_identity 0.0 --gpu_ids 0 --load_size 512 --crop_size 512 --display_winsize 512')

# Dataset: Human ovarian cancer slide, input_nc=1
# Model: CycleGAN, SSIM loss, resnet_6blocks
#os.system('python train.py --dataroot ./datasets/train_data/set30-32/train_data_00_temp --name human_ovarian_ssim_loss --model cycle_gan_ssim_loss --input_nc 1 --output_nc 3 \
#    --trainA_normalize 255 --lambda_identity 0.0 --netG resnet_6blocks --gpu_ids 0 --load_size 512 --crop_size 512 --display_winsize 512')

# Dataset: Human ovarian cancer slide, input_nc=3
# Model: CycleGAN DCL, resnet_9blocks
#print("Training with human set30-32 AF-RGB dataset with DCL model")
#os.system('python train.py --dataroot ./datasets/train_data/set30-32/train_data_AF-RGB --dataset_mode unaligned_DCL \
#          --name set30-32_DCL --model dcl --input_nc 3 --output_nc 3 --netG resnet_9blocks \
#          --trainA_normalize 255 --gpu_ids 0 --load_size 256 --crop_size 256 --display_winsize 512')

# Dataset: Grumpifycat, input_nc=3
# Model: CycleGAN DCL, resnet_9blocks
#print("Training with sample grumpycat train set with DCL model")
#os.system('python train.py --dataroot ./datasets/grumpifycat --dataset_mode unaligned_DCL --model dcl --input_nc 3 --netG resnet_9blocks \
#           --gpu_ids 0 --load_size 256 --crop_size 256 --name grumpycat_DCL')

# Dataset: Human ovarian cancer section, input_nc=3, paired
# Model: Pix2pix baseline, 
#os.system('python train.py --dataroot ./datasets/train_data/set37_paired_final --name set37_paired_final --model pix2pix \
#    --input_nc 3 --output_nc 3 --dataset_mode aligned --direction AtoB --trainA_normalize 255 --gpu_ids 0 --load_size 512 \
#    --crop_size 512 --display_winsize 512')


#====================================================================================================
# Testing commands
# 2024/11/14
# AF -> HE
#print("Saliency loss, AF -> HE")
#os.system('python test.py --dataroot ./datasets/test_data/250529_slides --epoch 80 --direction AtoB \
#           --results_dir  ./datasets/test_data/250529_slides/results \
#           --name set37_train_1channel_saliency_A65_B220 --model cycle_gan --trainA_normalize 255 --trainB_normalize 255 \
#           --input_nc 1 --output_nc 3 --gpu_ids 0 --load_size 512 --crop_size 512 --display_winsize 512 --num_test 999')
# HE -> AF
#print("Saliency loss, HE -> AF")
#os.system('python test.py --dataroot ./datasets/test_data/set37_human_ovarian_HE2AF/Line03-0002 --epoch 100 --direction BtoA \
#           --name set37_train_set37_unpaired2500+paired2500_512_A70_B220_lambda15 --model cycle_gan --trainA_normalize 255 --trainB_normalize 255 \
#           --input_nc 3 --output_nc 3 --gpu_ids 0 --load_size 512 --crop_size 512 --display_winsize 512 --num_test 999')
# AF -> HE 
os.system('python test.py --dataroot ./datasets/test_data/250711_slides --epoch 80 --direction AtoB \
           --results_dir  ./datasets/test_data/250711_slides/05_results_vHE_patches_nuc_hi \
           --name set37_train_1channel_saliency_A65_B220 --model cycle_gan --trainA_normalize 255 --trainB_normalize 255 \
           --input_nc 1 --output_nc 3 --gpu_ids 0 --load_size 512 --crop_size 512 --display_winsize 512 --num_test 999')

# Test with human colon dataset "set01", checkpoints loaded from "--name human_colon" folder
#os.system('python test.py --dataroot ./datasets/test_data/set01 --name temp_human_colon --model cycle_gan \
#          --trainA_normalize 255 --input_nc 1 --output_nc 3 --gpu_ids 0 --load_size 512 --crop_size 512 --display_winsize 512 --num_test 468')
# Test with mouse liver dataset "set02", checkpoints loaded from "--name set03_mouse_liver" folder
#os.system('python test.py --dataroot ./datasets/test_data/test04_240613 --name temp_mouse_liver --model cycle_gan \
#           --trainA_normalize 255 --input_nc 1 --output_nc 3 --gpu_ids 0 --load_size 512 --crop_size 512 --display_winsize 512 --num_test 60')
#os.system('python test.py --dataroot ./datasets/test_data/nuc_enhance_test2 --name temp_mouse_liver --model cycle_gan \
#           --trainA_normalize 255 --input_nc 1 --output_nc 3 --gpu_ids 0 --load_size 512 --crop_size 512 --display_winsize 512 --num_test 60')
# Test with human ovarian dataset "set30", checkpoints loaded from "--name temp_human_ovarian" folder
#os.system('python test.py --dataroot ./datasets/test_data/240629_01 --name temp_human_ovarian --model cycle_gan \
#           --trainA_normalize 255 --input_nc 1 --output_nc 3 --gpu_ids 0 --load_size 512 --crop_size 512 --display_winsize 512 --num_test 125')
# Test with human ovarian dataset "set30" for SSIM loss, checkpoints loaded from "--name temp_human_ovarian" folder
#os.system('python test.py --dataroot ./datasets/test_data/set33 --name temp_human_ovarian_ssim_loss --model cycle_gan_ssim_loss \
#           --netG resnet_6blocks --trainA_normalize 255 --input_nc 1 --output_nc 3 --gpu_ids 0 --load_size 512 --crop_size 512 --display_winsize 512 --num_test 425')

# Test with human ovarian dataset "set30-32", checkpoints loaded from "--name temp_human_ovarian" folder
#print("Testing with set30-32 AF-RGB on unet256 backbone with saliency loss...")
#os.system('python test.py --dataroot ./datasets/test_data/set37_human_ovarian/Line11 --epoch 80 \
#           --name set37_train_1channel_saliency_A65_B220 --model cycle_gan --trainA_normalize 255 --input_nc 1 --output_nc 3 \
#           --gpu_ids 0 --load_size 512 --crop_size 512 --display_winsize 512 --num_test 175')

# Test with human ovarian dataset "set30" for SSIM loss, checkpoints loaded from "--name temp_human_ovarian" folder
#print("Testing with set30-32 AF-RGB on resnet_6blocks backbone with SSIM loss...")
#os.system('python test.py --dataroot ./datasets/test_data/set37_human_ovarian/Line11 --epoch 120 --name set30-32_train_02_AF-RGB_ssim_loss --model cycle_gan_ssim_loss \
#           --netG resnet_6blocks --trainA_normalize 255 --input_nc 3 --output_nc 3 --gpu_ids 0 --load_size 512 --crop_size 512 \
#           --display_winsize 512 --num_test 64')

#os.system('python test.py --dataroot ./datasets/1123_lung --name temp_human_colon --model cycle_gan \
#          --input_nc 1 --output_nc 3 --gpu_ids 0 --load_size 512 --crop_size 512 --display_winsize 512 --num_test 32')

# Testing with Pix2pix model trained with paired dataset
#os.system('python test.py --dataroot ./datasets/test_data/set37_human_ovarian/Line11 --direction AtoB --epoch 200 \
#          --trainA_normalize 255 --input_nc 3 --output_nc 3 --gpu_ids 0 --model pix2pix --name set37_paired_final \
#          --num_test 448')
