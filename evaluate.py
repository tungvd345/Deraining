import os
import argparse
import numpy as np
import math
from math import log10
import time
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


parser = argparse.ArgumentParser(description='Deraining')

# validation data
# parser.add_argument('--data_dir_in', required=False, default="D:/rain_comparison/HeavyRainRemoval_CVPR2019/HeavyRainRemoval-master/out/test_601_700_with_train_param")
parser.add_argument('--data_dir_in', required=False, default="D:/rain_comparison/RCAN/RCAN-master/RCAN_TestCode/SR/BI/RCAN/Heavy_rain_2019/x2")
parser.add_argument('--data_dir_tar', required=False, default='D:/DATASETS/Heavy_rain_image_cvpr2019/test_with_train_param/gt')

args = parser.parse_args()

def evaluate(args):
    path_in = args.data_dir_in
    path_tar = args.data_dir_tar
    file_in = sorted(os.listdir(path_in))
    file_tar = sorted(os.listdir(path_tar))
    len_list_in = len(file_in)

    # calculate PSNR, SSIM
    psnr_avg = 0
    ssim_avg = 0
    for i in range(len_list_in):
        list_in = os.path.join(path_in, file_in[i])
        list_tar = os.path.join(path_tar, file_tar[i//15])
        img_in = cv2.imread(list_in)
        img_tar = cv2.imread(list_tar)

        mse = ((img_in - img_tar) ** 2).mean()
        # psnr_tmp = 10 * log10(255 * 255 / (mse + 10 ** (-10)))
        # psnr_avg += psnr_tmp
        psnr_tmp = psnr(img_in, img_tar, data_range=255)
        psnr_avg += psnr_tmp

        ssim_tmp = ssim(img_in, img_tar, data_range=255, multichannel=True, gaussian_weights=True)
        ssim_avg += ssim_tmp
        print('%s: PSNR = %2.5f, SSIM = %2.5f' % (file_in[i], psnr_tmp, ssim_tmp))

    psnr_avg = psnr_avg / len_list_in
    ssim_avg = ssim_avg / len_list_in
    print('avg psnr = %2.5f, avg SSIM = %1.5f' %(psnr_avg, ssim_avg))

if __name__ == '__main__':
    evaluate(args)