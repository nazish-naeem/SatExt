import sys
sys.path.append(r"/home/t-nnaeem/workspace/RemoteSensingFoundationModels/Image-Super-Resolution-Iterative-Refinement")

from core.metrics import calculate_ssim, calculate_psnr
import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--img1', type=str, default='/home/t-nnaeem/workspace/RemoteSensingFoundationModels/Image-Super-Resolution-Iterative-Refinement/experiments/distributed_high_sr_ffhq_240719_215454/results/4750/190000/3_11_hr.png',
                        help='imag1 path')
parser.add_argument('--img2', type=str, default='/home/t-nnaeem/workspace/RemoteSensingFoundationModels/Image-Super-Resolution-Iterative-Refinement/experiments/distributed_high_sr_ffhq_240719_215454/results/4750/190000/3_11_sr.png',
                        help='imag2 path')
args = parser.parse_args()
img1 = cv2.imread(args.img1)
img2 = cv2.imread(args.img2)

print('SSIM: ', calculate_ssim(img1,img2))
print('PSNR: ', calculate_psnr(img1,img2))