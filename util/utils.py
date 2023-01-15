import os

import torch
from torch import nn

import numpy as np

from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MeanSquaredError

from matplotlib import pyplot as plt

def normlize_tensor(X):
    x_min = torch.min(X)
    x_max = torch.max(X)
    nom = X - x_min
    denom = x_max - x_min
        
    return nom / (denom + 1.0e-5)

def display_func(display_list, epoch_save_dic=None, epoch=-1):
    psnr, ssim, mse = PeakSignalNoiseRatio(), StructuralSimilarityIndexMeasure(data_range=1.0), MeanSquaredError()
    plt.figure(figsize=(25, 25))
    
    step = display_list[0]

    input_title = "Input Sinogram with Delta theta = {} ".format(step)
    predicted = ""
    
    if len(display_list) > 3 :
        ti = display_list[2].cpu()
        tp = display_list[3].cpu()
        
        psnr_p, ssim_p, rmse_p = psnr(ti, tp), ssim(ti, tp), torch.sqrt(mse(ti, tp))
        
        predicted = "Predicted Object PSNR = {:.2f}, SSIM = {:.3f}, RMSE = {:.4f}".format(psnr_p, ssim_p, rmse_p)

    title = [input_title, 'True Mask', predicted]

    for i in range(1,len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i-1])
        plt.imshow(torch.squeeze(display_list[i]).cpu().numpy(), cmap='gray')
        plt.axis('off')
    
    if epoch == -1:
        plt.show()
    else:    
        plt.savefig(epoch_save_dic+"epoch_{}.png".format(epoch))
