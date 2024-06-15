import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import nibabel as nb
from scipy import ndimage, stats
from scipy.stats import ranksums
import os 
import subprocess
from pathlib import Path
import warnings
import sys
import seaborn as sns
import math
from skimage.metrics import structural_similarity as ssim
from skimage import feature, transform
from skimage.metrics import peak_signal_noise_ratio as psnr

def get_data(file, mmap=True):
    """
    Load NIfTI image data from a file.

    Parameters:
        file (str): The path to the NIfTI file.
        mmap (bool, optional): Whether to use memory-mapped file access. Default is True.

    Returns:
        numpy.ndarray: The voxel data from the NIfTI file.
    """
    import nibabel as nb
    img = nb.load(file, mmap=mmap)
    img_voxels = img.get_fdata()
    return img_voxels

def plot_gaussian_rician_raw_noise():
    fig,ax = plt.subplots(1,2, sharex=True, sharey=True, figsize=(10, 4))
    for snr in SNRs:
        print(f'snr:{snr}')
        gaussian_noisy_data = get_data(f'{dPath_gaussian}/RAW/snr{snr}/main-Gaussian-noisy_data_snr{snr}.nii.gz')
        gaussian_noisy_data_pos = gaussian_noisy_data[0:13, 0:13, 0:5]
        sns.kdeplot(gaussian_noisy_data_pos.ravel(), ax=ax[0])

        rician_noisy_data = get_data(f'{dPath_rician}/RAW/snr{snr}/main-Rician-noisy_data_snr{snr}.nii.gz')
        rician_noisy_data_pos = rician_noisy_data[0:13, 0:13, 0:5]
        sns.kdeplot(rician_noisy_data_pos.ravel(), ax=ax[1])

    SNRs_leyend = ['SNR=3','SNR=5','SNR=10','SNR=20','SNR=40']

    ax[0].set_title('Gaussian noise')
    ax[1].set_title('Rician noise')

    ax[0].set_yticks([])
    ax[0].set_ylabel('')

    ax[1].set_yticks([])
    ax[1].set_ylabel('')

    plt.legend(SNRs_leyend)
    fig.suptitle('Simulated distributions', fontsize=14)   

    plt.savefig(f'{dPath_gaussian}/figures/{iter}_Fig1_histogram_raw_rician_gaussian_noise_screenshots.png', dpi=600, bbox_inches='tight')
    plt.show()



SNRs = [3,5,10,20,40]
iter = 20

dPath_gaussian = f'/Simulations/Experiments/Exp6-data-gaussian'
dPath_rician = f'/Simulations/Experiments/Exp6-data-rician'


plot_gaussian_rician_raw_noise()



