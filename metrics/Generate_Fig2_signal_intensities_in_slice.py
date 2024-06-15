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



SNRs = [5, 10]
iter = 20

dPath_gaussian = f'/Simulations/Experiments/Exp6-data-gaussian'
dPath_rician = f'/Simulations/Experiments/Exp6-data-rician'
ground_truth = get_data(f'/Simulations/Experiments/Exp6-data-gaussian/Dataset/noisyfree_data_full_b0_first.nii.gz')

# Position
x, y, z = 51, 51, 35

ground_truth_signal_timeseries = ground_truth[x, y, z, :]

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 4))
colors = ['orange', 'green']


data_gaussian = {}
data_rician = {}


for snr in SNRs:
    print(f'snr: {snr}')

    # Load noisy data
    gaussian_noisy_data = get_data(f'{dPath_gaussian}/RAW/snr{snr}/main-Gaussian-noisy_data_snr{snr}.nii.gz')
    rician_noisy_data = get_data(f'{dPath_rician}/RAW/snr{snr}/main-Rician-noisy_data_snr{snr}.nii.gz')
    
    #Extract the signal in position
    gaussian_signal_timeseries = gaussian_noisy_data[x, y, z, :]
    rician_signal_timeseries = rician_noisy_data[x, y, z, :]    
    
    data_gaussian[snr] = gaussian_signal_timeseries
    data_rician[snr] = rician_signal_timeseries

ax[0].plot(ground_truth_signal_timeseries, label='noise-free', color='black', linestyle = 'dashed')
ax[1].plot(ground_truth_signal_timeseries, label='noise-free', color='black', linestyle = 'dashed')


snr_index = 0
for snr in SNRs:
    ax[0].plot(data_gaussian[snr], label=f'SNR={snr}', color=colors[snr_index])
    ax[1].plot(data_rician[snr], label=f'SNR={snr}', color=colors[snr_index])
    snr_index = snr_index + 1

# Configurar los gráficos
ax[0].set_title('Gaussian Noisy Data')
ax[0].set_xlabel('')
ax[0].set_ylabel('Signal Intensity')
ax[0].legend()

ax[1].set_title('Rician Noisy Data')
ax[1].set_xlabel('')
ax[1].legend()

plt.tight_layout()

plt.savefig(f'{dPath_gaussian}/figures/{iter}_Fig2_signal_intensities_in_slice_screenshots.png')
plt.show()





