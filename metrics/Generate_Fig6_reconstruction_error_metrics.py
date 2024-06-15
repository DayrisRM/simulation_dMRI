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
from sklearn.metrics import mean_squared_error, r2_score

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

def calculate_psnr(difference_in_mask, max_pixel_value):
    mse = np.mean(difference_in_mask ** 2)
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr

def calculate_snr(denoised_data_mask):
    # SNR = mean(signal) / std(noise)
    denoised_data_flat = denoised_data_mask.reshape(-1, denoised_data_mask.shape[-1])    
    noise_std = np.std(denoised_data_flat)  # Calcula la desviación estándar del ruido    
    mean_signal = np.mean(denoised_data_flat)  # Calcula la media de la señal    
    snr = mean_signal / noise_std  # Calcula la SNR
    return snr

def calculate_metric(metric_name, ground_truth, noisy_data, denoised_data):      
    
    if metric_name == 'rmse':
        #se calcula con la imagen original - GroundTruth        
        ground_truth_in_mask = ground_truth[mask_positions]
        denoised_in_mask = denoised_data[mask_positions]

        difference_grount_denoise_in_mask = ground_truth_in_mask - denoised_in_mask

        MSE = np.square(difference_grount_denoise_in_mask).mean()
        RMSE = math.sqrt(MSE)
        return RMSE
    if metric_name == 'psnr': 
        #se calcula con la imagen original - GroundTruth
        ground_truth_in_mask = ground_truth[mask_positions]
        denoised_in_mask = denoised_data[mask_positions]
        difference_grount_denoise_in_mask = ground_truth_in_mask - denoised_in_mask
        max_pixel_value = np.max(difference_grount_denoise_in_mask)        
        psnr = calculate_psnr(difference_grount_denoise_in_mask, max_pixel_value)
        return psnr
    if metric_name == 'ssmi':
        #se calcula con la imagen original - GroundTruth         
        original_image_mask = ground_truth[mask_positions]
        denoised_data_mask = denoised_data[mask_positions]
        ssim_value, _ = ssim(original_image_mask, denoised_data_mask,  data_range=denoised_data_mask.max() - denoised_data_mask.min(), full=True)
        return ssim_value
    if metric_name == 'r2':  #SNR
        ground_truth_in_mask = ground_truth[mask_positions]
        denoised_in_mask = denoised_data[mask_positions]

        ground_truth_flatten = ground_truth_in_mask.flatten()
        denoised_truth_flatten = denoised_in_mask.flatten()    
        r2 = r2_score(ground_truth_flatten, denoised_truth_flatten)
        return r2
    if metric_name == 'snr':  #SNR
        original_image_mask = ground_truth[mask_positions]
        denoised_data_mask = denoised_data[mask_positions]
        snr_val = calculate_snr(denoised_data_mask)
        return snr_val


#####

SNRs = [3,5,10,20,40]
methods = ['NLMEANS','MPPCA','Patch2Self', 'DDM2', 'AVG'] #
method_int = [0, 1, 2, 3]
metric_names = ['rmse', 'r2', 'ssmi'] #, 'psnr', 'snr', 'ssmi', 'r2'
iter = 20
exp = 'Exp6-data-gaussian'
noise_type = 'Gaussian'
list_colors = ['pink', 'blue', 'darkorange', 'lightseagreen', 'green']


mask = get_data(f'/Simulations/Experiments/{exp}/Dataset/nodif_brain_mask.nii.gz')
ground_truth = get_data(f'/Simulations/Experiments/{exp}/Dataset/noisyfree_data_full_b0_first.nii.gz')

# Identify the positions where the mask has value = 1 (inside the brain)
mask_positions = np.where(mask == 1)
dPath = f'/Simulations/Experiments/{exp}'

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# Configurar tamaños de fuente específicos
plt.rcParams.update({
    'font.size': 11,       
    'axes.titlesize': 12, 
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11
})
for ds, metric in enumerate(metric_names):    
    median_difference_methods = []
    for method in methods:
        print(f'method:{method}')
        differencebymethod = []
        for snr in SNRs:
            if (method == 'DDM2' or method == 'Patch2Self') and snr == 3:
                continue  # Omitir el método DDM2 y P2S con snr=3
            print(f'snr:{snr}')
            noisy_data = get_data(f'{dPath}/RAW/snr{snr}/main-{noise_type}-noisy_data_snr{snr}.nii.gz')
            denoised_data = get_data(f'{dPath}/{method}/snr{snr}/{method.lower()}-denoised_main_snr{snr}.nii.gz')
            
            difference = calculate_metric(metric, ground_truth, noisy_data, denoised_data)
            print(f'difference: {difference}')
            differencebymethod.append((snr, difference))  # Guardar pares (snr, diferencia)

        median_difference_methods.append(differencebymethod)

    print(differencebymethod)
    print(median_difference_methods)

    # Graficar los errores de cada método como una línea
    method_index = 0
    for method in methods:
        if method == 'DDM2' or method == 'Patch2Self':
            # Filtrar los valores para no incluir snr=3
            filtered_data = [(snr, difference) for snr, difference in median_difference_methods[method_index] if snr != 3]
            filtered_snr = [snr for snr, _ in filtered_data]
            filtered_differences = [difference for _, difference in filtered_data]
            axs[ds].plot(filtered_snr, filtered_differences, marker='o', label=method, color=list_colors[method_index])
        else:
            snr_values, differences = zip(*median_difference_methods[method_index])
            axs[ds].plot(snr_values, differences, marker='o', label=method, color=list_colors[method_index])
        method_index += 1


   
   
    y_name = metric.upper()
    axs[ds].set_xlabel('SNR')
    axs[ds].set_ylabel('') 
    axs[ds].set_title(f'{y_name}') 

    axs[2].legend()
        
    

plt.tight_layout()
plt.savefig(f'{dPath}/figures/{noise_type}_{iter}_Fig6_reconstruction_error_metrics.png', dpi=600, bbox_inches='tight')
plt.show()


