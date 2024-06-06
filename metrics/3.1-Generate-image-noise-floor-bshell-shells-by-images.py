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

def calculate_median_noise(denoised_data, bvals, mask):
    #calcular la media por cada b-shell
    indices_b1 = np.where((bvals > 100) & (bvals < 1500))[0]
    indices_b2 = np.where(bvals > 1500)[0]

    masked_data_list_b1 = [denoised_data[:, :, :, i][mask == 1] for i in indices_b1]
    combined_masked_data_b1 = np.concatenate(masked_data_list_b1)
    median_b1 = np.median(combined_masked_data_b1)
    
    masked_data_list_b2 = [denoised_data[:, :, :, i][mask == 1] for i in indices_b2]
    combined_masked_data_b2 = np.concatenate(masked_data_list_b2)
    median_b2 = np.median(combined_masked_data_b2)    

    return median_b1, median_b2
#####

SNRs = [3,5,10,20,40] #3,5,10,20,40
methods = ['Raw','NLMEANS','MPPCA','Patch2Self', 'DDM2', 'AVG']
methods_plot_names = ['Noisy','NLMEANS','MPPCA','Patch2Self', 'DDM2', 'AVG']

iter = 20
exp = 'Exp6-data-gaussian'
noise_type = 'Gaussian'
list_colors = ['gold','deeppink', 'blue', 'darkorange', 'lightseagreen', 'green']


mask = get_data(f'C:/Users/dayri/Documents/UNED/TFM/Related_projects/Simulations/Simulations/Experiments/{exp}/Dataset/mean_f1samples_mask_new.nii.gz')
ground_truth = get_data(f'C:/Users/dayri/Documents/UNED/TFM/Related_projects/Simulations/Simulations/Experiments/{exp}/Dataset/noisyfree_data_full_b0_first.nii.gz')
bvals = np.genfromtxt(f'C:/Users/dayri/Documents/UNED/TFM/Related_projects/Simulations/Simulations/Experiments/{exp}/Dataset/data_b0b1b2k.bval', dtype=np.float32)


# Identificar las posiciones donde la máscara tiene un valor de 1 (dentro de la máscara)
mask_positions = np.where(mask == 1)
dPath = f'C:/Users/dayri/Documents/UNED/TFM/Related_projects/Simulations/Simulations/Experiments/{exp}'

data1k = {}
data2k = {}
for method in methods:
    data1k[method] = []
    data2k[method] = []


for method in methods:
    print(f'method:{method}')
    for snr in SNRs:
        print(f'snr:{snr}')
        name_data = f'{method.lower()}-denoised_main_snr'
        if method == 'Raw':
            name_data = f'main-{noise_type}-noisy_data_snr'

        denoised_data = get_data(f'{dPath}/{method}/snr{snr}/{name_data}{snr}.nii.gz')
                
        median_b1, median_b2 = calculate_median_noise(denoised_data, bvals, mask)
        print(f'noisefloorb1: {median_b1}')
        print(f'noisefloorb2: {median_b2}')
        data1k[method].append(median_b1)
        data2k[method].append(median_b2)


print(data1k)
print(data1k)


# Crear la figura y los ejes
fig, ax = plt.subplots(1, 2)
#markers = ['o', 's'] 

# Graficar los errores de cada método como una línea
method_index = 0
linestyle= ['solid', 'dashdot', 'solid','--', 'solid']
for method in methods:   
   print(method)
   marker='o'
   linestyle='solid'
   if method == 'NLMEANS':
       marker='X'
   if method == 'Patch2Self':
       linestyle='dashdot'


   #gausian noise
   ax[0].plot(SNRs, data1k[method], marker=marker, label=methods_plot_names[method_index], color=list_colors[method_index], linestyle = linestyle)
   ax[1].plot(SNRs, data2k[method], marker='o', label=methods_plot_names[method_index], color=list_colors[method_index], linestyle = linestyle)

   #rician noise
   #ax[0].plot(SNRs, data1k[method], marker='o', label=methods_plot_names[method_index], color=list_colors[method_index])
   #ax[1].plot(SNRs, data2k[method], marker='o', label=methods_plot_names[method_index], color=list_colors[method_index])

   method_index = method_index + 1


# Agregar etiquetas, título y leyenda
ax[0].set_xlabel('SNR')
ax[0].set_ylabel('noise-floor')
ax[0].set_title(r'$b=1000 \, \mathrm{mm/s^2}$')
#ax[0].legend()

ax[1].set_xlabel('SNR')
ax[1].set_ylabel('')
ax[1].set_title(r'$b=2000 \, \mathrm{mm/s^2}$')
ax[1].legend()

fig.suptitle('Noise-Floor', fontsize=14)   

plt.tight_layout()
plt.savefig(f'{dPath}/figures/{noise_type}_{iter}_group_noise_floor_screenshots.png')
# Mostrar la gráfica
plt.show()

