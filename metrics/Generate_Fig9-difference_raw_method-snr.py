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


SNRs = [5]
methods = ['NLMEANS','MPPCA','Patch2Self', 'DDM2', 'AVG']
methods_names = ['NLMEANS','MPPCA','Patch2Self', 'DDM2', 'AVG']
list_colors = ['pink', 'blue', 'darkorange', 'lightseagreen', 'green']

iter = 20
export_figures = True
exp = 'Exp6-data-rician'
noise_type = 'rician'

mask = get_data(f'/Simulations/Experiments/{exp}/Dataset/nodif_brain_mask.nii.gz')
ground_truth = get_data(f'/Simulations/Experiments/{exp}/Dataset/noisyfree_data_full_b0_first.nii.gz')

# Identificar las posiciones donde la máscara tiene un valor de 1 (dentro del cerebro)
mask_positions = np.where(mask == 1)
dPath = f'/Simulations/Experiments/{exp}'


for ds, dataset in enumerate(SNRs):
    nrows = 2
    ncols = len(methods_names)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 6*nrows), sharex='row', sharey='row', gridspec_kw={'height_ratios': [3, 1]})    

    for i in range(0,ncols):
        print(dataset)        
        method = methods_names[i]
        print(method)
        img = get_data(f'{dPath}/{method}/snr{dataset}/differences.nii.gz')  #difference between raw and denoised data
        img_volume = img[:,:,:,69]
        masked_volume = img_volume[mask>0].ravel()          #535k voxels at 1.5mm
        img_slice = img[:,:,36,69]
            
        # Exemplar slice map masked to brain
        masked_slice = img_slice*mask[:,:,36]
        v_max = 3*np.std(masked_slice)
        ax_imshow = ax[0,i]
        ax_imshow.imshow(ndimage.rotate(masked_slice, 90), cmap='gray', vmin=-v_max, vmax=v_max)
        ax_imshow.axis('off')
        ax_imshow.set_title(method, fontweight='bold', size=fig.dpi*0.3, color=list_colors[i])    # Size of titles in function of the figsize
        plt.setp(ax_imshow, xticks=[], yticks=[])

        sns.kdeplot(masked_volume,color=list_colors[i], ax=ax[1,i], fill=True)#, bins=200)
        ax[1,i].axvline(x=np.mean(masked_volume),color=list_colors[i], ls='--', lw=2)
        ax[1,i].axvline(x=0,color='black', ls='--', lw=1)
        #ax[1,i].set_xlim(left=x_limit, right=x_limit)
        ax[1, i].set_xlim(-0.8, 0.8)
        plt.setp(ax[1,i], yticks=[]) 
            
    

    print('----')
    fig.suptitle(f'Dataset SNR={SNRs[ds]}', fontweight='bold', size=fig.dpi*0.4)
    plt.subplots_adjust(wspace=0.0, hspace=-0.25)
    fig.tight_layout()
    plt.show() 
    fig.savefig(f'{dPath}/figures/{noise_type}_{iter}_Fig9_snr{dataset}_screenshots.png', dpi=600, bbox_inches='tight')