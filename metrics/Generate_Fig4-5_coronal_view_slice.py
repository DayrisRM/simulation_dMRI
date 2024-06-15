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


def make_plotgrid(colnames, ylabels, list_colors, suptitle=None, no_ticks=True, equal_aspect=True):
    """
    Creates a grid of subplots with the specified column names and y-axis labels.

    Parameters:
    colnames (list): List of strings containing the column names for each subplot
    ylabels (list): List of strings containing the y-axis labels for each subplot
    suptitle (str): Title for the entire figure
    no_ticks (bool): True if ticks should be removed from all subplots
    equal_aspect (bool): True if all subplots should have equal aspect ratio

    Returns:
    fig (Figure): The generated figure
    axes (ndarray): Array of axes objects for each subplot
    """

    import matplotlib.pyplot as plt

    plt.style.use('dark_background')

    #if colnames is not None:
    ncols = len(colnames)
    nrows = len(ylabels)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 5*nrows))
    
    for i, title in enumerate(colnames):
        axes[0,i].set_title(f'SNR={title}', fontweight='bold', size=fig.dpi*0.5)    # Size of titles in function of the figsize
    
    for i, ylabel in enumerate(ylabels):        
        axes[i,0].set_ylabel(f'{ylabel}', fontweight='bold', size=fig.dpi*0.5, color=list_colors[i])
        
    axes_list = axes.flatten()
    if equal_aspect:
        for i, ax in enumerate(axes_list):
            ax.set_aspect('equal')

    if no_ticks:
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])    # Remove ticks on all axis in all subplots

    if suptitle:    
        plt.suptitle(suptitle, fontweight='bold', size=fig.dpi*0.4)
    
    
    # Calculate necessary spacing
    left = 0.1 + 0.1 * (ylabels is not None)
    top = 0.6 - 0.1 * (colnames is not None)
    right = 0.6 - 0.1 * (colnames is not None)
    bottom = 0.2 + 0.1 * (ylabels is not None)
    # Adjust spacing to make room for titles and labels
    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top)

    fig.tight_layout()

    return fig, axes

def getVMinMax(method, snr):
    vmin = 0
    vmax = 1 
    return vmin, vmax

def printGroundTruth(ground_truth):    
    plt.imshow(ndimage.rotate(ground_truth[:,42,:,61], 90), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'{dPath}/figures/{iter}_coronal_view_slice_ground-truth_screenshots.png')
    plt.show()


SNRs = [3,5,10,20,40]
methods = ['Raw','NLMEANS','MPPCA','Patch2Self', 'DDM2', 'AVG']
methods_plot_names = ['Noisy','NLMEANS','MPPCA','Patch2Self', 'DDM2', 'AVG']
list_colors = ['gold', 'pink', 'blue', 'darkorange', 'lightseagreen', 'green']
iter = 20
exp = 'Exp6-data-rician'
noise_type = 'Rician'

ground_truth = get_data(f'/Simulations/Experiments/{exp}/Dataset/noisyfree_data_full_b0_first.nii.gz')
dPath = f'/Simulations/Experiments/{exp}'
mask = get_data(f'/Simulations/Experiments/{exp}/Dataset/nodif_brain_mask.nii.gz')

#Get ground-truth
#print('Printing GroundTruth')
#printGroundTruth(ground_truth)

#Get image by methods and snr
colnames = SNRs
ylabels = methods_plot_names

fig, axes = make_plotgrid(colnames, ylabels, list_colors)
   

for ds, method in enumerate(methods):
    print(f'method: {method}')
    for i, ax in enumerate(axes[0]):
        snr = SNRs[i]
        print(f'snr:{snr}')
        name_data = f'{method.lower()}-denoised_main_snr'
        if method == 'Raw':
            name_data = f'main-{noise_type}-noisy_data_snr'
        
        img = get_data(f'{dPath}/{method}/snr{snr}/{name_data}{snr}.nii.gz') #denoised dataset or raw dataset
        vmin, vmax = getVMinMax(method, snr)

        vmin, vmax = getVMinMax(method, snr)
        mask_expanded = np.expand_dims(mask, axis=-1)
        imagex = img * mask_expanded

        image_slice = imagex[:,42,:,61]

        axes[ds,i].imshow(ndimage.rotate(image_slice, 90), cmap='gray', vmin=vmin, vmax=vmax)

        
    print('--------------')
plt.tight_layout()

print('Saving fig')
plt.savefig(f'{dPath}/figures/{iter}_Fig4-5_coronal_view_slice_screenshots.png', dpi=600, bbox_inches='tight')

