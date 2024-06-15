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

SNRs = [3,5,10,20,40]
methods = ['Raw','NLMEANS','MPPCA','Patch2Self', 'DDM2', 'AVG'] #
methods_names_short =  ['Noisy', 'NLM', 'MPPCA', 'P2S','DDM2', 'AVG']
list_colors = ['gold', 'pink', 'blue', 'darkorange', 'lightseagreen', 'green']
custom_palette = sns.color_palette(list_colors) # To get the same color palette than in seaborn

iter = 20
exp = 'Exp6-data-gaussian'
noise_type = 'gaussian'

dPath = f'/Simulations/Experiments/{exp}'

nrows = len(methods)
ncols = 1
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5,10), sharex=True) 

snr=5
x_limit = 0.8

for i in range(0, nrows):    
    ts_csf = np.loadtxt(f'{dPath}/{methods[i]}/snr{snr}/analysis/dMRI/processed_data/ts_CSF.txt')[3:].ravel() #First 3 columns of the fslmeants output are not of interest 
    ts_csf = ts_csf #[ts_csf>0.001] 
    weights_csf = np.ones_like(ts_csf) / float(len(ts_csf))
    # Histogram
    ax[i].hist(ts_csf, color=custom_palette[i], bins=100, density=True, alpha=0.5)
    # Find the maximum of the histogram
    hist, bin_edges = np.histogram(ts_csf, bins=100) #[ts_csf>10]
    max_bin = np.argmax(hist)
    x_peak = (bin_edges[max_bin] + bin_edges[max_bin+1])/2

    # And plot that and the x=0 as a reference
    ax[i].axvline(x=x_peak, color=list_colors[i], ls='--', lw=2)
    ax[i].axvline(x=0, color='black', ls='--', lw=2)
    ax[i].set_ylabel(methods_names_short[i], fontweight='bold', size=fig.dpi*0.2, color=list_colors[i])    # Size of titles in function of the figsize
    ax[i].set_xlim(right=x_limit)
    
fig.suptitle(f'Noise distribution Dataset SNR={snr}', size=14)
plt.tight_layout()  # Ajustar el espacio entre las subtramas
plt.show()

fig.savefig(f'{dPath}/figures/{noise_type}_{iter}_fig10_snr{snr}_noisefloor_dist.png', dpi=600, bbox_inches='tight')