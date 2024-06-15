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



iter = 21
exp = 'Exp6-data-rician'
noise_type = 'Rician'


dPath = f'C:/Users/dayri/Documents/UNED/TFM/Related_projects/Simulations/Simulations/Experiments/{exp}'

fsl_cc = np.loadtxt(f'{dPath}/AVG/snr3/analysis/dMRI/processed_data/fslcc_with_noisyfree_res.txt', usecols=2)[3:].ravel() # First 2 columns of the fslcc output are not of interest
median_value = np.median(fsl_cc)
print(f'fslcc_with_noisyfree_res: {median_value}')


fsl_cc_mppca = np.loadtxt(f'{dPath}/MPPCA/snr3/analysis/dMRI/processed_data/fslcc_with_noisyfree_res.txt', usecols=2)[3:].ravel() # First 2 columns of the fslcc output are not of interest
median_value_mppca = np.median(fsl_cc_mppca)
print(f'fslcc_with_noisyfree_res: {median_value_mppca}')

