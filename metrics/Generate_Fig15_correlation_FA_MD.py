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
methods_names_short =  ['Noisy','NLM', 'MPPCA', 'P2S','DDM2', 'AVG']
list_colors = ['gold','pink', 'blue', 'darkorange', 'lightseagreen', 'green']
custom_palette = sns.color_palette(list_colors) # To get the same color palette than in seaborn

iter = 21
exp = 'Exp6-data-gaussian'
noise_type = 'Gaussian'
metric = 'MD'

dPath = f'/Simulations/Experiments/{exp}'


data = {}

for method in methods:
    print(f'method:{method}')
    data[method] = []
    for snr in SNRs:
        if (method == 'DDM2' or method == 'Patch2Self') and snr == 3:
            continue  # Omitir el método DDM2 y P2S con snr=3
        print(f'snr:{snr}')
        fsl_cc = np.loadtxt(f'{dPath}/{method}/snr{snr}/analysis/data.dti/fslcc{metric}_res.txt')[2:].ravel() # First 2 columns of the fslcc output are not of interest
        fsl_val = 0
        if fsl_cc.size > 0:
            fsl_val = fsl_cc[0]
        print(f'fsl_cc: {fsl_val}')
        data[method].append((snr, fsl_val))  # Guardar pares (snr, fsl_val)

print(data)

# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(5, 5))

method_index = 0
for method in methods:
    # Filtrar los valores para no incluir snr=3 para los métodos DDM2 y P2S
    if method == 'DDM2' or method == 'Patch2Self':
        filtered_data = [(snr, value) for snr, value in data[method] if snr != 3]
        filtered_snr = [snr for snr, _ in filtered_data]
        filtered_values = [value for _, value in filtered_data]
        ax.plot(filtered_snr, filtered_values, marker='o', label=methods_names_short[method_index], color=list_colors[method_index])
    else:
        snr_values, values = zip(*data[method])
        ax.plot(snr_values, values, marker='o', label=methods_names_short[method_index], color=list_colors[method_index])
    method_index += 1


ax.set_xlabel('SNR')
ax.set_ylabel('Correlation')
plt.title(metric)
ax.legend()

plt.tight_layout()
plt.savefig(f'{dPath}/figures/Fig15_{noise_type}{iter}_correlation_FA_MD_fslcc{metric}_screenshots.png')
# Mostrar la gráfica
plt.show()


