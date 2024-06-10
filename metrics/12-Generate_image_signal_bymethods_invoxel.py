import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import nibabel as nb
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

def get_voxel_coords(voxel_type):
    '''
    mean_f1samples (f1), mean_f2samples (f2), mean_f3samples (f3). 
    Para 1fib, coge cualquier voxel donde f2 y f3 <0.05
    Para 2fib, cualquier voxel donde f3<0.05 y f2>0.05
    Para 3fib, cualquier voxel donde f3>0.05.

    Los valores que se usan cumplen los requisitos de arriba.
    '''
    if voxel_type == '1fib':
        return (39, 71, 47)
    elif voxel_type == '2fib':
        return (36, 25, 29)
    elif voxel_type == '3fib':
        return (76, 35, 29)
    else:
        raise ValueError("Unknown voxel type")


snr = 5 #
methods = ['Raw','Patch2Self', 'DDM2'] #'Raw','NLMEANS','MPPCA','Patch2Self', 'DDM2', 'AVG'
methods_plot_names = ['Noisy','Patch2Self', 'DDM2'] #'Noisy','NLMEANS','MPPCA','Patch2Self', 'DDM2', 'AVG'
list_colors = ['gold', 'darkorange', 'lightseagreen', 'green']#'gold', 'pink', 'blue', 'darkorange', 'lightseagreen', 'green'

iter = 20
exp = 'Exp6-data-rician'
noise_type = 'Rician'
voxel_type = '1fib'

x, y, z = get_voxel_coords(voxel_type)

ground_truth = get_data(f'C:/Users/dayri/Documents/UNED/TFM/Related_projects/Simulations/Simulations/Experiments/{exp}/Dataset/noisyfree_data_full_b0_first.nii.gz')
dPath = f'C:/Users/dayri/Documents/UNED/TFM/Related_projects/Simulations/Simulations/Experiments/{exp}'

signal_timeseries_ground_truth = ground_truth[x, y, z]

data_in_voxel = {}

for ds, method in enumerate(methods):
    print(f'method: {method}')
    name_data = f'{method.lower()}-denoised_main_snr'
    if method == 'Raw':
        name_data = f'main-{noise_type}-noisy_data_snr'
        
    img = get_data(f'{dPath}/{method}/snr{snr}/{name_data}{snr}.nii.gz') #esta es la imagen denoised o el raw 
    signal_timeseries = img[x, y, z]
    data_in_voxel[method] = signal_timeseries


fig, ax = plt.subplots(figsize=(5, 5))

ax.plot(signal_timeseries_ground_truth, label='noise free', color='black', linestyle = 'dashed')

# Graficar las series temporales
method_index = 0
for method in methods:
    ax.plot(data_in_voxel[method], label=methods_plot_names[method_index], color=list_colors[method_index])
    method_index = method_index +1
   

plt.xlabel('Volumes')
plt.ylabel('Signal intensity')
plt.legend()
plt.title(f'SNR = {snr}')

plt.savefig(f'{dPath}/figures/{noise_type}_{voxel_type}_{iter}_signal_bymethod_invoxel_snr{snr}_screenshots.png', dpi=600, bbox_inches='tight')
plt.show()

