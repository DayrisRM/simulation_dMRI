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


snr = 10
iter = 20
method = 'DDM2'

dPath_gaussian = f'C:/Users/dayri/Documents/UNED/TFM/Related_projects/Simulations/Simulations/Experiments/Exp6-data-gaussian'

exec1_ddm2_data = get_data(f'{dPath_gaussian}/{method}/snr{snr}/diff_executions/exec1_hardi150_denoised.nii.gz')
exec2_ddm2_data = get_data(f'{dPath_gaussian}/{method}/snr{snr}/diff_executions/exec2_hardi150_denoised.nii.gz')
diff_ddm2_data = get_data(f'{dPath_gaussian}/{method}/snr{snr}/diff_executions/diff_executions.nii.gz')

voxel1_coords = (62, 51, 40)
voxel2_coords = (43, 51, 30)

diff_ddm2_data_voxel1 = diff_ddm2_data[62, 51, 40]
diff_ddm2_data_voxel2 = diff_ddm2_data[43, 51, 30]

plt.figure(figsize=(10, 6))

plt.plot(diff_ddm2_data_voxel1, 'o-', label=f'Voxel 1 - coords:{voxel1_coords}')
plt.plot(diff_ddm2_data_voxel2, 'o-', label=f'Voxel 2 - coords {voxel2_coords}')

plt.title('Individual voxel signal in the differential')
plt.xlabel('Volumes')
plt.ylabel('Signal')
plt.legend()

plt.savefig(f'{dPath_gaussian}/figures/{iter}_{method}_voxel_signal_in_differential_screenshots.png', dpi=600, bbox_inches='tight')
plt.show()


