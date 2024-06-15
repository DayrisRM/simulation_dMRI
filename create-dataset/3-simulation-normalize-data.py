import numpy as np
import torch
import os 
import sbi
import sbi.utils
from sbi.utils.user_input_checks_utils import MultipleIndependent
from sbi.utils import BoxUniform
from sbi.inference import prepare_for_sbi, simulate_for_sbi, SNPE, SNLE, SNRE
from sbi.analysis import pairplot

import nibabel as nb



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


def export_nifti(data, orig_data, output_path, name):
    """
    Args:
        data:
        orig_data:
        output_path:
        name:
    """
    import nibabel as nb
    import os
    # Copy the header of the original image
    aff_mat = orig_data.affine
    nb.save(nb.Nifti2Image(data, affine=aff_mat), os.path.join(output_path, name))


#Load files
dPath = '/Simulations/noise-free-data'
data = get_data(f'{dPath}/test-noisy_data_snr10_b0b1b2k.nii.gz')


#b0s are the first 5 elements
ax_signal = len(data.shape) - 1
mean_nob0vols = np.mean(data[..., :5], axis=ax_signal)


data_norm = data / np.expand_dims(mean_nob0vols, axis=ax_signal)

orig_data = nb.load(f'{dPath}/test-noisy_data_snr10_b0b1b2k.nii.gz', mmap=True)	# This is to capture the correct header of the nifti image when exporting
export_nifti(data_norm, orig_data, dPath, 'test-noisy_data_snr10_b0b1b2k_normalized.nii.gz')





