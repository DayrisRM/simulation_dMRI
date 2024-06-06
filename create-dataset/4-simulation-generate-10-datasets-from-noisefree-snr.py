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

#Common functions
# Define the signal simulator
class ball_and_sticks:
    """ Ball&Sticks dMRI model

            Use the Ball&Sticks model to simulate the signal attenuation S/S0 (i.e. T2 contrast removed). 
            Hence, bvals and bvecs are assumed to not contain b0 volumes

            Args:
                params: model parameters -> [d, f_1, th_1, ph_1, ..., f_n, th_n, ph_n, SNR]
            """
    import numpy as np
    import torch
    import os 

    def __init__(self, bvals, bvecs):          
        self.bvals = bvals
        self.bvecs = bvecs

    def add_noise(Sj, SNR, type_noise):
        # b0vols = np.argwhere(bvals>50) # Get the b0 volumes, where there is no diffusion (pure T2 contrast)
        # mean_b0vols = np.mean(Sj[b0vols])
        # if mean_b0vols<0.000001: # Background voxels
        #     mean_b0vols = 1
        # sigma = mean_b0vols / SNR # SNR = mean(signal)/std(noise) --> mean(b0_vols)/sigma 
        sigma = 1 / SNR 
        
        if type_noise == 'Gaussian':
            try:
                random = np.random.normal(0, sigma, len(Sj))                
                Sj_noise = Sj + random #sig * np.random.normal(0, s0/SNR, len(Sj))
                Sj_noise_modificado = np.where(Sj_noise < 0, 0.0000001, Sj_noise)
            except Exception as e:                
                print("Se produjo una excepción:", e)             
        elif type_noise == 'Rician':  # Noise in quadrature
            noise_1 = np.random.normal(0, sigma, len(Sj))  #sigma or sigma**2?
            noise_2 = np.random.normal(0, sigma, len(Sj))
            Sj_noise = np.sqrt((Sj + noise_1) ** 2 + noise_2 ** 2)

        return Sj_noise

    def __call__(self, params):
        params = params.flatten()
        n_fib = int((len(params) - 1) / 3)
        s0 = 1
        d = params[0]
        v = np.zeros((n_fib, 3))
        sumf = 0
        signal = torch.tensor((np.zeros((len(self.bvals)))))  # np.zeros((len(b)))

        for i in range(0, n_fib):
            fi = params[1 + 3 * i]
            sumf += fi
            th = params[2 + 3 * i]
            phi = params[3 + 3 * i]
            #v = np.array([params[3 + 3 * i], params[4 + 3 * i], params[5 + 3 * i]])
            v = np.array([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)])  # conversion to cartesians
            signal += s0 * (fi * np.exp(-d * self.bvals * np.power(np.dot(self.bvecs.T, v), 2)))    # sticks contribution to the signal

        signal += s0 * (1 - sumf) * np.exp(-self.bvals * d)  # isotropic contribution
        return signal


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


def cart2sph(x,y,z):
  import numpy as np
  import math
  #takes list xyz (single coord)
  r       =  np.sqrt(x*x + y*y + z*z)
  if r==0:
    theta = math.acos(z / 1)    # To avoid NaN when r==0
  else:
    theta = math.acos(z/r) #*180/ math.pi #to degrees

  phi = math.atan2(y,x)  #*180/ math.pi
  return r, theta, phi

#Load files
dPath = 'C:/Users/dayri/Documents/UNED/TFM/Related_projects/Simulations/Simulations/data/generated_data_noisyfree_b0_first'
data = get_data(f'{dPath}/noisyfree_data_full_b0_first.nii.gz')
orig_data = nb.load(f'{dPath}/noisyfree_data_full_b0_first.nii.gz', mmap=True)	# This is to capture the correct header of the nifti image when exporting
dPath_toSave = 'C:/Users/dayri/Documents/UNED/TFM/Related_projects/Simulations/Simulations/data/generated_data_noisyfree_b0_first/snrs'

SNRs = [40]


print('Adding (Gaussian) noise at different SNR levels')
# Add (Gaussian) noise at different SNR levels
def add_noise_to_data(data, snrs, noise_type='Gaussian'):
    noisy_data = {}
    for snr in snrs:
        # Flatten the 4D array to 2D (voxels x timepoints) for vectorized noise addition
        flattened_data = data.reshape(-1, data.shape[-1])
        # Apply noise to the entire dataset in a vectorized manner        
        noisy_flattened = np.array([ball_and_sticks.add_noise(xi, snr, noise_type) for xi in flattened_data])
        # Reshape back to original 4D shape
        noisy_data[snr] = noisy_flattened.reshape(data.shape)
        
    return noisy_data

def save_noisy_dataset(noisy_dataset, snr, i):
    dataset_name = f'{i}noisy_data_snr{snr}.nii.gz'
    export_nifti(noisy_dataset, orig_data, dPath_toSave, dataset_name)

for i in range(1, 6): 
    print('Adding noise to dataset')
    noisy_data = add_noise_to_data(data, SNRs)
    print('Saving noising dataset')
    save_noisy_dataset(noisy_data[40], 40, i)









