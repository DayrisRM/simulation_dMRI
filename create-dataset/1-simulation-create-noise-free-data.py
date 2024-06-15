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
        sigma = 1 / SNR 
        
        if type_noise == 'Gaussian':
            try:
                random = np.random.normal(0, sigma, len(Sj))                
                Sj_noise = Sj + random
            except:                
                print("Exception sigma")                
        elif type_noise == 'Rician':  # Noise in quadrature
            noise_1 = np.random.normal(0, sigma, len(Sj)) 
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
dPath = '/Simulations/data'
data = get_data(f'{dPath}/data.nii.gz')
mask = get_data(f'{dPath}/nodif_brain_mask.nii.gz')
mean_d = get_data(f'{dPath}/data.bedpostX/mean_dsamples.nii.gz')
mean_f1 = get_data(f'{dPath}/data.bedpostX/mean_f1samples.nii.gz') #volume fraction f1 > f2 > f3
mean_f2 = get_data(f'{dPath}/data.bedpostX/mean_f2samples.nii.gz')
mean_f3 = get_data(f'{dPath}/data.bedpostX/mean_f3samples.nii.gz')
v1 = get_data(f'{dPath}/data.bedpostX/dyads1.nii.gz') #orientacion de la fibra 
v2 = get_data(f'{dPath}/data.bedpostX/dyads2.nii.gz')
v3 = get_data(f'{dPath}/data.bedpostX/dyads3.nii.gz')
mean_S0 = get_data(f'{dPath}/data.bedpostX/mean_S0samples.nii.gz')

bvals = np.genfromtxt(dPath + '/bvals', dtype=np.float32)
bvecs = np.genfromtxt(dPath + '/bvecs', dtype=np.float32)    


#Generate noisefree-data
# Initialize empty arrays
noisefree_data = np.zeros_like(data)
th1 = np.zeros_like(mean_d)
phi1 = np.zeros_like(mean_d)
th2 = np.zeros_like(mean_d)
phi2 = np.zeros_like(mean_d)
th3 = np.zeros_like(mean_d)
phi3 = np.zeros_like(mean_d)

print('Simulating data')
# Simulate data
simulator = ball_and_sticks(bvals, bvecs)
x, y, z = np.where(mask > 0)
for i, j, k in zip(x, y, z):
    _, th1[i,j,k], phi1[i,j,k] = cart2sph(v1[i,j,k,0], v1[i,j,k,1], v1[i,j,k,2])
    _, th2[i,j,k], phi2[i,j,k] = cart2sph(v2[i,j,k,0], v2[i,j,k,1], v2[i,j,k,2])
    _, th3[i,j,k], phi3[i,j,k] = cart2sph(v3[i,j,k,0], v3[i,j,k,1], v3[i,j,k,2])
    th = np.array([mean_d[i, j, k], mean_f1[i, j, k], th1[i, j, k], phi1[i, j, k], mean_f2[i, j, k], th2[i, j, k], phi2[i, j, k], mean_f3[i, j, k], th3[i, j, k], phi3[i, j, k]])
    noisefree_data[i, j, k] = simulator(th)

print('Save noise-free-data')
# If you want to export the niftis to visualize them in FSLeyes, do for example:
orig_data = nb.load(f'{dPath}/data.nii.gz', mmap=True)	# This is to capture the correct header of the nifti image when exporting
export_nifti(noisefree_data, orig_data, dPath, 'noisyfree_data_full.nii.gz')