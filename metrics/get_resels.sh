#!/bin/bash

dPath=/home/day/simulations/simulated_data/get-resels
mask=data.dti/nodif_brain_mask_ero.nii.gz

#dti_tensor = noisy_data_snr10
#dti_pred = snr10_denoised|denoised_data
#dti_residuals = differences

# DTI
#rm -r ${dPath}/data.dti
#mkdir -p ${dPath}/data.dti
dtifit -k ${dPath}/data.dti/data_b0-b1k.nii.gz -o ${dPath}/data.dti/dti -m ${mask} -r ${dPath}/data.dti/data_b0-b1k.bvec -b ${dPath}/data.dti/data_b0-b1k.bval --save_tensor 
dtigen -t ${dPath}/data.dti/dti_tensor.nii.gz -o ${dPath}/data.dti/dti_pred -b ${dPath}/data.dti/data_b0-b1k.bval -r ${dPath}/data.dti/data_b0-b1k.bvec -m ${mask} --s0=${dPath}/data.dti/dti_S0.nii.gz
fslcpgeom ${dPath}/data.dti/data_b0-b1k.nii.gz ${dPath}/data.dti/dti_pred.nii.gz
fslmaths ${dPath}/data.dti/data_b0-b1k.nii.gz -sub ${dPath}/data.dti/dti_pred.nii.gz -mas ${mask} ${dPath}/data.dti/dti_residuals.nii.gz
smoothest -d 7 -r ${dPath}/data.dti/dti_residuals.nii.gz -m ${mask} > ${dPath}/data.dti/smoothest_res.txt


#bash get_resels.sh ${pathData}  ${pathData}/nodif_brain_mask_ero.nii.gz &