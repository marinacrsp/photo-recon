#!/usr/bin/env bash
set -euo pipefail

# Downstream analyses for a photo-reconstructed brain volume.
# 1. Surface Reconstruction. 2. Volume segmentation. 3. Atlas registration.
# Requirements: FreeSurfer (surface reconstruction and SynthSeg steps), NiftyReg (registration).

atlas="/home/marina/ms_thesis/atlases/exvivo_mni_icbm152_t1_tal_nlin_sym_09c.nii.gz"

# e.g. For the one sample available (download data from Zenodo)
subjectid="18-0086"
thickness="8"

photo_recon="/home/marina/ms_thesis/photo_recon_uw/00_photo_recon/${subjectid}/photo_recon_${thickness}mm.nii.gz"
photo_recon_resample="/home/marina/ms_thesis/photo_recon_uw/00_photo_recon/${subjectid}/photo_recon_resampled_${thickness}mm.nii.gz"
mri_gt="/home/marina/ms_thesis/photo_recon_uw/00_photo_recon/${subjectid}/mri.deformed.photo_space.nii.gz"

# No trailing slash; build every sub-path with an explicit "/"
output_dir="/home/marina/ms_thesis/photo_recon_uw/downstream_tasks"
mkdir -p "$output_dir"

# 1. Surface reconstruction (recon-all-clinical / "Recon-Any"; requires separate installation)
#################################################################################################

echo "1. Surface Reconstruction"
run_recon-any -i "$photo_recon_resample" -subjid "$subjectid" -side both \
              -sdir "$output_dir/01_photo_recon_recon-all/" -threads 1

# 2. Volume segmentation with SynthSeg (FreeSurfer)
#################################################################################################

echo "2. Volume Segmentation"
mkdir -p "$output_dir/02_photo_recon_synthseg"
mri_synthseg --i "$photo_recon" \
             --o "$output_dir/02_photo_recon_synthseg/${subjectid}_${thickness}.mgz" \
             --cpu --photo both

example_segmentation="$output_dir/02_photo_recon_synthseg/${subjectid}_${thickness}.mgz"

# If MRI segmentation not available, please compute it.
gt_segmentation="$output_dir/02_mri_synthseg/${subjectid}/synthseg_mri.mgz" 

# Compute Dice overlap
dir_dice_scores="$output_dir/02_photo_recon_synthseg/dice_scores/${subjectid}_${thickness}.txt"
mri_compute_overlap example_segmentation gt_segmentation -a -s $file_dice_scores


# 3. Atlas registration with NiftyReg 
#################################################################################################

# 3.1 Linear (affine) registration: atlas (floating) -> photo-reconstruction (reference)
#################################################################################################

reg_dir="$output_dir/03_photo_recon_registration/${subjectid}"
mkdir -p "$reg_dir/derivatives"

reg_aladin \
    -ref "$photo_recon_resample" \
    -flo "$atlas" \
    -aff "$reg_dir/derivatives/atlas2_${thickness}mm_affine.txt" \
    -res "$reg_dir/mni_${thickness}mm_affine.nii.gz"

affine_file="$reg_dir/derivatives/atlas2_${thickness}mm_affine.txt"
atlas_aligned="$reg_dir/mni_${thickness}mm_affine.nii.gz"

# 3.2 Non linear registration: affined-registered-atlas (floating) -> photo-reconstructions (reference) 
#################################################################################################

reg_f3d \
    -ref "$photo_recon_resample" \
    -flo "$atlas_aligned" \
    -cpp "$reg_dir/derivatives/atlas2_${thickness}mm_nonrigid.nii.gz" \
    -res "$reg_dir/mni_${thickness}mm_nonrigid.nii.gz"