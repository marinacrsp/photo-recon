#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
#                                CONFIGURATION
# ==============================================================================
# Path to the base directory containing all subject folders
BASE_DIR="/home/marina/ms_thesis/photo_recon_uw/05_bicubic_atlas_registration/"

# Target slice resolutions to look for
RESOLUTIONS=("4mm" "8mm" "12mm")
# ==============================================================================

# Check if the base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory '$BASE_DIR' not found!"
    exit 1
fi

echo "=============================================================================="
echo " Downstream analysis 2. Volumetric Segmentation"
echo "=============================================================================="

OUTPUT_DIR_SynthSeg=/home/marina/ms_thesis/photo_recon_uw/04_bicubic_synthseg/
OUTPUT_DIR_Registr=/home/marina/ms_thesis/photo_recon_uw/05_bicubic_atlas_registration/
OUTPUT_DIR_Surf=/home/marina/ms_thesis/photo_recon_uw/06_bicubic_recon-any/
synthATLAS="/home/marina/ms_thesis/atlases/exvivo_synthseg_mni_icbm152_t1_tal_nlin_sym_09c.nii.gz"

# mkdir -p $OUTPUT_DIR_SynthSeg 
mkdir -p $OUTPUT_DIR_Registr 
# mkdir -p $OUTPUT_DIR_Surf 

# Loop through all items in the base directory
for subj_path in "$BASE_DIR"/*; do
    # Ensure it's a directory
    if [ -d "$subj_path" ]; then
        # Extract just the subject ID (e.g., 17-0333)
        SUBJ_ID=$(basename "$subj_path")
        
        echo "------------------------------------------------------------------------------"
        echo "Processing Subject: $SUBJ_ID"
        echo "------------------------------------------------------------------------------"
        
        # Loop through the target resolutions (4mm, 8mm, 12mm)
        for res in "${RESOLUTIONS[@]}"; do
            INPUT_FILE_linear="${BASE_DIR}${SUBJ_ID}/mni2cubic_${res}_affine.nii.gz"
            INPUT_FILE_nonrigid="${BASE_DIR}${SUBJ_ID}/mni2cubic_${res}_nonlinear.nii.gz"

            OUTPUT_FILEPATH="${OUTPUT_DIR_Registr}${SUBJ_ID}"

            mkdir -p $OUTPUT_FILEPATH

            # Verify the expected input file actually exists before running
            if [ -f "$INPUT_FILE_linear" ]; then
                echo " -> Sampling $res volume..."
                
                echo "2. Volume Segmentation"
                mkdir -p "$OUTPUT_FILEPATH/derivatives"

                synthseg_linear="$OUTPUT_FILEPATH/synthseg_${res}_affine.nii.gz"
                synthseg_nonrigid="$OUTPUT_FILEPATH/synthseg_${res}_nonlinear.nii.gz"

                AFF_file="$OUTPUT_FILEPATH/derivatives/affine_transform_${res}.txt"
                deffield_file="$OUTPUT_FILEPATH/derivatives/nonrigid_transform_${res}.nii.gz"

                reg_resample \
                    -ref "${INPUT_FILE_linear}" \
                    -flo "${synthATLAS}" \
                    -trans  "${AFF_file}"\
                    -res "${synthseg_linear}" \
                    -inter 0

                reg_resample \
                    -ref "${INPUT_FILE_nonrigid}" \
                    -flo "${synthseg_linear}" \
                    -trans "${deffield_file}"\
                    -res "${synthseg_nonrigid}" \
                    -inter 0

            fi
        done
    fi
done

echo "=============================================================================="
echo " All Downstream analyses finished successfully!"
echo "=============================================================================="