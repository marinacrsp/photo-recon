#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
#                                CONFIGURATION
# ==============================================================================
# Path to the base directory containing all subject folders
BASE_DIR="/home/marina/ms_thesis/photo_recon_uw/03_bicubic_interpolations"

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
ATLAS="/home/marina/ms_thesis/atlases/exvivo_mni_icbm152_t1_tal_nlin_sym_09c.nii.gz"

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
            INPUT_FILE="${BASE_DIR}/${SUBJ_ID}/photo_recon_${res}_tricubic_gray.nii.gz"
            OUTPUT_FILEPATH="${OUTPUT_DIR_Registr}${SUBJ_ID}"

            mkdir -p $OUTPUT_FILEPATH

            # Verify the expected input file actually exists before running
            if [ -f "$INPUT_FILE" ]; then
                echo " -> Sampling $res volume..."
                
                echo "2. Volume Segmentation"
                mkdir -p "$OUTPUT_FILEPATH/derivatives"

                ATLAS_aligned="$OUTPUT_FILEPATH/mni2cubic_${res}_affine.nii.gz"
                AFF_file="$OUTPUT_FILEPATH/derivatives/affine_transform_${res}.txt"

                reg_aladin \
                    -ref "${INPUT_FILE}" \
                    -flo "${ATLAS}" \
                    -aff  "${AFF_file}"\
                    -res "${ATLAS_aligned}"

                # # 3.2 Non linear registration: affined-registered-atlas (floating) -> photo-reconstructions (reference) 
                # #################################################################################################
                # nonlinear_field="$OUTPUT_FILEPATH/derivatives/nonrigid_transform_${res}.nii.gz"
                # ATLAS_aligned_nonlin="$OUTPUT_FILEPATH/mni2cubic_${res}_nonlinear.nii.gz"

                # reg_f3d \
                #     -ref "${INPUT_FILE}" \
                #     -flo "${ATLAS_aligned}" \
                #     -cpp "${nonlinear_field}" \
                #     -res "${ATLAS_aligned_nonlin}"

            fi
        done
    fi
done

echo "=============================================================================="
echo " All Downstream analyses finished successfully!"
echo "=============================================================================="