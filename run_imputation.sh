# #!/usr/bin/env bash
# set -euo pipefail

# subjectid="18-0086"
# thickness="12"

# photo_recon="/home/marina/ms_thesis/photo_recon_uw/00_photo_recon/${subjectid}/photo_recon_${thickness}mm.nii.gz"
# output_dir="/home/marina/ms_thesis/photo_recon_uw/photo_reconstructions/${subjectid}"

# mkdir -p $output_dir 

# fullpath="${output_dir}/imputation_${thickness}mm.mgz"

# echo "Sampling"

# python sample.py --input "$photo_recon" --save_path "$fullpath" 

#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# ==============================================================================
#                                CONFIGURATION
# ==============================================================================
# Path to the base directory containing all subject folders
BASE_DIR="/home/marina/ms_thesis/photo_recon_uw/00_photo_recon"

# Path to your python script and the pretrained model weights
PYTHON_SCRIPT="sample_machine_learning.py"
MODEL_PATH="/home/marina/ms_thesis/imputation_unet_2026_code/model_weights.pth"

# Target slice resolutions to look for
RESOLUTIONS=("4mm" "8mm" "12mm")
# ==============================================================================

# Check if the base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory '$BASE_DIR' not found!"
    exit 1
fi

# Check if the python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found in current directory!"
    exit 1
fi

echo "=============================================================================="
echo " Starting Tricubic Spline Imputation Pipeline"
echo "=============================================================================="

OUTPUT_DIR=/home/marina/ms_thesis/photo_recon_uw/03_bicubic_interpolations/
mkdir -p $OUTPUT_DIR
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
            INPUT_FILE="${subj_path}/photo_recon_correct_${res}.nii.gz"
            OUTPUT_FILEPATH="${OUTPUT_DIR}${SUBJ_ID}/"

            mkdir -p $OUTPUT_FILEPATH
            # Verify the expected input file actually exists before running
            if [ -f "$INPUT_FILE" ]; then
                echo " -> Sampling $res volume..."
                
                # Execute your python script with the mapped file parameters
                python "$PYTHON_SCRIPT" \
                    --input_file "$INPUT_FILE" \
                    --save_path "${OUTPUT_FILEPATH}/photo_recon_${res}.nii.gz"
                    
                echo "    ✓ Saved to: $OUTPUT_FILE"
            else
                echo " -> [Skipping] Resolution file not found: photo_recon_${res}.nii.gz"
            fi
        done
    fi
done

echo "=============================================================================="
echo " All experiments finished successfully!"
echo "=============================================================================="