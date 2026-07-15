#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Configuration
# ==============================================================================
BASE_DIR="/home/marina/ms_thesis/photo_recon_madrc/03_bicubic_interpolations"
OUTPUT_DIR_REG="/home/marina/ms_thesis/photo_recon_madrc/05_bicubic_registration_rerun"
MRI_SYNTHSEG="/home/marina/ms_thesis/photo_recon_madrc/best_recon_synthseg_rerun"

ATLAS="/home/marina/ms_thesis/atlases/exvivo_mni_icbm152_t1_tal_nlin_sym_09c.nii.gz"
SYNTH_ATLAS="/home/marina/ms_thesis/atlases/exvivo_synthseg_mni_icbm152_t1_tal_nlin_sym_09c.nii.gz"

mkdir -p "$OUTPUT_DIR_REG"

[[ -d "$BASE_DIR" ]] || { echo "Base directory not found."; exit 1; }
[[ -d "$MRI_SYNTHSEG" ]] || { echo "MRI directory not found."; exit 1; }

FAILED_COMPARISONS=()

echo "=============================================================================="
echo "Downstream analysis 3. Registration"
echo "=============================================================================="

files=(
    'sub-2745_both'
    'sub-2746_left'
    'sub-2748_left'
    'sub-2752_left'
    'sub-2769_both'
    'sub-2773_left'
    'sub-2774_left'
    'sub-2775_left'
    'sub-2785_left'
    'sub-2791_left'
    'sub-2809_left'
    )

for SUBJ_ID in "${files[@]}"; do

    SUBJ_DIR="${SUBJ_ID%%_*}"

    INPUT_FILE="${BASE_DIR}/${SUBJ_DIR}/photo_recon_tricubic.nii.gz"
    MRI_ORIGINAL="/home/marina/ms_thesis/photo_recon_madrc/photo_reconstruction/${SUBJ_DIR}/best_photo_recon/mri.deformed.mgz"

    if [[ ! -f "$INPUT_FILE" ]]; then
        echo "Missing $INPUT_FILE"
        FAILED_COMPARISONS+=("${SUBJ_ID} :: missing input")
        continue
    fi

    for folder in "$MRI_SYNTHSEG/$SUBJ_ID"/*; do

        [[ -d "$folder" ]] || continue

        if ! (

            TARGET_SEG="$folder/mri.deformed_synthseg.nii.gz"

            [[ -f "$TARGET_SEG" ]] || {
                echo "Missing $TARGET_SEG"
                exit 1
            }

            OUTPUT_SUBJ="${OUTPUT_DIR_REG}/${SUBJ_ID}"
            mkdir -p "$OUTPUT_SUBJ"
            mkdir -p "${OUTPUT_SUBJ}/derivatives"

            RESAMPLED="${OUTPUT_SUBJ}/photo_recon_tricubic_resampled.nii.gz"

            ATLAS_AFFINE="${OUTPUT_SUBJ}/mni2cubic_affine.nii.gz"
            ATLAS_NONLINEAR="${OUTPUT_SUBJ}/mni2cubic_nonlinear.nii.gz"

            AFFINE_FILE="${OUTPUT_SUBJ}/derivatives/affine_transform.txt"
            DEFFIELD_FILE="${OUTPUT_SUBJ}/derivatives/nonrigid_transform.nii.gz"

            SYNTHSEG_LINEAR="${OUTPUT_SUBJ}/synthseg_affine.nii.gz"
            SYNTHSEG_NONRIGID="${OUTPUT_SUBJ}/synthseg_nonlinear.nii.gz"

            echo "--------------------------------------------------"
            echo "Subject: $SUBJ_ID"
            echo "Comparison: $(basename "$folder")"
            echo "--------------------------------------------------"

            mri_convert \
                -i "$INPUT_FILE" \
                -o "$RESAMPLED" \
                -rl "$MRI_ORIGINAL"

            reg_aladin \
                -ref "$RESAMPLED" \
                -flo "$ATLAS" \
                -aff "$AFFINE_FILE" \
                -res "$ATLAS_AFFINE"

            reg_f3d \
                -ref "$RESAMPLED" \
                -flo "$ATLAS_AFFINE" \
                -cpp "$DEFFIELD_FILE" \
                -res "$ATLAS_NONLINEAR"

            reg_resample \
                -ref "$ATLAS_AFFINE" \
                -flo "$SYNTH_ATLAS" \
                -trans "$AFFINE_FILE" \
                -res "$SYNTHSEG_LINEAR" \
                -inter 0

            reg_resample \
                -ref "$ATLAS_NONLINEAR" \
                -flo "$SYNTHSEG_LINEAR" \
                -trans "$DEFFIELD_FILE" \
                -res "$SYNTHSEG_NONRIGID" \
                -inter 0

            [[ -f "$SYNTHSEG_NONRIGID" ]] || {
                echo "Registration failed."
                exit 1
            }

            mri_compute_overlap \
                -a \
                -l "${OUTPUT_SUBJ}/dice_$(basename "$folder").json" \
                "$SYNTHSEG_NONRIGID" \
                "$TARGET_SEG"

        ); then

            echo
            echo "ERROR processing:"
            echo "  Subject : $SUBJ_ID"
            echo "  Folder  : $(basename "$folder")"
            echo

            FAILED_COMPARISONS+=("${SUBJ_ID} :: $(basename "$folder")")

        fi

    done

done

echo
echo "=============================================================================="
echo "Registration finished."
echo "=============================================================================="

if ((${#FAILED_COMPARISONS[@]})); then
    echo
    echo "Failed comparisons:"
    printf '  - %s\n' "${FAILED_COMPARISONS[@]}"
else
    echo
    echo "All comparisons completed successfully."
fi