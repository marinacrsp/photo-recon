#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Configuration
# ==============================================================================
BASE_DIR="/home/marina/ms_thesis/photo_recon_madrc/03_bicubic_interpolations"
OUTPUT_DIR_SynthSeg="/home/marina/ms_thesis/photo_recon_madrc/04_bicubic_photosynthseg"
MRI_SYNTHSEG="/home/marina/ms_thesis/photo_recon_madrc/best_recon_synthseg_rerun"

mkdir -p "$OUTPUT_DIR_SynthSeg"

# ==============================================================================
# Checks
# ==============================================================================
[[ -d "$BASE_DIR" ]] || { echo "ERROR: $BASE_DIR not found."; exit 1; }
[[ -d "$MRI_SYNTHSEG" ]] || { echo "ERROR: $MRI_SYNTHSEG not found."; exit 1; }


FAILED_SUBJECTS=()

echo "=============================================================================="
echo "Downstream analysis 2. Volumetric Segmentation"
echo "=============================================================================="


for subj_path in "$MRI_SYNTHSEG"/*; do

    if ! (
        SUBJ_ID=$(basename "$subj_path")
        SUBJ_DIR="${SUBJ_ID%%_*}"

        echo "------------------------------------------------------------------------------"
        echo "Processing $SUBJ_ID"
        echo "------------------------------------------------------------------------------"

        INPUT_FILE="${BASE_DIR}/${SUBJ_DIR}/photo_recon_tricubic.nii.gz"

        OUTPUT_SUBJ="${OUTPUT_DIR_SynthSeg}/${SUBJ_ID}"
        mkdir -p "$OUTPUT_SUBJ"

        COMMANDS_LOG="${OUTPUT_SUBJ}/commands.txt"

        OUTPUT_SEG="${OUTPUT_SUBJ}/synthseg_photo_recon_tricubic.nii.gz"
        OUTPUT_RESAMPLED="${OUTPUT_SUBJ}/synthseg_photo_recon_tricubic_resampled.nii.gz"

        for folder in "$MRI_SYNTHSEG/$SUBJ_ID"/*; do

            [[ -d "$folder" ]] || continue

            TARGET_SEG="$folder/mri.deformed_synthseg.nii.gz"

            ###################################################################
            # SynthSeg
            ###################################################################

            if [[ "$SUBJ_ID" == *left* ]]; then

                CMD=(
                    mri_synthseg
                    --i "$INPUT_FILE"
                    --o "$OUTPUT_SEG"
                    --cpu
                    --photo left
                )

            else

                CMD=(
                    mri_synthseg
                    --i "$INPUT_FILE"
                    --o "$OUTPUT_SEG"
                    --cpu
                    --photo both
                )

            fi

            printf '%q ' "${CMD[@]}" >> "$COMMANDS_LOG"
            echo >> "$COMMANDS_LOG"

            "${CMD[@]}"

            ###################################################################
            # Resample
            ###################################################################

            CMD=(
                mri_convert
                -i "$OUTPUT_SEG"
                -o "$OUTPUT_RESAMPLED"
                -rl "$TARGET_SEG"
                -rt nearest
            )

            printf '%q ' "${CMD[@]}" >> "$COMMANDS_LOG"
            echo >> "$COMMANDS_LOG"

            "${CMD[@]}"

            ###################################################################
            # Dice
            ###################################################################

            echo "Comparing against $(basename "$folder")"

            CMD=(
                mri_compute_overlap
                -a
                -l "${OUTPUT_SUBJ}/dice_$(basename "$folder").json"
                "$OUTPUT_RESAMPLED"
                "$TARGET_SEG"
            )

            printf '%q ' "${CMD[@]}" >> "$COMMANDS_LOG"
            echo >> "$COMMANDS_LOG"

            "${CMD[@]}"

            echo >> "$COMMANDS_LOG"

        done

        echo "✓ Finished $SUBJ_ID"

    ); then

        echo
        echo "ERROR while processing $SUBJ_ID"
        echo "Skipping..."
        echo

        FAILED_SUBJECTS+=("$SUBJ_ID")

    fi

done

if ((${#FAILED_SUBJECTS[@]})); then
    echo
    echo "The following subjects failed:"
    printf '  %s\n' "${FAILED_SUBJECTS[@]}"
fi