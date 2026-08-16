#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Configuration
# ==============================================================================
BASE_DIR="/home/marina/ms_thesis/photo_recon_madrc/photo_reconstruction"
OUTPUT_DIR_SynthSeg="/home/marina/ms_thesis/photo_recon_madrc/ADRC_synthseg/synthseg"
MRI_SYNTHSEG="/home/marina/ms_thesis/photo_recon_madrc/ADRC_synthseg/synthseg"

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

        INPUT_FILE="${BASE_DIR}/${SUBJ_DIR}/best_photo_recon/photo_recon.machine_learning.mgz"

        OUTPUT_SUBJ="${OUTPUT_DIR_SynthSeg}/${SUBJ_ID}"
        # mkdir -p "$OUTPUT_SUBJ"

        COMMANDS_LOG="${OUTPUT_SUBJ}/commands.txt"

        for folder in "$MRI_SYNTHSEG/$SUBJ_ID"/*; do

            name_file=$(basename "$folder")

            # OUTPUT_SEG="$folder/photo_recon.machine_learning_synthseg.nii.gz"
            OUTPUT_SEG="$folder/${name_file}_synthseg.nii.gz"
            # OUTPUT_RESAMPLED="${OUTPUT_SUBJ}/photo_recon.machine_learning_synthseg_resampled.nii.gz"

            [[ -d "$folder" ]] || continue

            TARGET_SEG="$folder/mri.deformed_synthseg.nii.gz"
            OUTPUT_FILESTATS="$folder/seg_stats.txt"
            # OUTPUT_FILESTATS="$folder/seg_stats_unet.txt"

            # ###################################################################
            # # SynthSeg
            # ###################################################################

            # if [[ "$SUBJ_ID" == *left* ]]; then

            #     CMD=(
            #         mri_synthseg
            #         --i "$INPUT_FILE"
            #         --o "$OUTPUT_SEG"
            #         --cpu
            #         --photo left
            #     )

            # else

            #     CMD=(
            #         mri_synthseg
            #         --i "$INPUT_FILE"
            #         --o "$OUTPUT_SEG"
            #         --cpu
            #         --photo both
            #     )

            # fi

            # printf '%q ' "${CMD[@]}" >> "$COMMANDS_LOG"
            # echo >> "$COMMANDS_LOG"

            # "${CMD[@]}"

            # ###################################################################
            # # Resample
            # ###################################################################

            # CMD=(
            #     mri_convert
            #     -i "$OUTPUT_SEG"
            #     -o "$OUTPUT_RESAMPLED"
            #     -rl "$TARGET_SEG"
            #     -rt nearest
            # )

            # printf '%q ' "${CMD[@]}" >> "$COMMANDS_LOG"
            # echo >> "$COMMANDS_LOG"

            # "${CMD[@]}"

            # ###################################################################
            # # Dice
            # ###################################################################

            # echo "Comparing against $(basename "$folder")"

            # CMD=(
            #     mri_compute_overlap
            #     -a
            #     -l "${OUTPUT_SUBJ}/dice_$(basename "$folder").json"
            #     "$OUTPUT_RESAMPLED"
            #     "$TARGET_SEG"
            # )

            # printf '%q ' "${CMD[@]}" >> "$COMMANDS_LOG"
            # echo >> "$COMMANDS_LOG"

            # "${CMD[@]}"

            # echo >> "$COMMANDS_LOG"
            ###################################################################
            # Stats
            ###################################################################
            echo "Computing statistics from the segmentation volume"

            CMD=(
                mri_segstats
                --seg "${OUTPUT_SEG}" 
                --sum "${OUTPUT_FILESTATS}"
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