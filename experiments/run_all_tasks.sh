#!/usr/bin/env bash
#
# run_all_tasks.sh - unified driver for the neuropathology reconstruction
# evaluation pipeline (UW + MADRC).
#
# Tasks (default order):
#   surface        task_surface_reconstruction.py   -> task_1_surface_reconstruction
#   segmentation   task_synthseg_segmentation.py    -> task_2_volume_segmentations
#   atlas          task_atlas_registration.py       -> task_3_atlas_registration
#   volumes        volume_correlations.py           -> task_4_volume_correlation_reconany
#   consistency    task_consistency.py              -> task_5_consistency
#
# Location: experiments/run_all_tasks.sh. CODE_ROOT and EXP_DIR are derived from
# this file's own path, so the driver can be invoked from any working directory.
#
# Usage:
#   ./experiments/run_all_tasks.sh                     # run every task
#   ./experiments/run_all_tasks.sh volumes consistency # subset, in the given order
#   ./experiments/run_all_tasks.sh --dry-run           # print the commands only
#   ./experiments/run_all_tasks.sh --recompute-surface # force stage-1 recomputation
#   ./experiments/run_all_tasks.sh --keep-going        # do not abort on first failure
#   ./experiments/run_all_tasks.sh --push --overleaf-repo /path/to/repo
#
# Every configuration variable can also be overridden from the environment, e.g.
#   RESULTS_ROOT=/tmp/eval ./experiments/run_all_tasks.sh volumes
#
set -Eeuo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON="${PYTHON:-/home/marina/envs/photo-imputation/bin/python3.11}"
EXP_DIR="${EXP_DIR:-$SELF_DIR}"                    # experiments/ (this script lives here)
CODE_ROOT="${CODE_ROOT:-$(dirname -- "$EXP_DIR")}" # repository root (parent of experiments/)

UW_DIR="${UW_DIR:-/home/marina/ms_thesis/photo_recon_uw}"
MADRC_DIR="${MADRC_DIR:-/home/marina/ms_thesis/photo_recon_madrc}"
RESULTS_ROOT="${RESULTS_ROOT:-/home/marina/ms_thesis/evaluation_results}"

# Output directories (one per task).
OUT_SURFACE="${OUT_SURFACE:-$RESULTS_ROOT/task_1_surface_reconstruction}"
OUT_SEGMENTATION="${OUT_SEGMENTATION:-$RESULTS_ROOT/task_2_volume_segmentations}"
OUT_ATLAS="${OUT_ATLAS:-$RESULTS_ROOT/task_3_atlas_registration}"
OUT_VOLUMES="${OUT_VOLUMES:-$RESULTS_ROOT/task_4_volume_correlation_reconany}"
OUT_CONSISTENCY="${OUT_CONSISTENCY:-$RESULTS_ROOT/task_5_consistency}"
LOG_DIR="${LOG_DIR:-$RESULTS_ROOT/logs}"

# --- Task 4 (volume correlations): per-cohort, per-method SynthSeg stats trees --
UW_REF_DIR="${UW_REF_DIR:-$UW_DIR/14_MRI_UW}"
UW_PHOTO_DIR="${UW_PHOTO_DIR:-$UW_DIR/04_photo_recon_synthseg}"
UW_TRICUBIC_DIR="${UW_TRICUBIC_DIR:-$UW_DIR/04_bicubic_synthseg}"
UW_IMPUTED_DIR="${UW_IMPUTED_DIR:-$UW_DIR/04_unet_synthseg}"

MADRC_REF_DIR="${MADRC_REF_DIR:-$MADRC_DIR/ADRC_synthseg/synthseg}"
MADRC_PHOTO_DIR="${MADRC_PHOTO_DIR:-$MADRC_DIR/best_recon_ss_qc_compute_overlap}"
MADRC_TRICUBIC_DIR="${MADRC_TRICUBIC_DIR:-$MADRC_DIR/04_bicubic_synthseg_andreconany}"
MADRC_IMPUTED_DIR="${MADRC_IMPUTED_DIR:-$MADRC_DIR/best_recon_ss_qc_compute_overlap}"

# --- Task 5 (consistency): reference tree + one entry per method --------------
# Format: "NAME:DIR:PATTERN", PATTERN must contain the {thick} placeholder.
CONSISTENCY_REF_DIR="${CONSISTENCY_REF_DIR:-$UW_DIR/00_photo_recon}"
# NB: a literal '}' cannot appear inside ${VAR:-default}, hence the two-step default.
CONSISTENCY_REF_PATTERN="${CONSISTENCY_REF_PATTERN:-}"
[[ -n "$CONSISTENCY_REF_PATTERN" ]] || CONSISTENCY_REF_PATTERN='photo_recon_correct_{thick}mm.nii.gz'
CONSISTENCY_METHODS=(
  "Imputed:$UW_DIR/02_imputations_unet:imputed_unet_{thick}mm.nii.gz"
  "Tricubic:$UW_DIR/03_bicubic_interpolations:photo_recon_{thick}mm_tricubic.nii.gz"
)
CONSISTENCY_PLOTS="${CONSISTENCY_PLOTS:-interval}"        # all | interval | none
CONSISTENCY_PLOT_INTERVAL="${CONSISTENCY_PLOT_INTERVAL:-5}"

# --- Optional Overleaf push (tasks 1-3 support it) ----------------------------
OVERLEAF_REPO="${OVERLEAF_REPO:-}"

ALL_TASKS=(surface segmentation atlas volumes consistency)

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
DRY_RUN=0
KEEP_GOING=0
RECOMPUTE_SURFACE=0
PUSH=0
SELECTED=()

usage() {
  awk 'NR > 1 { if ($0 ~ /^set -/) exit; sub(/^# ?/, ""); print }' "$0"
  exit "${1:-0}"
}

while (($#)); do
  case "$1" in
    -h|--help)            usage 0 ;;
    -n|--dry-run)         DRY_RUN=1 ;;
    -k|--keep-going)      KEEP_GOING=1 ;;
    --recompute-surface)  RECOMPUTE_SURFACE=1 ;;
    --push)               PUSH=1 ;;
    --overleaf-repo)      OVERLEAF_REPO="$2"; shift ;;
    --python)             PYTHON="$2"; shift ;;
    --results-root)       RESULTS_ROOT="$2"; shift ;;
    --list)               printf '%s\n' "${ALL_TASKS[@]}"; exit 0 ;;
    -*)                   echo "Unknown option: $1" >&2; usage 2 ;;
    *)                    SELECTED+=("$1") ;;
  esac
  shift
done

if ((${#SELECTED[@]} == 0)); then
  SELECTED=("${ALL_TASKS[@]}")
fi

for t in "${SELECTED[@]}"; do
  found=0
  for known in "${ALL_TASKS[@]}"; do [[ "$t" == "$known" ]] && found=1; done
  ((found)) || { echo "Unknown task: $t (known: ${ALL_TASKS[*]})" >&2; exit 2; }
done

if ((PUSH)) && [[ -z "$OVERLEAF_REPO" ]]; then
  echo "--push requires --overleaf-repo (or the OVERLEAF_REPO variable)." >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
STATUS_LINES=()
FAILED=0

# =============================================================================
# PREFLIGHT
# =============================================================================
preflight() {
  local missing=0

  [[ -x "$PYTHON" ]] || { echo "Python interpreter not executable: $PYTHON" >&2; missing=1; }

  local -A scripts=(
    [surface]="$EXP_DIR/task_surface_reconstruction.py"
    [segmentation]="$EXP_DIR/task_synthseg_segmentation.py"
    [atlas]="$EXP_DIR/task_atlas_registration.py"
    [volumes]="$EXP_DIR/volume_correlations.py"
    [consistency]="$EXP_DIR/task_consistency.py"
  )
  for t in "${SELECTED[@]}"; do
    [[ -f "${scripts[$t]}" ]] || { echo "Missing script: ${scripts[$t]}" >&2; missing=1; }
  done

  for d in "$UW_DIR" "$MADRC_DIR"; do
    [[ -d "$d" ]] || { echo "Missing data directory: $d" >&2; missing=1; }
  done

  ((missing == 0)) || exit 1

  mkdir -p "$LOG_DIR" "$RESULTS_ROOT"

  # Import roots differ per module: ext.photo_imputation_utils resolves from the
  # repository root, while utils.summary_tables / utils.combine_hemispheres live
  # in experiments/utils. Python also prepends each script's own directory to
  # sys.path, but both roots are exported so the imports do not rely on that.
  cd "$CODE_ROOT"
  export PYTHONPATH="$CODE_ROOT:$EXP_DIR${PYTHONPATH:+:$PYTHONPATH}"
  export MPLBACKEND="${MPLBACKEND:-Agg}"

  cat <<EOF
-------------------------------------------------------------------------------
 Pipeline run ${STAMP}
   python        : $PYTHON
   code root     : $CODE_ROOT
   experiments   : $EXP_DIR
   UW dir        : $UW_DIR
   MADRC dir     : $MADRC_DIR
   results root  : $RESULTS_ROOT
   logs          : $LOG_DIR
   tasks         : ${SELECTED[*]}
   dry run       : $DRY_RUN     keep going: $KEEP_GOING
-------------------------------------------------------------------------------
EOF
}

# =============================================================================
# RUNNER
# =============================================================================
run_py() {
  local name="$1"; shift
  local log="$LOG_DIR/${name}_${STAMP}.log"

  printf '\n==> [%s] %s\n' "$name" "$*"
  if ((DRY_RUN)); then
    STATUS_LINES+=("$(printf '%-13s %-8s %8s  %s' "$name" "DRY-RUN" "-" "-")")
    return 0
  fi

  local start=$SECONDS status=0
  set +e
  "$@" 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  set -e
  local dur=$((SECONDS - start))

  local verdict="OK"
  ((status == 0)) || verdict="FAIL($status)"
  STATUS_LINES+=("$(printf '%-13s %-8s %7ds  %s' "$name" "$verdict" "$dur" "$log")")

  if ((status != 0)); then
    FAILED=1
    echo "[$name] failed with exit code $status; log: $log" >&2
    ((KEEP_GOING)) || exit "$status"
  fi
  return 0
}

overleaf_args() {
  ((PUSH)) || return 0
  printf '%s\n' --overleaf-repo "$OVERLEAF_REPO" --push
}

# =============================================================================
# TASKS
# =============================================================================
task_surface() {
  local extra=()
  ((RECOMPUTE_SURFACE)) && extra+=(--recompute)
  mapfile -t -O "${#extra[@]}" extra < <(overleaf_args)
  run_py surface "$PYTHON" "$EXP_DIR/task_surface_reconstruction.py" \
    --uw-dir "$UW_DIR" \
    --madrc-dir "$MADRC_DIR" \
    --out-dir "$OUT_SURFACE" \
    ${extra[@]+"${extra[@]}"}
}

task_segmentation() {
  local extra=()
  mapfile -t extra < <(overleaf_args)
  run_py segmentation "$PYTHON" "$EXP_DIR/task_synthseg_segmentation.py" \
    --uw-dir "$UW_DIR" \
    --madrc-dir "$MADRC_DIR" \
    --out-dir "$OUT_SEGMENTATION" \
    ${extra[@]+"${extra[@]}"}
}

task_atlas() {
  local extra=()
  mapfile -t extra < <(overleaf_args)
  run_py atlas "$PYTHON" "$EXP_DIR/task_atlas_registration.py" \
    --uw-dir "$UW_DIR" \
    --madrc-dir "$MADRC_DIR" \
    --out-dir "$OUT_ATLAS" \
    ${extra[@]+"${extra[@]}"}
}

task_volumes() {
  run_py volumes "$PYTHON" "$EXP_DIR/volume_correlations.py" \
    --uw-ref-dir            "$UW_REF_DIR" \
    --uw-photo-recon-dir    "$UW_PHOTO_DIR" \
    --uw-tricubic-dir       "$UW_TRICUBIC_DIR" \
    --uw-imputed-dir        "$UW_IMPUTED_DIR" \
    --madrc-ref-dir         "$MADRC_REF_DIR" \
    --madrc-photo-recon-dir "$MADRC_PHOTO_DIR" \
    --madrc-tricubic-dir    "$MADRC_TRICUBIC_DIR" \
    --madrc-imputed-dir     "$MADRC_IMPUTED_DIR" \
    --out-dir               "$OUT_VOLUMES"
}

task_consistency() {
  local method_args=()
  for spec in "${CONSISTENCY_METHODS[@]}"; do
    method_args+=(--method "$spec")
  done
  run_py consistency "$PYTHON" "$EXP_DIR/task_consistency.py" \
    --ref-dir        "$CONSISTENCY_REF_DIR" \
    --ref-pattern    "$CONSISTENCY_REF_PATTERN" \
    --output-dir     "$OUT_CONSISTENCY" \
    --save-plots     "$CONSISTENCY_PLOTS" \
    --plot-interval  "$CONSISTENCY_PLOT_INTERVAL" \
    "${method_args[@]}"
}

# =============================================================================
# MAIN
# =============================================================================
summary() {
  printf '\n-------------------------------------------------------------------------------\n'
  printf ' Summary (%s)\n' "$STAMP"
  printf '%-13s %-8s %8s  %s\n' TASK STATUS TIME LOG
  printf '%s\n' "${STATUS_LINES[@]}"
  printf -- '-------------------------------------------------------------------------------\n'
  ((FAILED == 0)) || echo "One or more tasks failed." >&2
}
trap summary EXIT

preflight

for t in "${SELECTED[@]}"; do
  "task_${t}"
done

exit "$FAILED"