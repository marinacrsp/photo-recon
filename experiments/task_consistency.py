#!/usr/bin/env python3
"""
Imputation-consistency experiment (UW cohort).

Slice-wise comparison of each candidate reconstruction (8 mm and 12 mm) against
the 4 mm reference reconstruction, in terms of MAE, PSNR and SSIM, for an
arbitrary number of imputation methods declared on the command line.

Scope of this refactor
----------------------
The metric definitions, the canonicalisation step, the PSNR/SSIM data range,
the multi-channel SSIM averaging, the slice-matching arithmetic
(idx = 2*i - 1 for 8 mm; idx = 3*i and 3*i + 1 for 12 mm; j = floor(t*idx - t/2))
and the two-level aggregation (per volume, then per cohort) are IDENTICAL to the
original hard-coded script. Only the configuration surface is parameterised:
input paths, the method list, subject selection and the plotting policy.

Method specification
--------------------
One --method flag per method, formatted as:

    NAME:DIR:PATTERN

  NAME     label used in the tables and as the output sub-directory name
  DIR      directory containing one sub-directory per subject
  PATTERN  file name inside DIR/<subject>, containing the {thick} placeholder,
           which is substituted with 8 and 12

NAME, DIR and PATTERN must not themselves contain ':'.

Examples
--------
    python task_consistency.py \
        --ref-dir /home/marina/ms_thesis/photo_recon_uw/00_photo_recon \
        --output-dir /home/marina/ms_thesis/evaluation_results/task_5_consistency \
        --method "Imputed:/home/marina/ms_thesis/photo_recon_uw/02_imputations_unet:imputed_unet_{thick}mm.nii.gz" \
        --method "Tricubic:/home/marina/ms_thesis/photo_recon_uw/03_bicubic_interpolations:photo_recon_{thick}mm_tricubic.nii.gz"

Outputs
-------
Per method, under <output-dir>/<NAME>/:
    metrics_raw_slices.csv
    metrics_per_volume.csv
    metrics_overall_cohort_summary.csv
    sample_<subject>_<condition>_slice_<idx>.png     (see --save-plots)

Across methods, under <output-dir>/:
    metrics_per_volume_all_methods.csv
    metrics_overall_cohort_summary_all_methods.csv
"""

from __future__ import annotations

import os
import re
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")                 # batch-safe; figures are only written to disk
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio

# Custom external functions/classes (unchanged)
from ext.photo_imputation_utils import MRIread, eugenios_closest_canonical

# =============================================================================
# CONFIGURATION DEFAULTS
# =============================================================================
REF_PATTERN_DEFAULT = "photo_recon_correct_{thick}mm.nii.gz"
REF_THICKNESSES = ["4", "8", "12"]     # reference volumes loaded per subject
PLOT_POLICIES = ["all", "interval", "none"]


@dataclass(frozen=True)
class MethodSpec:
    """One imputation method: label, subject-level directory, file pattern."""
    name: str
    directory: str
    pattern: str

    def volume_path(self, subject: str, thick: int | str) -> str:
        return os.path.join(self.directory, subject,
                            self.pattern.format(thick=thick))

    @property
    def slug(self) -> str:
        return re.sub(r"[^0-9A-Za-z._-]+", "_", self.name).strip("_") or "method"


def parse_method(spec: str) -> MethodSpec:
    parts = spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--method expects 'NAME:DIR:PATTERN' (3 colon-separated fields), got: {spec!r}"
        )
    name, directory, pattern = (p.strip() for p in parts)
    if not name or not directory or not pattern:
        raise argparse.ArgumentTypeError(f"--method has an empty field: {spec!r}")
    if "{thick}" not in pattern:
        raise argparse.ArgumentTypeError(
            f"--method PATTERN must contain the {{thick}} placeholder, got: {pattern!r}"
        )
    return MethodSpec(name=name, directory=directory, pattern=pattern)


# =============================================================================
# EVALUATION (logic unchanged)
# =============================================================================
def evaluate_slice(orig_slice, imput_slice, slice_idx, condition_name, subject_id,
                   data_range, *, method_name, results_list, plot_dir,
                   save_plots="interval", plot_interval=5):
    """Computes evaluation metrics and optionally saves comparison plots."""
    img_true = orig_slice.astype(np.float32)
    img_test = imput_slice.astype(np.float32)

    # Calculate metrics
    mae_val = float(nn.L1Loss()(torch.tensor(img_true), torch.tensor(img_test)))
    psnr_val = peak_signal_noise_ratio(img_true, img_test, data_range=data_range)

    if psnr_val == float('inf'):
        if mae_val == 0:
            print(f"Warning: PSNR is infinite for {subject_id} | {condition_name} | "
                  f"Slice {slice_idx}. Images are identical.")
            psnr_val = 50.0  # Assign a high PSNR value for identical images

    # Handle multi-channel SSIM safely
    num_channels = img_test.shape[-1] if img_test.ndim > 2 else 1
    if num_channels > 1:
        ssim_val = 0
        for c in range(num_channels):
            ssim_val += ssim(img_true[..., c], img_test[..., c], data_range=data_range)
        ssim_val /= num_channels
    else:
        ssim_val = ssim(img_true, img_test, data_range=data_range)

    # Save metrics to our collector list
    results_list.append({
        'Method': method_name,
        'Subject': subject_id,
        'Condition': condition_name,
        'Slice_Index': slice_idx,
        'MAE': mae_val,
        'PSNR': psnr_val,
        'SSIM': ssim_val
    })

    # Plotting logic
    if save_plots == "none":
        should_plot = False
    elif save_plots == "all":
        should_plot = True
    else:
        should_plot = (plot_interval > 0) and (slice_idx % plot_interval == 0)

    if should_plot:
        plt.figure(figsize=(10, 5))
        plt.suptitle(f"{subject_id} | {condition_name} - Slice {slice_idx}\n"
                     f"MAE: {mae_val:.4f} | PSNR: {psnr_val:.2f}dB | SSIM: {ssim_val:.4f}")

        plt.subplot(1, 2, 1)
        plt.title("Original (4mm)")
        plt.imshow(img_true.astype(np.uint8), cmap='gray' if num_channels == 1 else None)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title(f"Imputed ({condition_name})")
        plt.imshow(img_test.astype(np.uint8), cmap='gray' if num_channels == 1 else None)
        plt.axis('off')

        plot_name = f'sample_{subject_id}_{condition_name}_slice_{slice_idx}.png'
        plt.savefig(os.path.join(plot_dir, plot_name), bbox_inches='tight')
        plt.close()


# =============================================================================
# PER-METHOD RUNNER (loop structure and indexing unchanged)
# =============================================================================
def list_subjects(method: MethodSpec, requested):
    if requested:
        return list(requested)
    if not os.path.isdir(method.directory):
        print(f"[{method.name}] directory not found: {method.directory}")
        return []
    return sorted(d for d in os.listdir(method.directory)
                  if os.path.isdir(os.path.join(method.directory, d)))


def run_method(method: MethodSpec, ref_dir: str, ref_pattern: str, out_dir: str,
               save_plots: str, plot_interval: int, subjects=None) -> pd.DataFrame:
    """Evaluate one method over all its subjects; returns the raw slice table."""
    plot_dir = os.path.join(out_dir, method.slug)
    os.makedirs(plot_dir, exist_ok=True)

    results_list: list = []
    subject_ids = list_subjects(method, subjects)
    print(f"\n=== Method: {method.name} ({len(subject_ids)} candidate subject(s)) ===")
    print(f"    source : {method.directory}")
    print(f"    pattern: {method.pattern}")

    for subject_id in subject_ids:
        print(f"\nProcessing Subject: {subject_id}...")

        # Define subject-specific paths
        impute_8_path = method.volume_path(subject_id, 8)
        impute_12_path = method.volume_path(subject_id, 12)

        # Check if imputation files exist before proceeding
        if not os.path.exists(impute_8_path) or not os.path.exists(impute_12_path):
            print(f"Skipping {subject_id}: Imputation files not found.")
            continue

        affs_orig, I_origs = [], []
        skip_subject = False

        for thick in REF_THICKNESSES:
            ref = os.path.join(ref_dir, subject_id, ref_pattern.format(thick=thick))

            if not os.path.exists(ref):
                print(f"Skipping {subject_id}: Missing reference {thick}mm file.")
                skip_subject = True
                break

            ref_vol, head_ref = MRIread(ref)
            I_orig, aff_orig, ap_flip = eugenios_closest_canonical(
                ref_vol, head_ref, return_ap_flip=True)

            voxsize = np.sqrt(np.sum(aff_orig[:-1, :-1] ** 2, axis=0))
            av_thickness = voxsize[1]
            affs_orig.append(av_thickness)
            I_origs.append(I_orig)

        if skip_subject:
            continue

        print(f"Loading volumes for {subject_id}...")
        vol8, head_imput8 = MRIread(impute_8_path)
        vol12, head_imput12 = MRIread(impute_12_path)

        # Establish dynamic data range for PSNR and SSIM based on the ground truth volume
        data_range = float(np.max(I_origs[0]) - np.min(I_origs[0]))

        kw = dict(method_name=method.name, results_list=results_list,
                  plot_dir=plot_dir, save_plots=save_plots,
                  plot_interval=plot_interval)

        # --- 8mm Imputation Evaluation ---
        for i in range(1, I_origs[1].shape[1]):
            idx = 2 * i - 1
            j = int(np.floor(affs_orig[0] * idx - affs_orig[0] / 2))

            if idx >= I_origs[0].shape[1] or j >= vol8.shape[1]:
                continue

            orig_slice = I_origs[0][:, idx]
            imput_slice = vol8[:, j]
            evaluate_slice(orig_slice, imput_slice, idx, '8mm', subject_id,
                           data_range, **kw)

        # --- 12mm Imputation Evaluation ---
        for i in range(1, I_origs[2].shape[1]):
            idx = 3 * i  # NOTE: 3*i targets kept slices. Use 3*i-1 or 3*i-2 for missing slices.
            j = int(np.floor(affs_orig[0] * idx - affs_orig[0] / 2))

            if idx >= I_origs[0].shape[1] or j >= vol12.shape[1]:
                continue

            orig_slice = I_origs[0][:, idx]
            imput_slice = vol12[:, j]
            evaluate_slice(orig_slice, imput_slice, idx, '12mm', subject_id,
                           data_range, **kw)

            idx = 3 * i + 1  # NOTE: 3*i targets kept slices. Use 3*i-1 or 3*i-2 for missing slices.
            j = int(np.floor(affs_orig[0] * idx - affs_orig[0] / 2))

            if idx >= I_origs[0].shape[1] or j >= vol12.shape[1]:
                continue

            orig_slice = I_origs[0][:, idx]
            imput_slice = vol12[:, j]
            evaluate_slice(orig_slice, imput_slice, idx, '12mm', subject_id,
                           data_range, **kw)

    # --- PROCESSING & SAVING RESULTS (per method) ---
    if not results_list:
        print(f"\n[{method.name}] No data was processed. "
              f"Please check the subject IDs and file paths.")
        return pd.DataFrame(columns=['Method', 'Subject', 'Condition',
                                     'Slice_Index', 'MAE', 'PSNR', 'SSIM'])

    df_raw = pd.DataFrame(results_list)

    # Save raw slice-by-slice metrics
    df_raw.to_csv(os.path.join(plot_dir, 'metrics_raw_slices.csv'), index=False)

    # ---------------------------------------------------------
    # TABLE 1: Per-Volume Results (Mean metrics per subject)
    # ---------------------------------------------------------
    df_per_volume = (df_raw.groupby(['Subject', 'Condition'])[['MAE', 'PSNR', 'SSIM']]
                     .mean().reset_index())
    df_per_volume.to_csv(os.path.join(plot_dir, 'metrics_per_volume.csv'), index=False)

    print("\n" + "=" * 60)
    print(f"TABLE 1: PER-VOLUME METRICS SUMMARY [{method.name}]")
    print("=" * 60)
    print(df_per_volume.to_string(index=False))

    # ---------------------------------------------------------
    # TABLE 2: Overall statistics across the entire cohort
    # ---------------------------------------------------------
    # We group by the per-volume dataframe so that volumes with more slices
    # don't unfairly weight the overall average.
    df_overall_summary = (df_per_volume.groupby('Condition')[['MAE', 'PSNR', 'SSIM']]
                          .agg(['mean', 'std']))
    df_overall_summary.to_csv(os.path.join(plot_dir,
                                           'metrics_overall_cohort_summary.csv'))

    print("\n" + "=" * 60)
    print(f"TABLE 2: OVERALL POPULATION SUMMARY (MEAN & STD ACROSS VOLUMES) [{method.name}]")
    print("=" * 60)
    print(df_overall_summary)

    return df_raw


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    p = argparse.ArgumentParser(
        description="Slice-wise consistency metrics (MAE / PSNR / SSIM) of one or "
                    "more imputation methods against the 4 mm reference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example method spec:\n'
               '  --method "Imputed:/data/02_imputations_unet:imputed_unet_{thick}mm.nii.gz"',
    )
    p.add_argument("--ref-dir", required=True,
                   help="Directory with the reference reconstructions "
                        "(one sub-directory per subject).")
    p.add_argument("--ref-pattern", default=REF_PATTERN_DEFAULT,
                   help=f"Reference file name pattern with a {{thick}} placeholder "
                        f"(default: {REF_PATTERN_DEFAULT}).")
    p.add_argument("--method", dest="methods", action="append", required=True,
                   type=parse_method, metavar="NAME:DIR:PATTERN",
                   help="Imputation method to evaluate; repeat for several methods.")
    p.add_argument("--output-dir", required=True,
                   help="Directory where per-method sub-directories are written.")
    p.add_argument("--subjects", nargs="+", default=None,
                   help="Optional explicit subject list; default is every "
                        "sub-directory of each method directory.")
    p.add_argument("--save-plots", choices=PLOT_POLICIES, default="interval",
                   help="Plot policy: every evaluated slice ('all'), every "
                        "--plot-interval slice ('interval', default), or none.")
    p.add_argument("--plot-interval", type=int, default=5,
                   help="Slice stride used when --save-plots interval (default: 5).")
    args = p.parse_args()

    if not os.path.isdir(args.ref_dir):
        raise SystemExit(f"Reference directory not found: {args.ref_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    frames = []
    for method in args.methods:
        frames.append(run_method(method, ref_dir=args.ref_dir,
                                 ref_pattern=args.ref_pattern,
                                 out_dir=args.output_dir,
                                 save_plots=args.save_plots,
                                 plot_interval=args.plot_interval,
                                 subjects=args.subjects))

    df_all = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
        if any(not f.empty for f in frames) else pd.DataFrame()

    if df_all.empty:
        print("\nNo data was processed for any method.")
        return

    per_volume = (df_all.groupby(['Method', 'Subject', 'Condition'])
                  [['MAE', 'PSNR', 'SSIM']].mean().reset_index())
    per_volume_path = os.path.join(args.output_dir,
                                   'metrics_per_volume_all_methods.csv')
    per_volume.to_csv(per_volume_path, index=False)

    overall = (per_volume.groupby(['Method', 'Condition'])
               [['MAE', 'PSNR', 'SSIM']].agg(['mean', 'std']))
    overall_path = os.path.join(args.output_dir,
                                'metrics_overall_cohort_summary_all_methods.csv')
    overall.to_csv(overall_path)

    print("\n" + "=" * 60)
    print("COMBINED SUMMARY (MEAN & STD ACROSS VOLUMES, BY METHOD)")
    print("=" * 60)
    print(overall)
    print("\nWrote:")
    print(f"  {per_volume_path}")
    print(f"  {overall_path}")


if __name__ == "__main__":
    main()