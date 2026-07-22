#!/usr/bin/env python3
"""Slice-wise pixel-consistency evaluation of slab-thickness imputations.

Computes MAE, PSNR and SSIM between imputed volumes and their ground-truth
photo-reconstruction references at 8 mm and 12 mm slab thickness, then writes
per-slice, per-volume and cohort-level summaries to CSV. Two imputation
conventions are supported out of the box: UNet ("imputed_unet_{8,12}mm.nii.gz")
and tricubic ("photo_recon_{8,12}mm_tricubic.nii.gz").

Input layout
    <ref-dir>/<subject>/photo_recon_correct_{4,8,12}mm.nii.gz
    <impute-dir>/<subject>/<8 mm imputed>, <12 mm imputed>

Output layout
    <output-dir>/<method>/metrics_raw_slices.csv
    <output-dir>/<method>/metrics_per_volume.csv
    <output-dir>/<method>/metrics_overall_cohort_summary.csv
    <output-dir>/<method>/plots/sample_<subject>_<condition>_slice_<idx>.png

Example
    python evaluate_imputations.py \\
        --ref-dir    /data/photo_recon_uw/00_photo_recon \\
        --impute-dir /data/photo_recon_uw/02_imputations_unet \\
        --method unet
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for batch figure export
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio

from ext import MRIread, eugenios_closest_canonical


# --- Method -> imputed-filename convention -----------------------------------
METHOD_FILENAMES = {
    "unet": {"8mm": "imputed_unet_8mm.nii.gz",
             "12mm": "imputed_unet_12mm.nii.gz"},
    "tricubic": {"8mm": "photo_recon_8mm_tricubic.nii.gz",
                 "12mm": "photo_recon_12mm_tricubic.nii.gz"},
}
REF_TEMPLATE = "photo_recon_correct_{thick}mm.nii.gz"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ref-dir", required=True,
                   help="Base directory of ground-truth reconstructions.")
    p.add_argument("--impute-dir", required=True,
                   help="Base directory of imputed volumes.")
    p.add_argument("--output-dir", default="./evaluation_results/imputations",
                   help="Base output directory; results go to <output-dir>/<method>/.")
    p.add_argument("--method", choices=sorted(METHOD_FILENAMES), default="unet",
                   help="Imputation method selecting the default filename convention.")
    p.add_argument("--name-8mm", default=None,
                   help="Override the 8 mm imputed filename.")
    p.add_argument("--name-12mm", default=None,
                   help="Override the 12 mm imputed filename.")
    p.add_argument("--subjects", nargs="+", default=None,
                   help="Explicit subject IDs (default: every entry in --impute-dir).")
    p.add_argument("--plot-interval", type=int, default=5,
                   help="Save one comparison plot every N slices.")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--save-all-plots", action="store_true",
                     help="Save a comparison plot for every evaluated slice.")
    grp.add_argument("--no-plots", action="store_true",
                     help="Disable plotting entirely (metrics only).")
    return p.parse_args()


def make_should_plot(save_all, no_plots, interval):
    """Return a predicate deciding whether slice `idx` should be plotted."""
    if no_plots:
        return lambda idx: False
    if save_all:
        return lambda idx: True
    return lambda idx: (idx % interval == 0)


def _save_comparison_plot(img_true, img_test, num_channels, slice_idx,
                          condition, subject_id, mae, psnr, ssim_val, plots_dir):
    cmap = "gray" if num_channels == 1 else None
    plt.figure(figsize=(10, 5))
    plt.suptitle(f"{subject_id} | {condition} - Slice {slice_idx}\n"
                 f"MAE: {mae:.4f} | PSNR: {psnr:.2f}dB | SSIM: {ssim_val:.4f}")
    plt.subplot(1, 2, 1)
    plt.title("Original (4mm)")
    plt.imshow(img_true.astype(np.uint8), cmap=cmap)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title(f"Imputed ({condition})")
    plt.imshow(img_test.astype(np.uint8), cmap=cmap)
    plt.axis("off")
    fname = f"sample_{subject_id}_{condition}_slice_{slice_idx}.png"
    plt.savefig(os.path.join(plots_dir, fname), bbox_inches="tight")
    plt.close()


def evaluate_slice(orig_slice, imput_slice, slice_idx, condition, subject_id,
                   data_range, plots_dir, save_plot):
    """Compute MAE/PSNR/SSIM for one slice; optionally save a comparison plot."""
    img_true = orig_slice.astype(np.float32)
    img_test = imput_slice.astype(np.float32)

    mae_val = float(nn.L1Loss()(torch.tensor(img_true), torch.tensor(img_test)))
    psnr_val = peak_signal_noise_ratio(img_true, img_test, data_range=data_range)
    if psnr_val == float("inf") and mae_val == 0:
        psnr_val = 100.0  # identical slices: PSNR undefined, use a finite sentinel

    num_channels = img_test.shape[-1] if img_test.ndim > 2 else 1
    if num_channels > 1:
        ssim_val = float(np.mean([
            ssim(img_true[..., c], img_test[..., c], data_range=data_range)
            for c in range(num_channels)
        ]))
    else:
        ssim_val = float(ssim(img_true, img_test, data_range=data_range))

    if save_plot:
        _save_comparison_plot(img_true, img_test, num_channels, slice_idx,
                              condition, subject_id, mae_val, psnr_val,
                              ssim_val, plots_dir)

    return {"Subject": subject_id, "Condition": condition,
            "Slice_Index": slice_idx, "MAE": mae_val,
            "PSNR": psnr_val, "SSIM": ssim_val}


def load_references(subject_id, ref_dir):
    """Load 4/8/12 mm references in canonical orientation.

    Returns (thicknesses, volumes) where `thicknesses[k]` is the mean slice
    thickness of reference k, or None if any reference is missing.
    """
    thicknesses, volumes = [], []
    for thick in ("4", "8", "12"):
        ref = os.path.join(ref_dir, subject_id, REF_TEMPLATE.format(thick=thick))
        if not os.path.exists(ref):
            print(f"Skipping {subject_id}: missing reference {thick} mm file.")
            return None
        ref_vol, head_ref = MRIread(ref)
        I_orig, aff_orig, _ = eugenios_closest_canonical(
            ref_vol, head_ref, return_ap_flip=True)
        voxsize = np.sqrt(np.sum(aff_orig[:-1, :-1] ** 2, axis=0))
        thicknesses.append(voxsize[1])  # slice axis is axis 1
        volumes.append(I_orig)
    return thicknesses, volumes


def evaluate_condition(I_origs, thicknesses, imputed_vol, ref_level, idx_fn,
                       condition, subject_id, data_range, plots_dir, should_plot):
    """Evaluate one slab-thickness condition (8 mm or 12 mm) for one subject.

    `ref_level` indexes the driving reference (1 -> 8 mm, 2 -> 12 mm);
    `idx_fn(i)` maps the downsampled index i to the full-resolution slice index.
    """
    records = []
    for i in range(1, I_origs[ref_level].shape[1]):
        j = int(np.ceil(thicknesses[ref_level] * i))
        idx = idx_fn(i)
        if idx >= I_origs[0].shape[1] or j >= imputed_vol.shape[1]:
            continue
        records.append(evaluate_slice(
            I_origs[0][:, idx], imputed_vol[:, j], idx, condition,
            subject_id, data_range, plots_dir, should_plot(idx)))
    return records


def process_subject(subject_id, ref_dir, impute_dir, filenames,
                    plots_dir, should_plot):
    """Return the list of per-slice metric records for one subject."""
    impute_8 = os.path.join(impute_dir, subject_id, filenames["8mm"])
    impute_12 = os.path.join(impute_dir, subject_id, filenames["12mm"])
    if not (os.path.exists(impute_8) and os.path.exists(impute_12)):
        print(f"Skipping {subject_id}: imputation files not found.")
        return []

    refs = load_references(subject_id, ref_dir)
    if refs is None:
        return []
    thicknesses, I_origs = refs

    print(f"Loading imputed volumes for {subject_id} ...")
    vol8, _ = MRIread(impute_8)
    vol12, _ = MRIread(impute_12)

    data_range = float(np.max(I_origs[0]) - np.min(I_origs[0]))

    records = []
    records += evaluate_condition(
        I_origs, thicknesses, vol8, ref_level=1, idx_fn=lambda i: 2 * i - 1,
        condition="8mm", subject_id=subject_id, data_range=data_range,
        plots_dir=plots_dir, should_plot=should_plot)
    # NOTE: idx = 3*i targets the *kept* slices. Use 3*i-1 or 3*i-2 to score
    # the interpolated (missing) slices instead.
    records += evaluate_condition(
        I_origs, thicknesses, vol12, ref_level=2, idx_fn=lambda i: 3 * i,
        condition="12mm", subject_id=subject_id, data_range=data_range,
        plots_dir=plots_dir, should_plot=should_plot)
    return records


def write_results(records, out_dir):
    """Write raw, per-volume and cohort-level CSV summaries; echo the tables."""
    if not records:
        print("\nNo data was processed. Check subject IDs and file paths.")
        return

    df_raw = pd.DataFrame(records)
    df_raw.to_csv(os.path.join(out_dir, "metrics_raw_slices.csv"), index=False)

    df_per_volume = (df_raw.groupby(["Subject", "Condition"])
                     [["MAE", "PSNR", "SSIM"]].mean().reset_index())
    df_per_volume.to_csv(os.path.join(out_dir, "metrics_per_volume.csv"),
                         index=False)

    # Group on the per-volume table so slice count does not skew the cohort mean.
    df_overall = (df_per_volume.groupby("Condition")
                  [["MAE", "PSNR", "SSIM"]].agg(["mean", "std"]))
    df_overall.to_csv(os.path.join(out_dir, "metrics_overall_cohort_summary.csv"))

    print("\n" + "=" * 60)
    print("TABLE 1: PER-VOLUME METRICS SUMMARY")
    print("=" * 60)
    print(df_per_volume.to_string(index=False))
    print("\n" + "=" * 60)
    print("TABLE 2: COHORT SUMMARY (MEAN & STD ACROSS VOLUMES)")
    print("=" * 60)
    print(df_overall)


def main():
    args = parse_args()

    filenames = dict(METHOD_FILENAMES[args.method])
    if args.name_8mm:
        filenames["8mm"] = args.name_8mm
    if args.name_12mm:
        filenames["12mm"] = args.name_12mm

    method_dir = os.path.join(args.output_dir, args.method)
    plots_dir = os.path.join(method_dir, "plots")
    os.makedirs(method_dir, exist_ok=True)
    if not args.no_plots:
        os.makedirs(plots_dir, exist_ok=True)

    should_plot = make_should_plot(args.save_all_plots, args.no_plots,
                                   args.plot_interval)
    subjects = args.subjects or sorted(os.listdir(args.impute_dir))

    records = []
    for subject_id in subjects:
        print(f"\nProcessing subject: {subject_id} ...")
        records += process_subject(subject_id, args.ref_dir, args.impute_dir,
                                   filenames, plots_dir, should_plot)

    write_results(records, method_dir)


if __name__ == "__main__":
    main()