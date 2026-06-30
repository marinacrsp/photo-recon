import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

# Assuming these are your custom external functions/classes
from ext import MRIread, eugenios_closest_canonical

# --- CONFIGURATION ---
# Define your list of subject IDs here
SUBJECT_IDS = ['18-0086', '18-0087', '18-0088'] 

# Base directories to construct paths dynamically
BASE_REF_DIR = '/home/marina/ms_thesis/photo_recon_uw/00_photo_recon'
BASE_IMPUTE_DIR = '/home/marina/ms_thesis/photo_recon_uw/photo_reconstructions'

output_dir = './evaluation_results'
os.makedirs(output_dir, exist_ok=True)

# Plotting control
SAVE_ALL_PLOTS = False
PLOT_INTERVAL = 4

# Global list to hold slice-by-slice metric dictionaries across ALL subjects
results_list = []

# --- EVALUATION FUNCTION ---
def evaluate_slice(orig_slice, imput_slice, slice_idx, condition_name, subject_id, data_range):
    """Computes evaluation metrics and optionally saves comparison plots."""
    img_true = orig_slice.astype(np.float32)
    img_test = imput_slice.astype(np.float32)
    
    # Calculate metrics
    mse_val = mean_squared_error(img_true, img_test)
    psnr_val = peak_signal_noise_ratio(img_true, img_test, data_range=data_range)
    
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
        'Subject': subject_id,
        'Condition': condition_name,
        'Slice_Index': slice_idx,
        'MSE': mse_val,
        'PSNR': psnr_val,
        'SSIM': ssim_val
    })
    
    # Plotting logic
    should_plot = SAVE_ALL_PLOTS or (slice_idx % PLOT_INTERVAL == 0)
    if should_plot:
        plt.figure(figsize=(10, 5))
        plt.suptitle(f"{subject_id} | {condition_name} - Slice {slice_idx}\nMSE: {mse_val:.4f} | PSNR: {psnr_val:.2f}dB | SSIM: {ssim_val:.4f}")
        
        plt.subplot(1, 2, 1)
        plt.title("Original (4mm)")
        plt.imshow(img_true.astype(np.uint8), cmap='gray' if num_channels == 1 else None)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title(f"Imputed ({condition_name})")
        plt.imshow(img_test.astype(np.uint8), cmap='gray' if num_channels == 1 else None)
        plt.axis('off')
        
        plot_name = f'sample_{subject_id}_{condition_name}_slice_{slice_idx}.png'
        plt.savefig(os.path.join(output_dir, plot_name), bbox_inches='tight')
        plt.close()

# --- MAIN RUNNER LOOP ---
for subject_id in SUBJECT_IDS:
    print(f"\nProcessing Subject: {subject_id}...")
    
    # Define subject-specific paths
    impute_8_path = os.path.join(BASE_IMPUTE_DIR, subject_id, 'imputation_8mm.mgz')
    impute_12_path = os.path.join(BASE_IMPUTE_DIR, subject_id, 'imputation_12mm.mgz')
    
    # Check if imputation files exist before proceeding
    if not os.path.exists(impute_8_path) or not os.path.exists(impute_12_path):
        print(f"Skipping {subject_id}: Imputation files not found.")
        continue

    affs_orig, I_origs = [], []
    skip_subject = False
    
    for thick in ['4', '8', '12']:
        ref = os.path.join(BASE_REF_DIR, subject_id, f'photo_recon_correct_{thick}mm.nii.gz')
        
        if not os.path.exists(ref):
            print(f"Skipping {subject_id}: Missing reference {thick}mm file.")
            skip_subject = True
            break

        ref_vol, head_ref = MRIread(ref)
        I_orig, aff_orig, ap_flip = eugenios_closest_canonical(ref_vol, head_ref, return_ap_flip=True)
        
        voxsize = np.sqrt(np.sum(aff_orig[:-1,:-1]**2, axis=0))
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

    # --- 8mm Imputation Evaluation ---
    for i in range(1, I_origs[1].shape[1]):
        j = int(np.ceil(affs_orig[1]*i))
        idx = 2*i - 1
        
        if idx >= I_origs[0].shape[1] or j >= vol8.shape[1]:
            continue

        orig_slice = I_origs[0][:, idx]
        imput_slice = vol8[:, j] 
        evaluate_slice(orig_slice, imput_slice, idx, '8mm', subject_id, data_range)

    # --- 12mm Imputation Evaluation ---
    for i in range(1, I_origs[2].shape[1]):
        j = int(np.ceil(affs_orig[2]*i))
        idx = 3*i  # NOTE: 3*i targets kept slices. Use 3*i-1 or 3*i-2 for missing slices.
        
        if idx >= I_origs[0].shape[1] or j >= vol12.shape[1]:
            continue

        orig_slice = I_origs[0][:, idx]
        imput_slice = vol12[:, j] 
        evaluate_slice(orig_slice, imput_slice, idx, '12mm', subject_id, data_range)

# --- PROCESSING & SAVING RESULTS ---
if not results_list:
    print("\nNo data was processed. Please check your subject IDs and file paths.")
else:
    df_raw = pd.DataFrame(results_list)
    
    # Save raw slice-by-slice metrics
    df_raw.to_csv(os.path.join(output_dir, 'metrics_raw_slices.csv'), index=False)

    # ---------------------------------------------------------
    # TABLE 1: Per-Volume Results (Mean metrics per subject)
    # ---------------------------------------------------------
    df_per_volume = df_raw.groupby(['Subject', 'Condition'])[['MSE', 'PSNR', 'SSIM']].mean().reset_index()
    df_per_volume.to_csv(os.path.join(output_dir, 'metrics_per_volume.csv'), index=False)
    
    print("\n" + "="*60)
    print("TABLE 1: PER-VOLUME METRICS SUMMARY")
    print("="*60)
    print(df_per_volume.to_string(index=False))

    # ---------------------------------------------------------
    # TABLE 2: Overall statistics across the entire cohort
    # ---------------------------------------------------------
    # We group by the per-volume dataframe so that volumes with more slices 
    # don't unfairly weight the overall average.
    df_overall_summary = df_per_volume.groupby('Condition')[['MSE', 'PSNR', 'SSIM']].agg(['mean', 'std'])
    df_overall_summary.to_csv(os.path.join(output_dir, 'metrics_overall_cohort_summary.csv'))
    
    print("\n" + "="*60)
    print("TABLE 2: OVERALL POPULATION SUMMARY (MEAN & STD ACROSS VOLUMES)")
    print("="*60)
    print(df_overall_summary)