#!/usr/bin/env python3
"""
Consolidates the four sweep-writing blocks (UNet imputation, photo-reconstruction,
MRI, UNet+SynthSeg segmentation) into one script. The -slice lines, loop ranges and
frame naming are kept exactly as tested per volume. This also creates the folder tree,
writes run_all_videos.sh, and that script renders the frames and encodes the mp4s.

    python3 make_videos.py       # write sweeps + run_all_videos.sh
    bash run_all_videos.sh       # render frames with freeview, then encode mp4s
"""

import os
import nibabel as nib

ROOT      = "/home/marina/ms_thesis/photo_recon_uw"
OUT_ROOT  = "/home/marina/ms_thesis/evaluation_results/videos"
SWEEP_DIR = os.path.join(OUT_ROOT, "sweeps")
VIEWSIZE  = "750 750"
FRAMERATE = 10
LABELS    = "2,3,4,10,11,12,13,17,18,26,28,41,42,43,49,50,51,52,53,54,58,60"  # all but CSF(24)

os.makedirs(SWEEP_DIR, exist_ok=True)
jobs = []  # (name, plane, vol_arg, sweep_path, frames_dir)

def frames_dir(base, subdir, plane):
    d = os.path.join(OUT_ROOT, base, f"{subdir}_{plane}")
    os.makedirs(d, exist_ok=True)
    return d

# ============================ UNET IMPUTATION ============================
# vol_path = f"{ROOT}/02_imputations_unet/18-0086/imputed_unet_correct_8mm.nii.gz"
# vol_arg  = f"{vol_path}:rgb=true"
# img = nib.load(vol_path)
# nx, ny, nz = img.shape[:3]
# cx, cy, cz = nx // 2, ny // 2, nz // 2
# pad = max(3, len(str(nx - 1)))

# d = frames_dir("unet_frames", "unet_frames", "axial")
# sweep = os.path.join(SWEEP_DIR, "sweep_unet_axial.txt")
# with open(sweep, "w") as f:
#     f.write("-viewport axial\n")
#     for i in range(ny,0,-1): 
#         f.write(f"-slice {cx} {cz} {i}\n")
#         f.write(f"-ss {d}/frame_{-(i-ny):0{pad}d}.png 1\n")
#     f.write("-quit\n")
# jobs.append(("unet", "axial", vol_arg, sweep, d))

# d = frames_dir("unet_frames", "unet_frames", "sagittal")
# sweep = os.path.join(SWEEP_DIR, "sweep_unet_sagittal.txt")
# with open(sweep, "w") as f:
#     f.write("-viewport sagittal\n")
#     for i in range(ny):        # sweep sagittal (X); use range(a, b) to trim empty ends
#         f.write(f"-slice {i} {cz} {cy}\n")
#         f.write(f"-ss {d}/frame_{i:0{pad}d}.png 1\n")
#     f.write("-quit\n")
# jobs.append(("unet", "sagittal", vol_arg, sweep, d))

# ============================ PHOTO RECON ============================
# NOTE: this path matches the imputation block in your pasted code. Set the real
# baseline photo-reconstruction volume here, otherwise this renders the imputed one.
vol_path = f"{ROOT}/00_photo_recon/18-0086/photo_recon_8mm.nii.gz"
vol_arg  = f"{vol_path}:rgb=true"
img = nib.load(vol_path)
nx, ny, nz = img.shape[:3]
cx, cy, cz = nx // 2, ny // 2, nz // 2
pad = max(3, len(str(nx - 1)))

d = frames_dir("photo_recon_frames", "photo_recon_frames", "axial")
sweep = os.path.join(SWEEP_DIR, "sweep_photo_recon_axial.txt")
with open(sweep, "w") as f:
    f.write("-viewport axial\n")
    for i in range(ny):
        f.write(f"-slice {i} {cy} {cz}\n")
        f.write(f"-ss {d}/frame_{i:0{pad}d}.png 1\n")
    f.write("-quit\n")
jobs.append(("photo_recon", "axial", vol_arg, sweep, d))

d = frames_dir("photo_recon_frames", "photo_recon_frames", "sagittal")
sweep = os.path.join(SWEEP_DIR, "sweep_photo_recon_sagittal.txt")
with open(sweep, "w") as f:
    f.write("-viewport sagittal\n")
    for i in range(ny):
        f.write(f"-slice {cx} {i} {cz}\n")
        f.write(f"-ss {d}/frame_{i:0{pad}d}.png 1\n")
    f.write("-quit\n")
jobs.append(("photo_recon", "sagittal", vol_arg, sweep, d))

# ============================ MRI ============================
# vol_path = f"{ROOT}/00_photo_recon/18-0086/mri.deformed.mgz"
# vol_arg  = vol_path
# img = nib.load(vol_path)
# nx, ny, nz = img.shape[:3]
# cx, cy, cz = nx // 2, ny // 2, nz // 2
# pad = max(3, len(str(nx - 1)))

# d = frames_dir("mri_frames", "mri_frames", "axial")
# sweep = os.path.join(SWEEP_DIR, "sweep_mri_axial.txt")
# with open(sweep, "w") as f:
#     f.write("-viewport axial\n")
#     for i in range(nx):
#         f.write(f"-slice {i} {cy} {cz}\n")
#         f.write(f"-ss {d}/frame_{i:0{pad}d}.png 1\n")
#     f.write("-quit\n")
# jobs.append(("mri", "axial", vol_arg, sweep, d))

# d = frames_dir("mri_frames", "mri_frames", "sagittal")
# sweep = os.path.join(SWEEP_DIR, "sweep_mri_sagittal.txt")
# with open(sweep, "w") as f:
#     f.write("-viewport sagittal\n")
#     for i in range(ny):
#         f.write(f"-slice {cx} {i} {cz}\n")
#         f.write(f"-ss {d}/frame_{i:0{pad}d}.png 1\n")
#     f.write("-quit\n")
# jobs.append(("mri", "sagittal", vol_arg, sweep, d))

# ============================ UNET SEGMENTATION ============================
# vol_path = f"{ROOT}/04_unet_synthseg/18-0086/synthseg_mri.mgz"
# vol_arg  = f"{vol_path}:colormap=lut:select_label={LABELS}"
# img = nib.load(vol_path)
# nx, ny, nz = img.shape[:3]
# cx, cy, cz = nx // 2, ny // 2, nz // 2
# pad = max(3, len(str(nx - 1)))

# d = frames_dir("mri_segmentation", "mri_segmentation_frames", "axial")
# sweep = os.path.join(SWEEP_DIR, "sweep_seg_axial.txt")
# with open(sweep, "w") as f:
#     f.write("-viewport axial\n")
#     for i in range(nx):
#         f.write(f"-slice {i} {cy} {cz}\n")
#         f.write(f"-ss {d}/frame_{i:0{pad}d}.png 1\n")
#     f.write("-quit\n")
# jobs.append(("mri_segmentation", "axial", vol_arg, sweep, d))

# d = frames_dir("mri_segmentation", "mri_segmentation_frames", "sagittal")
# sweep = os.path.join(SWEEP_DIR, "sweep_seg_sagittal.txt")
# with open(sweep, "w") as f:
#     f.write("-viewport sagittal\n")
#     for i in range(ny):
#         f.write(f"-slice {cx} {i} {cz}\n")
#         f.write(f"-ss {d}/frame_{i:0{pad}d}.png 1\n")
#     f.write("-quit\n")
# jobs.append(("mri_segmentation", "sagittal", vol_arg, sweep, d))

# ============================ BASH RUNNER ============================
lines = ["#!/usr/bin/env bash", "# generated by make_videos.py", "set -uo pipefail", ""]
for name, plane, vol_arg, sweep, d in jobs:
    video = os.path.join(os.path.dirname(d), f"{name}_{plane}.mp4")
    lines += [
        f"echo '>> {name} {plane}'",
        f'xvfb-run -a freeview -v "{vol_arg}" \\',
        f"    -viewport {plane} -layout 1 -viewsize {VIEWSIZE} \\",
        f'    -nocursor -noquit -cmd "{sweep}"',
        f"ffmpeg -y -framerate {FRAMERATE} -pattern_type glob \\",
        f'    -i "{d}/frame_*.png" \\',
        f'    -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \\',
        f'    "{video}"',
        "",
    ]

run_path = os.path.join(OUT_ROOT, "run_all_videos.sh")
with open(run_path, "w") as f:
    f.write("\n".join(lines) + "\n")
os.chmod(run_path, 0o755)

for name, plane, _, sweep, d in jobs:
    print(f"{name:18s} {plane:9s} -> {d}")
print(f"\nsweeps in {SWEEP_DIR}")
print(f"runner    {run_path}")
print(f"run:      bash {run_path}")