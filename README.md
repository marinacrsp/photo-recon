# Improving Neuropathological Reconstruction Fidelity via AI Slice Imputation

A 2D U-Net that imputes intermediate coronal slices to turn anisotropic 3D reconstructions
of dissection photographs into anatomically consistent, near-isotropic volumes. The network
is trained entirely on domain-randomized synthetic data generated on the fly from 1 mm
isotropic MRI, so it generalizes across photograph contrasts and across slab thicknesses that
are not known a priori. This is the imputation step described in *Improving Neuropathological
Reconstruction Fidelity via AI Slice Imputation* (see [References](#references)).

## Overview

Dissection photographs are routinely acquired by brain banks but the slabs are several
millimeters thick, leaving large gaps between slices after 3D reconstruction. This codebase
trains and applies a super-resolution model that fills those gaps: given two acquired coronal
slices that bracket a missing location, plus their distances to it, the model predicts the
slice in between. Applied slice by slice, it reconstructs a high-resolution isotropic volume
suitable for downstream atlas registration, segmentation, and volumetry.


## Repository layout

The scripts add their parent directory to `sys.path`, so the modules below must sit one level
above the scripts.

```
project_root/
├── ext/                      # project utilities
├── generators.py             # synthetic triplet generator (hemi_generator)
└── scripts/
    ├── train.py              # training and validation loops
    └── sample.py             # inference on a real stack of dissection photographs
```

## Installation

Full instructions are in `SETUP.md`. In brief, on Linux with an NVIDIA GPU:

```bash
python3.11 -m venv ~/envs/photo-imputation && source ~/envs/photo-imputation/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match cuXXX to your driver
pip install -r requirements.txt
```

The PyPI dependencies are `torch`, `numpy`, `nibabel`, `matplotlib`, and `opencv-python`.
A compatible NVIDIA driver is required; a system CUDA toolkit is not, since the PyTorch
wheels bundle their own runtime. `ext`, `unet`, and `generators` come from this repository.

## Training Data
Training data was recovered from the following 10 publicly available datasets: ABIDE [28], ADHD200
[29], ADNI [30], AIBL [31], COBRE [32], Chinese-HCP [33], HCP [34], ISBI2015 [35],
MCIC [36], and OASIS3 [37]

## Usage

### Training

Configuration is set at the top of `train.py` (data directory, output directory,
device, U-Net width, spacing limits, epochs). Edit the `# TODO` lines, then run:

```bash
python scripts/train.py
```

Checkpoints are written to `output_directory`. Validation uses a fixed synthetic set
materialized once before training and reused every epoch, with the inter-slice spacing fixed at
the midpoint of `spacing_limits` (`mid_loc=True`). 

### Inference

```bash
python scripts/sample.py \
    --model_path /path/to/model_weights.pth \
    --input_file /path/to/photo_reconstruction.mgz \
    --illumination <value> \
    --unsharp_sigma 1.0 \
    --unsharp_amount 1.0
```

`sample.py` reorients the input to a canonical frame, resamples in-plane to 1 mm, normalizes,
and for each output coronal slice imputes from the two nearest acquired slices, adds the linear
interpolation baseline, denormalizes, applies unsharp masking, and writes
`photo_recon.imputation.mgz`. On a headless server set `export MPLBACKEND=Agg` first.

Download `model_weights.pth` from: https://ftp.nmr.mgh.harvard.edu/pub/dist/lcnpublic/dist/dissection_photo_model/photo_imputation_unet.pth

## References

This repository contains the slice-imputation method proposed in the manuscript below. The bracketed numbers match the numbering of that manuscript's bibliography.

**Manuscript**

- M. Crespo Aguirre, J. Williams-Ramirez, D. Zemlyanker, X. Hu, L. J. Deden-Binder, R. Herisse,
  M. Montine, T. R. Connors, C. Mount, C. L. MacDonald, C. D. Keene, C. S. Latimer, D. H. Oakley,
  B. T. Hyman, A. Lawry Aguila, and J. E. Iglesias, "Improving Neuropathological Reconstruction
  Fidelity via AI Slice Imputation," arXiv:2602.00669, 2026. 

**Training datasets**

- [28] A. Di Martino, C.-G. Yan, Q. Li, E. Denio, F. X. Castellanos, K. Alaerts, J. S. Anderson,
  M. Assaf, S. Y. Bookheimer, M. Dapretto, et al., "The autism brain imaging data exchange:
  towards a large-scale evaluation of the intrinsic brain architecture in autism," *Molecular
  Psychiatry*, vol. 19, no. 6, pp. 659-667, 2014.
- [29] M. R. Brown, G. S. Sidhu, R. Greiner, N. Asgarian, M. Bastani, P. H. Silverstone,
  A. J. Greenshaw, and S. M. Dursun, "ADHD-200 global competition: diagnosing ADHD using personal
  characteristic data can outperform resting state fMRI measurements," *Frontiers in Systems
  Neuroscience*, vol. 6, p. 69, 2012.
- [30] C. R. Jack Jr, M. A. Bernstein, N. C. Fox, P. Thompson, G. Alexander, D. Harvey,
  B. Borowski, P. J. Britson, J. L. Whitwell, C. Ward, et al., "The Alzheimer's Disease
  Neuroimaging Initiative (ADNI): MRI methods," *Journal of Magnetic Resonance Imaging*, vol. 27,
  no. 4, pp. 685-691, 2008.
- [31] C. Fowler, S. R. Rainey-Smith, S. Bird, J. Bomke, P. Bourgeat, B. M. Brown, S. C. Burnham,
  A. I. Bush, C. Chadunow, S. Collins, et al., "Fifteen years of the Australian Imaging,
  Biomarkers and Lifestyle (AIBL) study: progress and observations from 2,359 older adults
  spanning the spectrum from cognitive normality to Alzheimer's disease," *Journal of Alzheimer's
  Disease Reports*, vol. 5, no. 1, pp. 443-468, 2021.
- [32] A. R. Mayer, D. Ruhl, F. Merideth, J. Ling, F. M. Hanlon, J. Bustillo, and J. Canive,
  "Functional imaging of the hemodynamic sensory gating response in schizophrenia," *Human Brain
  Mapping*, vol. 34, no. 9, pp. 2302-2312, 2013.
- [33] N. Vogt, "The Chinese Human Connectome Project," *Nature Methods*, vol. 20, no. 2, p. 177,
  2023.
- [34] D. C. Van Essen, K. Ugurbil, E. Auerbach, D. Barch, T. E. Behrens, R. Bucholz, A. Chang,
  L. Chen, M. Corbetta, S. W. Curtiss, et al., "The Human Connectome Project: a data acquisition
  perspective," *NeuroImage*, vol. 62, no. 4, pp. 2222-2231, 2012.
- [35] A. Carass, S. Roy, A. Jog, J. L. Cuzzocreo, E. Magrath, A. Gherman, J. Button, J. Nguyen,
  F. Prados, C. H. Sudre, et al., "Longitudinal multiple sclerosis lesion segmentation: resource
  and challenge," *NeuroImage*, vol. 148, pp. 77-102, 2017.
- [36] R. L. Gollub, J. M. Shoemaker, M. D. King, T. White, S. Ehrlich, S. R. Sponheim,
  V. P. Clark, J. A. Turner, B. A. Mueller, V. Magnotta, et al., "The MCIC collection: a shared
  repository of multi-modal, multi-site brain image data from a clinical investigation of
  schizophrenia," *Neuroinformatics*, vol. 11, no. 3, pp. 367-388, 2013.
- [37] P. J. LaMontagne, T. L. Benzinger, J. C. Morris, S. Keefe, R. Hornbeck, C. Xiong, E. Grant,
  J. Hassenstab, K. Moulder, A. G. Vlassenko, et al., "OASIS-3: longitudinal neuroimaging,
  clinical, and cognitive dataset for normal aging and Alzheimer disease," *medRxiv*,
  pp. 2019-12, 2019.
