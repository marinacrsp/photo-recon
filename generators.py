"""
Script to generate random triplets of coronal slices from a single hemisphere, for training the imputation network. 
The generator applies random affine and non-linear deformations, bias field, gamma correction, and blurring to the input images, and generates random triplets of coronal slices with specified spacing between them
"""
import argparse
import glob
import os
import numpy as np
import torch
from torch.nn.functional import conv3d
import nibabel as nib
from ext import myzoom_torch, gaussian_blur_3d, get_noninteger_coronal_slice, fast_3D_interp_torch

def hemi_generator(datadir,
              spacing_limits=[2, 12],
              flipping=True,
              provide_2d_gradients=False,
              labels_to_kill=None,
              siz=[160, 160],
              batchsize = 10,
              real_mix_prob=0.25,
              max_rotation=15,
              max_shear=0.2,
              max_scaling=0.2,
              nonlin_scale_min=0.03,
              nonlin_scale_max=0.06,
              nonlin_std_max=4,
              bf_scale_min=0.02,
              bf_scale_max=0.04,
              bf_std_min=0.1,
              bf_std_max=0.6,
              gamma_std=0.1,
              sigma_blur_min=0.1,
              sigma_blur_max=0.75,
              dtype=torch.float32,
              mid_loc:bool=False,
              device:str='cpu'):

    mid_loc = 0.5 if mid_loc is True else np.random.rand()  
    # Collect list of available images, per dataset
    datasets = []
    g = glob.glob(os.path.join(datadir, '*' + 'T1w.nii'))
    for i in range(len(g)):
        filename = os.path.basename(g[i])
        dataset = filename[:filename.find('.')]
        found = False
        for d in datasets:
            if dataset==d:
                found = True
        if found is False:
            datasets.append(dataset)
    print('Found ' + str(len(datasets)) + ' datasets with ' + str(len(g)) + ' scans in total')
    names = []
    for i in range(len(datasets)):
        names.append(glob.glob(os.path.join(datadir, datasets[i] + '.*' + 'T1w.nii')))

    # Get resolution and maximum modeled distance for white / pial surfaces
    aux = nib.load(names[0][0])
    res_training_data = np.sum(aux.affine ** 2, axis=0)[:-1]

    # to cover the whole hemisphere
    gensize = np.array([siz[0], 256, siz[1]]).astype(int)

    # Sobel operators, to compute image gradients
    sobel_x = 0.125 * torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=dtype, device=device).view((1, 1, 3, 3))
    sobel_y = 0.125 * torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=dtype, device=device).view((1, 1, 3, 3))

    with torch.no_grad():
        # prepare grid
        print('Preparing grid...')
        xx, yy, zz = np.meshgrid(range(gensize[0]), range(gensize[1]), range(gensize[2]), sparse=False, indexing='ij')
        xx = torch.tensor(xx, dtype=dtype, device=device)
        yy = torch.tensor(yy, dtype=dtype, device=device)
        zz = torch.tensor(zz, dtype=dtype, device=device)
        c = torch.tensor((np.array(gensize) - 1) / 2, dtype=dtype, device=device)
        xc = xx - c[0]
        yc = yy - c[1]
        zc = zz - c[2]

        # Array to kill background labels in photo mode if needed
        lut_kill = torch.arange(0, 10000, dtype=torch.int32, device=device)
        for lln in labels_to_kill:
            lut_kill[lln] = 0

        print('Generator is ready!')

        while True:

            # Select random case
            d_idx = np.random.randint(len(datasets))
            idx = np.random.randint(len(names[d_idx]))
            t1 = names[d_idx][idx]
            t2 = names[d_idx][idx][:-7] + 'T2w.nii'
            flair = names[d_idx][idx][:-7] + 'FLAIR.nii'
            generation_labels = names[d_idx][idx][:-7] + 'generation_labels.nii'
            segmentation_labels = names[d_idx][idx][:-7] + 'brainseg.nii'
            spacing_simulation = spacing_limits[0] + np.random.rand() * (spacing_limits[1] - spacing_limits[0])

            t2 = t2 if os.path.isfile(t2) else None
            flair = flair if os.path.isfile(flair) else None

            # Load generation labels off the bat
            Gimg = nib.load(generation_labels)

            # sample affine deformation
            rotations = (2 * max_rotation * np.random.rand(3) - max_rotation) / 180.0 * np.pi
            shears = (2 * max_shear * np.random.rand(3) - max_shear)
            scalings = 1 + (2 * max_scaling * np.random.rand(3) - max_scaling)
            scaling_factor_distances = np.prod(scalings) ** .33333333333 # we divide distance maps by this, not perfect, but better than nothing
            A = torch.tensor(make_affine_matrix(rotations, shears, scalings), dtype=torch.float, device=device)

            # sample center
            c2 = torch.tensor((np.array(Gimg.shape[0:3]) - 1)/2, dtype=dtype, device=device)

            # sample nonlinear deformation (photos a bit special)
            nonlin_scale = nonlin_scale_min + np.random.rand(1) * (nonlin_scale_max - nonlin_scale_min)
            siz_F_small = np.round(nonlin_scale * np.array(gensize)).astype(int).tolist()
            siz_F_small[1] = np.round(siz[1] / spacing_simulation).astype(int) # photos!
            nonlin_std = nonlin_std_max * np.random.rand()
            Fsmall = nonlin_std * torch.randn([*siz_F_small, 3], dtype=dtype, device=device)
            F = myzoom_torch(Fsmall, np.array(gensize) / siz_F_small, device)
            F[:, :, :, 1] = 0

            # deformed coordinates (we do nonlinear "first" ie after so we can do heavy coronal deformations in photo mode)
            xx1 = xc + F[:, :, :, 0]
            yy1 = yc + F[:, :, :, 1]
            zz1 = zc + F[:, :, :, 2]
            xx2 = A[0, 0] * xx1 + A[0, 1] * yy1 + A[0, 2] * zz1 + c2[0]
            yy2 = A[1, 0] * xx1 + A[1, 1] * yy1 + A[1, 2] * zz1 + c2[1]
            zz2 = A[2, 0] * xx1 + A[2, 1] * yy1 + A[2, 2] * zz1 + c2[2]

            # Get the margins for reading images
            x1 = max(0, torch.floor(torch.min(xx2)).int().cpu().numpy())
            y1 = max(0, torch.floor(torch.min(yy2)).int().cpu().numpy())
            z1 = max(0, torch.floor(torch.min(zz2)).int().cpu().numpy())
            x2 = min(Gimg.shape[0], 1 + torch.ceil(torch.max(xx2)).int().cpu().numpy())
            y2 = min(Gimg.shape[1], 1 + torch.ceil(torch.max(yy2)).int().cpu().numpy())
            z2 = min(Gimg.shape[2], 1 + torch.ceil(torch.max(zz2)).int().cpu().numpy())
            xx2 -= int(x1)
            yy2 -= int(y1)
            zz2 -= int(z1)

            # Read in data
            G = torch.squeeze(torch.tensor(Gimg.get_fdata()[x1:x2, y1:y2, z1:z2], dtype=torch.int, device=device))
            S = torch.squeeze(torch.tensor(nib.load(segmentation_labels).get_fdata()[x1:x2, y1:y2, z1:z2], dtype=torch.int, device=device))
            T1 = torch.squeeze(torch.tensor(nib.load(t1).get_fdata()[x1:x2, y1:y2, z1:z2], dtype=dtype, device=device))
            T2 = None if t2 is None else torch.squeeze(torch.tensor(nib.load(t2).get_fdata()[x1:x2, y1:y2, z1:z2], dtype=dtype, device=device))
            FLAIR = None if flair is None else torch.squeeze(torch.tensor(nib.load(flair).get_fdata()[x1:x2, y1:y2, z1:z2], dtype=dtype, device=device))

            # Kill a bunch of labels
            M = lut_kill[S]>0
            S[~M] = 0
            G[~M] = 0
            T1[~M] = 0
            if T2 is not None:
                T2[~M] = 0
            if FLAIR is not None:
                FLAIR[~M] = 0

            # normalize images for later mixing
            T1 /= torch.median(T1[M])
            T2 = None if T2 is None else (T2/torch.median(T2[M]))
            FLAIR = None if FLAIR is None else (FLAIR / torch.median(FLAIR[M]))

            # Sample Gaussian image
            mus = 25 + 200 * torch.rand(256, dtype=dtype, device=device)
            sigmas = 5 + 20 * torch.rand(256, dtype=dtype, device=device)
            # set the background to zero
            mus[0] = 0
            sigmas[0] = 0

            #  Crucial bit: partial volume!
            # 1 = lesion, 2 = WM, 3 = GM, 4 = CSF
            v = 0.02 * torch.arange(50).to(device)
            mus[100:150] = mus[1] * (1 - v) + mus[2] * v
            mus[150:200] = mus[2] * (1 - v) + mus[3] * v
            mus[200:250] = mus[3] * (1 - v) + mus[4] * v
            mus[250] = mus[4]
            sigmas[100:150] = torch.sqrt(sigmas[1]**2 * (1 - v) + sigmas[2]**2 * v)
            sigmas[150:200] = torch.sqrt(sigmas[2]**2 * (1 - v) + sigmas[3]**2 * v)
            sigmas[200:250] = torch.sqrt(sigmas[3]**2 * (1 - v) + sigmas[4]**2 * v)
            sigmas[250] = sigmas[4]

            SYN = mus[G] + sigmas[G] * torch.randn(G.shape, dtype=dtype, device=device)
            SYN[SYN < 0] = 0
            SYN /= torch.median(SYN[M])

            # cosmetic blurring
            # note that we don't worry blurring foreground with black foreground because we do that at test time anyway
            sigma = sigma_blur_min + (sigma_blur_max - sigma_blur_min) * np.random.rand()
            SYNblur = gaussian_blur_3d(SYN, sigma * np.ones(3), device, dtype=dtype)

            # Make random linear combinations
            if np.random.rand() < real_mix_prob:
                v = torch.rand(4)
                v[2] = 0 if T2 is None else v[2]
                v[3] = 0 if FLAIR is None else v[3]
                v /= torch.sum(v)
                HR = v[0] * SYNblur + v[1] * T1
                if T2 is not None:
                    HR += v[2] * T2
                if FLAIR is not None:
                    HR += v[3] * FLAIR
            else:
                HR = SYNblur

            # deform everything at the same time!
            HRdef = fast_3D_interp_torch(HR, xx2, yy2, zz2, 'linear', device, dtype=dtype)

            # Gamma transform
            gamma = torch.tensor(np.exp(gamma_std * np.random.randn(1)[0]), dtype=float, device=device)
            HRgamma = 3.0 * (HRdef / 3.0) ** gamma

            # Bias field
            bf_scale = bf_scale_min + np.random.rand(1) * (bf_scale_max - bf_scale_min)
            siz_BF_small = np.round(bf_scale * np.array(gensize)).astype(int).tolist()
            siz_BF_small[1] = np.round(gensize[1]/spacing_simulation).astype(int)
            BFsmall = torch.tensor(bf_std_min + (bf_std_max - bf_std_min) * np.random.rand(1), dtype=dtype, device=device) * torch.randn(siz_BF_small, dtype=dtype, device=device)
            BFlog = myzoom_torch(BFsmall, np.array(gensize) / siz_BF_small, device)
            BF = torch.exp(BFlog)
            HRbf = HRgamma * BF

            # Generate random triplets!
            inputs = torch.zeros([batchsize, 2, siz[0], siz[1]], dtype=dtype, device=device)
            outputs = torch.zeros([batchsize, 1, siz[0], siz[1]], dtype=dtype, device=device)
            dists = torch.zeros([batchsize, 2], dtype=dtype, device=device)

            for b in range(batchsize):
                spac = spacing_limits[0] + mid_loc * (spacing_limits[1] - spacing_limits[0])
                y_min = 0.5 * spac
                y_max = HRbf.shape[1] - 1.0 - 0.5 * spac
                y = y_min + (y_max-y_min) * torch.rand(1, device=device)
                y1 = y - 0.5 * spac
                y2 = y + 0.5 * spac
                yi = y1 + (y2 - y1) * torch.rand(1, device=device)
                inputs[b, 0] = get_noninteger_coronal_slice(HRbf, y1).squeeze()
                inputs[b, 1] = get_noninteger_coronal_slice(HRbf, y2).squeeze()
                outputs[b, 0] = get_noninteger_coronal_slice(HRbf, yi).squeeze()
                dists[b, 0] = yi - y1
                dists[b, 1] = y2 - yi

            # Flip 50% of times
            if (flipping and (np.random.rand()<0.5)):
                inputs = inputs.flip(dims=[2])
                outputs = outputs.flip(dims=[2])

            if provide_2d_gradients:
                G_x = torch.nn.functional.conv2d(outputs, sobel_x, padding='same')
                G_y = torch.nn.functional.conv2d(outputs, sobel_y, padding='same')
                gradient_images = torch.sqrt(G_x * G_x + G_y * G_y + 1e-8)
            else:
                gradient_images = None

            yield [inputs, gradient_images, dists, outputs]


#######################
# Auxiliary functions #
#######################

def make_affine_matrix(rot, sh, s):
    Rx = np.array([[1, 0, 0], [0, np.cos(rot[0]), -np.sin(rot[0])], [0, np.sin(rot[0]), np.cos(rot[0])]])
    Ry = np.array([[np.cos(rot[1]), 0, np.sin(rot[1])], [0, 1, 0], [-np.sin(rot[1]), 0, np.cos(rot[1])]])
    Rz = np.array([[np.cos(rot[2]), -np.sin(rot[2]), 0], [np.sin(rot[2]), np.cos(rot[2]), 0], [0, 0, 1]])

    SHx = np.array([[1, 0, 0], [sh[1], 1, 0], [sh[2], 0, 1]])
    SHy = np.array([[1, sh[0], 0], [0, 1, 0], [0, sh[2], 1]])
    SHz = np.array([[1, 0, sh[0]], [0, 1, sh[1]], [0, 0, 1]])

    A = SHx @ SHy @ SHz @ Rx @ Ry @ Rz
    A[0, :] = A[0, :] * s[0]
    A[1, :] = A[1, :] * s[1]
    A[2, :] = A[2, :] * s[2]

    return A