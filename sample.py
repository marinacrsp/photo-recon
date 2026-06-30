"""
Script used to sample brain volumes from a set of dissection photographs with a pretrained model
Script usage: 
python sample.py    --model_path <path_to_model> 
                    --input_file <path_to_input_mgz> 
                    --illumination <illumination_value> 
                    --unsharp_sigma <sigma_value> 
                    --unsharp_amount <amount_value>

"""
import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
# If everything went fine, we import the rest of packages
import torch
from ext import (MRIread, 
                 MRIwrite, 
                 eugenios_closest_canonical, 
                 gaussian_blur_3d, 
                 add_dict_to_argparser
                 )

import numpy as np
import cv2
from ext import UNet2D, logger
import torch.nn.functional as F
from pathlib import Path

def pad_to_symmetric(tensor):
    """
    Pads a [B, C, H, W] tensor so that H and W become 160 using symmetric (centered) padding.
    """
    h, w = tensor.shape[-2], tensor.shape[-1]
    max_size =  max(h,w)
    pad_h = max(max_size - h, 0) 
    pad_w = max(max_size - w, 0)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padding = (pad_left, pad_right, pad_top, pad_bottom)
    return F.pad(tensor, padding, mode='constant', value=0) # pad with zeros


def center_crop(tensor, original_hw):
    """
    Center-crops the last two dimensions of the tensor back to original_hw = (H, W).
    Works for tensors with shape [B, C, H, W].
    """
    orig_h, orig_w = original_hw
    h, w = tensor.shape[-2], tensor.shape[-1]

    start_h = (h - orig_h) // 2
    start_w = (w - orig_w) // 2

    return tensor[..., start_h:start_h+orig_h, start_w:start_w+orig_w]

def create_argparser():
    defaults = dict(
        model_path="/home/marina/ms_thesis/imputation_unet_2026_code/model_weights.pth",
        seed=42,
        num_workers=4,
        save_path="/home/marina/ms_thesis/imputation_unet_2026_code/file.mgz",
        input_file="/mnt/c/Users/marin/Desktop/master_thesis/reviewer_comments/photo_recon_uw/photo_recon_uw/00_photo_recon/18-0086/photo_recon_4mm.nii.gz",
        illumination=None,
        unsharp_sigma=1.0,
        unsharp_amount=1.0,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

# ================================================================================================
#                                         Main Entrypoint
# ================================================================================================

def main():
    args = create_argparser().parse_args()
    device = "cuda" if torch.cuda.is_available() else torch.device('cpu')
    args.use_fp16 = False
    dtype=torch.float32
    dist_scale = 0.1
    print(args)
    torch.set_num_threads(min(int(args.num_workers), os.cpu_count()))
    workdir = args.model_path[args.model_path.find("workdir"):-3]

    sample_dir = Path(workdir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    logger.log("creating model ...")
    model = UNet2D(4, 1, basic_module="conv", final_sigmoid=False, f_maps=128, layer_order='gcl', num_groups=8, num_levels=5, is_segmentation=False)

    ckpt = torch.load(args.model_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model = model.to(device)

    if args.use_fp16:
        model = model.half()
    model.eval()

    logger.log("sampling...")

    with torch.no_grad():

       # Read in input file with reorientation
        print('Reading and resizing input images')
        I_orig, aff_orig = MRIread(args.input_file)
        I_orig, aff_orig, ap_flip = eugenios_closest_canonical(I_orig, aff_orig, return_ap_flip=True)
        if len(I_orig.shape)==3:
            I_orig = I_orig[..., None]

        voxsize = np.sqrt(np.sum(aff_orig[:-1,:-1]**2, axis=0))
        av_thickness = voxsize[1]
        inplane_res = .5 * voxsize[0] + .5 * voxsize[2]

        ## Adapt in-plane resolution to 1mm/px
        I = []
        for j in range(I_orig.shape[1]):
            I.append(cv2.resize(I_orig[:,j,:,:], (0,0), fx=inplane_res, fy=inplane_res, interpolation=cv2.INTER_AREA))
        I = np.stack(I, axis=1)
        aff = aff_orig.copy()
        aff[:-1, 0] /= inplane_res
        aff[:-1, 2] /= inplane_res
        aff[:-1,-1] -= aff[:-1,:-1] @ np.array([0.5*(inplane_res-1), 0, 0.5*(inplane_res-1),])

        I = torch.tensor(I, dtype=dtype, device=device)
        # convert grayscale to RGB if needed
        if len(I.shape)==3:
            I = I[..., None]

        # Compute mask and normalize by median
        M = I.sum(dim=[3])>5

        medians = torch.zeros(I.shape[3], dtype=torch.float32)
        minmax = torch.zeros(I.shape[3], 2, dtype=torch.float32)
        for c in range(I.shape[3]):
            auxI = I[:,:,:,c]
            medians[c] = torch.median(auxI[M])
            I[:, :, :, c] /= medians[c]

        # Calculate areas and detect padding
        areas = M.sum(dim=[0, 2]).detach().cpu().numpy()
        aux = np.where(areas > 0)
        PAD = aux[0][0].astype(np.int32)
        # if (aux[0][-1] != (I.shape[1] - 4)):
        #     raise Exception('Uneven padding detected in anterior and posterior directions')

        thicknesses = av_thickness * np.ones(M.shape[1] - 2 * PAD + 1)

        # Create output image with appropriate header
        I2 = torch.zeros([I.shape[0], np.ceil(I.shape[1] * av_thickness).astype(np.int32) , I.shape[2], I.shape[3]], device=device, dtype=dtype)
        I2linear = torch.zeros_like(I2)
        aff2 = aff.copy()
        aff2[:-1,1] = aff2[:-1,1] / av_thickness # this is to reshape to 1mm/px
        aff2[:-1,-1] = aff2[:-1,-1] - aff2[:-1,:-1] @ np.array([0, 0.5*(av_thickness-1), 0])

        # Prepare input with size multiple of 32
        shape2d = np.array([I.shape[0], I.shape[2]]) # this is just the original shape of the volume
        W = (np.ceil(shape2d / 32.0) * 32).astype('int') # W is the new shape to be interpolated
        idx = np.floor((W - shape2d) / 2).astype('int')
        S = torch.zeros([I.shape[3], 4, *W], dtype=torch.float32, device=device)

        # get adjusted coordinates of photos; we assume that the thickness is av_thickness for the empty slices
        y_coords = av_thickness * np.arange(PAD)
        y_coords = np.concatenate([y_coords, y_coords[-1]+thicknesses.cumsum()])
        y_coords = np.concatenate([y_coords, y_coords[-1]+av_thickness * np.arange(1, PAD)])

        # Loop over coronal slices and interpolate
        for j in range(I2.shape[1]):
            print('Working on slice ' + str(j+1) + ' of ' + str(I2.shape[1]), end='\r')
            js = j - (av_thickness - 1 ) / 2
            aux = np.where(y_coords<=js)[0]
            idx1 = aux.max() if len(aux)>0 else 0
            idx2 = min(idx1+1, len(y_coords)-1)
            d1 = js - y_coords[idx1]
            d2 = y_coords[idx2] - js
            w1 = (d2 / (d1 + d2)) if ((d1+d2)>0) else 0.0
            w2 = 1 - w1
            linear_interp = w1 * I[:, idx1, :, :] + w2 * I[:, idx2, :, :]
            I2linear[:, j, :, :] = linear_interp
            
            for c in range(I.shape[3]):
                S[c, 0, idx[0]:idx[0] + I.shape[0], idx[1]:idx[1] + I.shape[2]] = I[:, idx1, :, c]
                S[c, 1, idx[0]:idx[0] + I.shape[0], idx[1]:idx[1] + I.shape[2]] = I[:, idx2, :, c]
                S[c, 2, :, :] = dist_scale * d1
                S[c, 3, :, :] = dist_scale * d2
            pred = model(S)

            I2[:, j, :, :] = pred[:,0,:,:].permute([1,2,0])[idx[0]:idx[0] + I.shape[0], idx[1]:idx[1] + I.shape[2], :]

    I2 = I2 + I2linear
    unsharp_amount, unsharp_sigma = 1, 1
    
    # DENORMALIZE
    for c in range(I2.shape[-1]):
        # I2[...,c] = (I2[...,c] + 1)*0.5*(minmax[c,1] - minmax[c,0]) + minmax[c,0] # from -1, 1 -> 0, 1
        I2[...,c] *= medians[c] # from 0, 1 -> 0, 255

    # # SHARPEN
    unsharp_amount, unsharp_sigma = 1, 1
    blurred = torch.zeros_like(I2) 
    for c in range(I2.shape[3]):
        blurred[..., c] = gaussian_blur_3d(I2[..., c], [unsharp_sigma, unsharp_sigma, unsharp_sigma], device)      
    I2 += unsharp_amount * (I2 - blurred) # cause in-plane slices look sharper than off-plane slices

    I2 = (I2.clip(0, 255).squeeze()).detach().cpu().numpy().astype(np.uint8)
    output_file_nn = f'{args.save_path}'
    MRIwrite(I2, aff2, output_file_nn)
    logger.log(f"Written file as: {output_file_nn}")
if __name__ == "__main__":
    main()


