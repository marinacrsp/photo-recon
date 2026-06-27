import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from generators import hemi_generator
import glob
import torch
from torch.optim import Adam
from torch.nn import L1Loss
from ext import UNet2D, mean_flat

def validate(model, val_data, device, dist_scale):
    """Evaluate on a fixed, pre-materialized validation set (identical on every call)."""
    model.to(torch.float16) # for memory purposes
    model.eval()
    
    val_loss_step = 0.0
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        for (inputs, _, dists_th, outputs) in val_data:
            input = torch.concatenate([inputs, dist_scale * dists_th[..., None, None].repeat([1, 1, *siz])], dim=1)
            w1 = (dists_th[:,1] / (dists_th.sum(dim=1)))[:, None, None, None]
            w2 = (1 - w1)
            linear_interp = w1 * inputs[:, 0:1, :, :] + w2 * inputs[:, 1:2, :, :]
            pred = model(input.to(device, dtype=torch.float32))
            model_prediction = linear_interp + pred.detach()
            val_loss = L1Loss()(model_prediction, outputs)
            val_loss_step += val_loss.item()
        
    val_loss_step /= len(val_data)

    model.train()       
    return val_loss_step

################
#   Options    #
################

training_data_dir_hemis = '/autofs/vast/lemon/data_curated_hemis/brain_mris_QCed/' # TODO: switch as needed

use_gradient_loss = True
use_NAFnet = False
output_directory ='.' #TODO: specify output directory 
dtype = torch.float32
device_generator = 'cuda:0'
device_training = 'cuda:0'
dist_scale = 0.1

spacing_limits=[2,12]
nonlin_std_max = 0.0 # default is 4.0
labels_to_kill = [3, 4]
siz = [160, 160]
batchsize = 32
flipping_generator = True

n_epochs = 250
n_its_per_epoch = 1000
log_every = 500

# Unet options
f_maps = 128
layer_order='gcl'
num_groups = 8
num_levels = 5


num_filters = 64  # these were the parameters for the lighter 3D version for Shahin
enc_blks = [1, 1, 1, 28]
middle_blk_num = 1
dec_blks = [1, 1, 1, 1]

patience = 10
best_val_loss = 100
n_val_samples = 100

#########################
# Training and validation
#########################

# Create output directory if needed
if os.path.exists(output_directory) is False:
    os.mkdir(output_directory)

# generator of the training samples
gen = hemi_generator(training_data_dir_hemis,
                     spacing_limits=spacing_limits,
                     labels_to_kill=labels_to_kill,
                     siz=siz,
                     batchsize=batchsize,
                     flipping=flipping_generator,
                     nonlin_std_max=nonlin_std_max,
                     provide_2d_gradients=use_gradient_loss,
                     device=device_generator,
                     dtype=dtype)

# generator of the validation samples (no flipping, no gradients, predict the middle slice only)
gen_val = hemi_generator(training_data_dir_hemis,
                     spacing_limits=spacing_limits,
                     labels_to_kill=labels_to_kill,
                     siz=siz,
                     batchsize=1,
                     flipping=False,
                     nonlin_std_max=nonlin_std_max,
                     provide_2d_gradients=False,   # gradients are unused in validation
                     device=device_generator,
                     mid_loc=True,
                     dtype=dtype)

# Initialize the validation set (fixed for all epochs)
val_set = [next(gen_val) for _ in range(n_val_samples)]

in_channels = 4
out_channels = 1

# Initialize model and optimizer
model = UNet2D(in_channels, out_channels, final_sigmoid=False, f_maps=f_maps, layer_order=layer_order,
            num_groups=num_groups, num_levels=num_levels, is_segmentation=False).to(device_training)
optimizer = Adam(model.parameters(), lr=1e-4)

# Load weights if available
g = sorted(glob.glob(output_directory + '/*.pth'))
if len(g)==0:
    print('Starting from scratch')
    epoch_ini = 0
else:
    print('Loading weights from ' + g[-1])
    checkpoint = torch.load(g[-1])
    epoch_ini = 1 + checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Sobel operators (note the proper normalization!)
sobel_x = 0.125 * torch.tensor([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]], dtype=dtype, device=device_training).view((1, 1, 3, 3))
sobel_y = 0.125 * torch.tensor([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]], dtype=dtype, device=device_training).view((1, 1, 3, 3))

# Train!
for j in range(n_epochs - epoch_ini):

    epoch = epoch_ini + j
    cumul_loss_epoch = 0.0

    print('Epoch ' + str(epoch+1) + ' of ' + str(n_epochs))
    loss_epoch_acc = 0.0

    for iteration in range(n_its_per_epoch):

        # Generate!
        [inputs, gradient_images, dists, outputs] = next(gen) # here we generate new samples
        w1 = (dists[:,1] / (dists.sum(dim=1)))[:, None, None, None]
        w2 = (1 - w1)
        linear_interp = w1 * inputs[:, 0:1, :, :] + w2 * inputs[:, 1:2, :, :]
        targets = outputs - linear_interp

        input = torch.concatenate([inputs, dist_scale * dists[..., None, None].repeat([1, 1, *siz])], dim=1)

        with torch.enable_grad():
            pred = model(input)
            loss_images = mean_flat((pred - targets).abs())
            loss_image = loss_images.mean()
            if use_gradient_loss:
                ims = pred + linear_interp

                G_x = torch.nn.functional.conv2d(ims, sobel_x, padding='same')
                G_y = torch.nn.functional.conv2d(ims, sobel_y, padding='same')
                gradient_pred = torch.sqrt(G_x * G_x + G_y * G_y + 1e-8)
                loss_gradients = L1Loss()(gradient_pred, gradient_images)
                loss = loss_image + loss_gradients
            else:
                loss = loss_image
            if torch.isnan(loss):
                loss = torch.tensor(cumul_loss_epoch).to(device_training) # just to keep stats nice
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        loss_epoch_acc = loss_epoch_acc + loss.detach().cpu().numpy()
        cumul_loss_epoch = loss_epoch_acc / (iteration + 1)
        print('   Iteration ' + str(1+iteration) + ' of ' + str(n_its_per_epoch) + ', loss = ' + str(cumul_loss_epoch), end="\r")


    val_loss = validate(model, gen_val, device_training, dist_scale, val_set)

    print('\n   End of epoch ' + str(epoch+1) + '; saving model... \n')

    val_loss = validate(model, val_set, device_training, dist_scale)

    # Update early-stopping bookkeeping BEFORE checkpointing, so the saved state is current
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        improved = True
    else:
        patience_counter += 1
        improved = False
 
    print('\n   End of epoch ' + str(epoch+1) + '; saving model... \n')
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': cumul_loss_epoch,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
    }
    torch.save(ckpt, os.path.join(output_directory, 'checkpoint.pth'))
    
    if improved:
        torch.save(ckpt, os.path.join(output_directory, 'best_model.pth'))
 
    if patience_counter >= patience:
        print(f'Early stopping triggered after {patience} epochs without improvement.')
        break

print('Training complete!')








