import torch
import torch.nn as nn

import numpy as np

import argparse

import matplotlib.pyplot as plt

from tqdm.auto import tqdm

import utils
from models import RESNET_BACKBONE, RESNET_HEAD, MODULAR_RESNET


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

class Centered_Weight_Decay(nn.Module):
    def __init__(self, center_net, reduction="mean"):
        super().__init__()

        self.mu = nn.utils.parameters_to_vector(center_net.parameters()).detach().clone()
        self.reduction = reduction

        self.mse_loss = nn.MSELoss(reduction=self.reduction)
    
    def forward(self, net):
        params_new = nn.utils.parameters_to_vector(net.parameters())

        return self.mse_loss(params_new, self.mu)

@torch.no_grad()
def add_noise_to_weights(model, noise_std):
    for param in model.parameters():
        additive_noise = torch.randn_like(param.data) * noise_std
        param.data.add_(additive_noise)

def run_dip(args):
    #track the outputs
    out_list = []

    mse_list_gt = []
    mse_list_meas = []
    mse_list_mean_gt = []
    mse_list_mean_meas = []

    reg_list = [] if args.reg_type in ["smoothing", "centered_wd"] else None

    mse_fn = nn.MSELoss(reduction=args.reduction)
    
    #make loss fns and optimizer
    optim = torch.optim.Adam(args.model.parameters(), lr=args.lr)
    
    criterion = utils.Measurement_MSE_Loss(kept_inds=args.kept_inds, per_param=False, reduction=args.reduction)
    criterion = criterion.to(args.device)

    if (args.reg_type == "smoothing") and (args.reg_lambda > 0):
        regularizer = utils.Smoothing_Loss(per_param=False, reduction=args.reduction, order=args.reg_order, norm=args.reg_norm)
        regularizer = regularizer.to(args.device)
    elif (args.reg_type == "centered_wd") and (args.reg_lambda > 0):
        center_net = args.model.backbone if args.reg_target == "backbone" else args.model
        regularizer = Centered_Weight_Decay(center_net=center_net, reduction=args.reduction)
        regularizer = regularizer.to(args.device)
    elif (args.reg_type == "wd") and (args.reg_lambda > 0):
        optim = torch.optim.Adam(args.model.parameters(), lr=args.lr, weight_decay=args.reg_lambda)

    #set up some running stuff for noise
    input_noise_start = getattr(args, 'input_noise_start', None)
    input_noise_decay = getattr(args, 'input_noise_decay', 1.0)

    perturb_weights = getattr(args, 'perturb_weights', False)
    burn_in_iter = getattr(args, 'burn_in_iter', 0)
    save_output_every = getattr(args, 'save_output_every', 50)
    
    #main loop
    if args.debug:
        iter_list = tqdm(range(args.num_iter))
    else:
        iter_list = range(args.num_iter) 

    for i in iter_list:
        optim.zero_grad()

        #get the output with or without additive noise in the input
        if (input_noise_start is not None):
            noisy_z = args.z + torch.randn_like(args.z) * input_noise_start
            out = args.model.forward(noisy_z)
            input_noise_start *= input_noise_decay
        else:
            out = args.model.forward(args.z)
        
        #loss and regularization
        error = criterion(out, args.y) 
        
        if (args.reg_type == "smoothing") and (args.reg_lambda > 0):
            reg = args.reg_lambda * regularizer(out)
            loss = error + reg
        elif (args.reg_type == "centered_wd") and (args.reg_lambda > 0):
            reg_target = args.model.backbone if args.reg_target == "backbone" else args.model
            reg = args.reg_lambda * regularizer(reg_target)
            loss = error + reg
        else:
            loss = error
        
        loss.backward()
        optim.step()

        #optional noise perturbation
        if perturb_weights:
            add_noise_to_weights(args.model, 2 * args.lr)

        #logging and metrics
        with torch.no_grad():
            mse_gt = mse_fn(out, args.x).item()
            mse_list_gt.append(mse_gt)

            mse_list_meas.append(error.item())

            if reg_list is not None:
                reg_list.append(reg.item())
            
            if (i >= burn_in_iter) and (i % save_output_every == 0):
                out_list.append(out.detach().clone())

                mean_out = torch.mean(torch.cat(out_list), dim=0, keepdim=True)
                
                mse_mean_gt = mse_fn(mean_out, args.x)
                mse_list_mean_gt.append(mse_mean_gt.item())

                mse_mean_meas = criterion(mean_out, args.y)
                mse_list_mean_meas.append(mse_mean_meas.item())
    
    out_dict = {"out_list": out_list,
                "mse_list_gt": mse_list_gt,
                "mse_list_meas": mse_list_meas,
                "mse_list_mean_gt": mse_list_mean_gt,
                "mse_list_mean_meas": mse_list_mean_meas}
    if reg_list is not None:
        out_dict["reg_list"] = reg_list

    return dict2namespace(out_dict)

def grab_sparams(root_pth, chip_num):
    """
    Returns a torch tensor of s-parameters with shape [1, 2*num_unique_sparams, num_freqs]
        given a path and a chip number.
    """
    #first grab the chip
    chip_dict = utils.get_network_from_file(root_pth, chip_num)
    
    out_network = chip_dict["network"]
    out_freqs = out_network.frequency
    
    #resample to minimum length if necessary
    MIN_LEN = 1000
    
    if out_freqs.npoints < MIN_LEN:
        scale_fac = int(np.ceil(MIN_LEN / out_freqs.npoints))
        new_len = scale_fac * (out_freqs.npoints - 1) + 1 #this is smarter scaling that just divides current spacing
        
        out_network.resample(new_len)
        out_freqs = out_network.frequency
    
    #convert to unique s-parameters tensor
    out_matrix_re = out_network.s.real
    out_matrix_im = out_network.s.imag
    out_matrix = np.stack((out_matrix_re, out_matrix_im), axis=-1)

    out_sparams = utils.matrix_to_sparams(out_matrix)

    out_sparams = out_sparams.reshape(1, -1, out_freqs.npoints)

    return torch.tensor(out_sparams)

def fit_DIP(model, y, z, 
            lr, num_iter, 
            train_loss, train_reg, reg_lambda=0, 
            start_noise=None, noise_decay=None):
    """
    Runs DIP for a single set of given measurements.

    Returns the fitted network and the final output.
    """
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    
    for i in range(num_iter):
        optim.zero_grad()
        
        #get the output with or without additive noise in the input
        if (start_noise is not None) and (noise_decay is not None):
            noisy_z = z + torch.randn_like(z) * start_noise
            out = model.forward(noisy_z)
            start_noise *= noise_decay
        else:
            out = model.forward(z)
        
        #loss and regularization
        error = train_loss(out, y) 
        if reg_lambda > 0:
            reg = reg_lambda * train_reg(out)
            loss = error + reg
        else:
            loss = error

        loss.backward()
        optim.step()
    
    return model, out

@torch.no_grad()
def grab_data_and_net(data_root, chip_num, measurement_spacing, num_measurements,
                      ngf, kernel_size, causal, passive, backbone):
    """
    Grabs the ground truth s-parameters for a chip along with measurements, adjoint
        solution as the network input, the indices of the measurements, and a network 
        head for the dimension of the data. 
    """
    
    x = grab_sparams(data_root, chip_num)

    #grab the appropriate measurements
    kept_inds, missing_inds = utils.get_inds(measurement_spacing, x.shape[-1], num_measurements)

    y = torch.clone(x)[:, :, kept_inds]

    z = torch.clone(x)
    z[:, :, missing_inds] = 0

    #set up the clone network and head and make modular net
    net_head = RESNET_HEAD(nz=x.shape[1],
                           ngf_in_out=ngf,
                           nc=x.shape[1],
                           output_size=x.shape[-1],
                           kernel_size=kernel_size,
                           causal=causal,
                           passive=passive)

    backbone_clone = backbone.make_clone() 

    net = MODULAR_RESNET(backbone=backbone_clone,
                         head=net_head)
    
    return x, y, z, kept_inds, net

def train_step(data_root, chip_num, measurement_spacing, num_measurements, ngf,
               kernel_size, causal, passive, backbone, device, lr_inner, 
               num_iters_inner, reg_lambda_inner, start_noise_inner, noise_decay_inner,
               plot_output=False):
    """
    Performs DIP on a single sample: grabs the measurements, fits the network, calculates the loss.

    Returns the fitted network and the test mse for the sample. 
    """
    #sample chip
    x, y, z, kept_inds, net = grab_data_and_net(data_root=data_root, chip_num=chip_num, 
                                measurement_spacing=measurement_spacing, 
                                num_measurements=num_measurements, ngf=ngf, kernel_size=kernel_size, 
                                causal=causal, passive=passive, backbone=backbone)
    x = x.to(device)
    y = y.to(device)
    z = z.to(device)
    net = net.to(device)

    #set up losses and regularisations
    criterion = utils.Measurement_MSE_Loss(kept_inds=kept_inds, per_param=True, reduction="mean")
    criterion = criterion.to(device)

    regularizer = utils.Smoothing_Loss(per_param=True, reduction="mean")
    regularizer = regularizer.to(device)

    #Run DIP and get the metrics 
    updated_net, x_hat = fit_DIP(model=net, y=y, z=z, 
                                 lr=lr_inner, num_iter=num_iters_inner, 
                                 train_loss=criterion, train_reg=regularizer, reg_lambda=reg_lambda_inner, 
                                 start_noise=start_noise_inner, noise_decay=noise_decay_inner) 
    with torch.no_grad():
        test_mse = nn.MSELoss()(x_hat, x).item()
    
        if plot_output:
            x_mag = utils.sparams_to_mag(x)
            out_mag = utils.sparams_to_mag(x_hat)
            dip_errors_mag = x_mag - out_mag 

            _, axes = plt.subplots(3,1, figsize=(8, 6))
            axes = axes.flatten()

            for i in range(x_mag.shape[1]):
                axes[0].plot(x_mag[0,i].cpu(), label=str(i))
            axes[0].set_title("Ground Truth Magnitude Spectrum")
            axes[0].set_ylim(0,1)

            for i in range(x_mag.shape[1]):
                axes[1].plot(out_mag[0,i].detach().cpu(), label=str(i))
            axes[1].set_title("DIP Output Magnitude Spectrum")
            axes[1].set_ylim(0,1)
                
            for i in range(x_mag.shape[1]):
                axes[2].plot(dip_errors_mag[0,i].detach().cpu(), label=str(i))
            axes[2].set_title("DIP Errors Magnitude Spectrum")
            axes[2].set_ylim(-1,1)
            
            plt.show()
    
    return updated_net, test_mse

def reptile(backbone, data_root, device, test_inds, train_inds, num_epochs, lr_meta, 
            ngf, kernel_size, causal, passive,
            measurement_spacing_inner, num_measurements_inner, lr_inner,
            num_iters_inner, reg_lambda_inner, start_noise_inner, noise_decay_inner,
            measurement_spacing_outer, num_measurements_outer, lr_outer,
            num_iters_outer, reg_lambda_outer, start_noise_outer, noise_decay_outer):
    """
    Performs REPTILE-style updates for a given backbone network over a training dataset. 

    Returns the updated network, test losses for the inner optimization, and test losses for the meta opt. 
    """
    
    optim = torch.optim.Adam(backbone.parameters(), lr=lr_meta)
    
    inner_test_losses = []
    outer_test_losses = []
    meta_losses = []

    epoch_pbar = tqdm(range(num_epochs))
    for epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch {epoch}")

        inner_test_losses_epoch = []
        outer_test_losses_epoch = []
        meta_losses_epoch = []

        #testing - don't update parameters, just track the metrics
        testing_pbar = tqdm(test_inds)
        for test_chip_ind in testing_pbar:
            testing_pbar.set_description(f"Testing - Sample {test_chip_ind}")

            _, outer_test_mse = train_step(data_root=data_root, chip_num=test_chip_ind, 
                                           measurement_spacing=measurement_spacing_outer, num_measurements=num_measurements_outer, 
                                           ngf=ngf, kernel_size=kernel_size, causal=causal, passive=passive, 
                                           backbone=backbone, device=device, lr_inner=lr_outer,
                                           num_iters_inner=num_iters_outer, reg_lambda_inner=reg_lambda_outer, 
                                           start_noise_inner=start_noise_outer, noise_decay_inner=noise_decay_outer)
            #update progress
            outer_test_losses.append(outer_test_mse)
            outer_test_losses_epoch.append(outer_test_mse)

            testing_pbar.set_postfix({'sample mse': outer_test_mse,
                                      'epoch mse': np.mean(outer_test_losses_epoch)})
            epoch_pbar.set_postfix({'mean outer mse': np.mean(outer_test_losses), 
                                    'mean inner mse': 'N/A',
                                    'mean meta loss': 'N/A'})
        
        #training - update params and track metrics
        training_pbar = tqdm(np.random.permutation(train_inds))
        for train_chip_ind in training_pbar:
            training_pbar.set_description(f"Training - Sample {train_chip_ind}")

            updated_net, inner_test_mse = train_step(data_root=data_root, chip_num=train_chip_ind, 
                                           measurement_spacing=measurement_spacing_inner, num_measurements=num_measurements_inner, 
                                           ngf=ngf, kernel_size=kernel_size, causal=causal, passive=passive, 
                                           backbone=backbone, device=device, lr_inner=lr_inner,
                                           num_iters_inner=num_iters_inner, reg_lambda_inner=reg_lambda_inner, 
                                           start_noise_inner=start_noise_inner, noise_decay_inner=noise_decay_inner)
            #update params
            new_backbone = updated_net.backbone.cpu()
            new_backbone.requires_grad_(False)

            params_cur = nn.utils.parameters_to_vector(backbone.parameters())
            params_new = nn.utils.parameters_to_vector(new_backbone.parameters())

            meta_loss = 0.5 * torch.sum((params_cur - params_new)**2)
            meta_loss.backward()

            optim.step()
            optim.zero_grad()

            #update progress bar
            inner_test_losses.append(inner_test_mse)
            inner_test_losses_epoch.append(inner_test_mse)

            meta_losses.append(meta_loss.item())
            meta_losses_epoch.append(meta_loss.item())

            training_pbar.set_postfix({'sample mse': inner_test_mse,
                                       'epoch mse': np.mean(inner_test_losses_epoch),
                                       'sample metaloss': meta_loss.item(),
                                       'epoch metaloss': np.mean(meta_losses_epoch)})
            epoch_pbar.set_postfix({'mean outer mse': np.mean(outer_test_losses), 
                                    'mean inner mse': np.mean(inner_test_losses),
                                    'mean meta loss': np.mean(meta_losses)})

    return backbone, inner_test_losses, outer_test_losses, meta_losses
