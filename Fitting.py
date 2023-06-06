import torch
import torch.nn as nn
import argparse
from tqdm.auto import tqdm
import utils

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

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

    reg_list = [] if args.reg_type in ["smoothing"] else None

    mse_fn = nn.MSELoss(reduction=args.reduction)
    
    #make loss fns and optimizer
    optim = torch.optim.Adam(args.model.parameters(), lr=args.lr)
    
    criterion = utils.Measurement_MSE_Loss(kept_inds=args.kept_inds, per_param=False, reduction=args.reduction)
    criterion = criterion.to(args.device)

    if (args.reg_type == "smoothing") and (args.reg_lambda > 0):
        regularizer = utils.Smoothing_Loss(per_param=False, reduction=args.reduction, order=args.reg_order, norm=args.reg_norm)
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
