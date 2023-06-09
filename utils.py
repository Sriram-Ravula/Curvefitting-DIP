import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt

def get_inds(problem_type, length, num_kept_samples):
    """
    Given a number of samples to keep and a problem type, returns indices to keep from a list.

    Args:
        problem_type: What type of inpainting problem we are dealing with.
                      String in {"random", "equal", "forecast", "full", "log", "sqrt"}.
        length: The original length of the signal.
                Int.
        num_kept_samples: The number (or proportion) of samples to keep.
                          Int <= length OR float <= 1.0.
    
    Returns:
        kept_inds: Array with (sorted, increasing) indices of samples to keep from the original signal.
                   Numpy array of Ints.
        missing_inds: Array with  (sorted, increasing) indices of discarded sampled from the original signal.
                      Numpy array of Ints.
    """
    if isinstance(num_kept_samples, float) and num_kept_samples <= 1.0:
        num_kept_samples = int(length * num_kept_samples)
        num_kept_samples = max(num_kept_samples, 1)

    if problem_type=="random":
        kept_inds = np.random.choice(length, num_kept_samples, replace=False)
    elif problem_type=="equal":
        kept_inds = np.linspace(0, length-1, num=num_kept_samples, dtype=int)
    elif problem_type=="forecast":
        kept_inds = np.arange(0, num_kept_samples)
    elif problem_type=="full":
        kept_inds = np.arange(0, length)
    elif problem_type=="log": 
        kept_inds = np.geomspace(1, length, num=num_kept_samples, dtype=int) #geomspace can't take 0 as start index
        kept_inds = kept_inds - 1 
        kept_inds = np.sort(kept_inds) #making the list unique
        for i in range(1, len(kept_inds)):
            if kept_inds[i] <= kept_inds[i-1]:
                kept_inds[i] = kept_inds[i-1] + 1 
    elif problem_type=="sqrt": 
        r = (length - 1.) / ((num_kept_samples - 1.)**2) #base quadratic factor
        kept_inds = [round(r * (s**2)) for s in range(num_kept_samples - 1)]
        kept_inds.append(length-1)
        kept_inds = np.array(kept_inds)
        kept_inds = np.sort(kept_inds) #making the list unique
        for i in range(1, len(kept_inds)):
            if kept_inds[i] <= kept_inds[i-1]:
                kept_inds[i] = kept_inds[i-1] + 1 
    else:
        raise NotImplementedError("THIS PROBLEM TYPE IS UNSUPPORTED")
    
    missing_inds = np.array([t for t in range(length) if t not in kept_inds])
    
    return np.sort(kept_inds), np.sort(missing_inds)

def get_network_from_file(root_pth, chip_num):
    """
    Grabs an SK-RF Network and its associated metadata from a given root path
        and chip number.
    Different from grab_chip_data since this function returns the raw Network object
        and not np arrays. 

    Args:
        root_pth: Root of the folder containing chip info.
        chip_num: Chip number.

    Returns:
        out_dict: Dictionary with relevant entries.
                  Entries:
                    network: The ground truth network.
                             Scikit-RF Network object. 
                    fmin: Minimum frequency sampled.
                          Float.
                    fmax: Max frequency sampled.
                          Float. 
                    length: The length of the channels.
                            String.
                    sweep: The spacing between sampled points.
                           String.
                    port_pairs: The paied ports on the chip.
                                Array of tuples.
    """
    from skrf import Network

    #Grab the correct folder
    chip_num = str(chip_num) if chip_num > 9 else "0" + str(chip_num)
    fname = os.path.join(root_pth, "case"+chip_num)

    net_str = str(chip_num) + ".s"
    desc_str = "description.txt"

    #grab the correct file we want
    children = os.listdir(fname)

    net_fpth = [f for f in children if net_str in f][0]
    net_fpth = os.path.join(fname, net_fpth)

    desc_fpth = [f for f in children if desc_str in f][0]
    desc_fpth = os.path.join(fname, desc_fpth)

    #grab the network and metadata
    out_network = Network(net_fpth)

    with open(desc_fpth) as file:
        lines = [line.rstrip() for line in file]

    fmin = None
    fmax = None
    length = None
    sweep = None
    port_pairs = []

    for line in lines:
        if "fmin" in line:
            fmin = float(line.split(" ")[-1])
        elif "fmax" in line:
            fmax = float(line.split(" ")[-1])
        elif "length" in line:
            length = line.split(" ")[-1]
        elif "sweep" in line:
            sweep = line.split(" ")[-1]
        elif "port_pair" in line:
            port1 = int(line.split(" ")[-2])
            port2 = int(line.split(" ")[-1])
            port_pairs.append((port1, port2))
    
    out_dict = {"network": out_network,
                "fmin": fmin,
                "fmax": fmax,
                "length": length,
                "sweep": sweep,
                "port_pairs": port_pairs}
    
    return out_dict

def network_to_sparams(network):
    """
    Converts a given Sk-RF network object into a [1, 2*N_-sparams, N_freqs] torch tensor.
    
    Args:
        network: sk-rf network object.
    
    Returns:
        [1, 2*N_sparams, N_freqs] torch tensor. Channel dimenson has reals
            on the even indices and their respective imag's on the odd indices. 
    """
    re_mat = network.s.real
    im_mat = network.s.imag
    out_mat = np.stack((re_mat, im_mat), axis=-1)

    out_mat = out_mat.reshape((out_mat.shape[0], -1, out_mat.shape[-1])) #[FREQ, N_PORT^2, RE/IM]
    out_mat = out_mat.reshape((out_mat.shape[0], -1)).transpose() #[Re/im Sparams, FREQ]
    
    return torch.from_numpy(out_mat).unsqueeze(0).type(torch.float32)

def matrix_to_network(data_matrix, data_freqs, name, resample_freqs=None):
    """
    Takes a raw 4D sparam matrix and returns a Scikit-RF Network object.

    Args:
        data_matrix: Raw 4D sparam matrix. 
                     (4D torch tensor) - Axes are [Freq, In_port, Out_port, Re/Im].
        data_freqs: Sampled frequencies. Must be same length as first axis of data_matrix.
                    (1D numpy array).   
        name: The name of the network.
              (String)
        resample_freqs: Frequencies to resample the data to. Default is None.
                        (1D numpy array). 
    
    Returns:
        net: A scikit-RF Network object.
    """
    from skrf import Network, Frequency
    
    #Convert the data properly to complex
    temp_data = data_matrix.detach().cpu().numpy().astype('float64')
    
    net_data = np.empty(temp_data.shape[:-1], dtype=np.complex128)
    net_data.real = temp_data[..., 0]
    net_data.imag = temp_data[..., 1]

    #Make the network
    net_freqs = Frequency.from_f(data_freqs, unit="hz")

    net = Network(frequency=net_freqs, s=net_data, name=name)

    #Check if we need to re-sample
    if resample_freqs is not None:
        new_freqs = Frequency.from_f(resample_freqs)
        net.resample(new_freqs)

    return net

def sparams_to_mag(sparams, get_phase=False, in_db=False):
    """
    Computes the magnitude, and optionally the phase, of a given complex signal.

    Args:
        sparams: Signal with real and imaginary components.
                 Torch tensor [1, 2*num_unique_sparams, F]. 
                 We expect the real and imaginary components for each s-parameter to be adjacent
                    channels in the second axis. 
        get_phase: If True, returns the phase of the signal as well. 
        in_db: If True, will return the magnitude in Decibels.  

    Returns:
        mag: Magnitude of the given signal. If in_db is True, will be in Decibels.
             Torch tensor [1, num_unique_sparams, F]. 
        phase (optional): Phase of the given signal in radians. Only returned if get_phase is true.
                          Torch tensor [1, num_unique_sparams, F]. 
    """    

    x_complex = torch.complex(sparams[:, ::2, :], sparams[:, 1::2, :]) #evens are real, odds are imaginary

    x_mag = torch.abs(x_complex)
    if in_db:
        x_mag = 20 * torch.log10(x_mag) #multiply here by 20 instead of 10 because abs is a square root term

    if get_phase:
        x_phase = torch.angle(x_complex)
        return x_mag, x_phase
    else:
        return x_mag

def plot_mags(sparams_list, freqs=None, chip_labels=None, not_reciprocal=False, plot_subset_list=None, ylims=False):
    mag_list = [sparams_to_mag(sparam) for sparam in sparams_list]
    
    num_ports = (-1 + np.sqrt(8*mag_list[0].shape[1] + 1)) // 2
    num_ports = int(num_ports)
    if not_reciprocal:
        num_ports = int(np.sqrt(mag_list[0].shape[1]))
    
    x = freqs if freqs is not None else [np.arange(mag_list[0].shape[-1])]*len(mag_list)
    labels = chip_labels if chip_labels is not None else np.arange(len(mag_list))
    
    if not_reciprocal:
        sparam_str_list = [str(s_idx//num_ports) + str(s_idx%num_ports) for s_idx in range(mag_list[0].shape[1])]
    else:
        sparam_str_list = []
        for i in range(num_ports):
            for j in range(i+1):
                sparam_str_list.append(str(i) + str(j))
    
    plot_chips = plot_subset_list if plot_subset_list is not None else np.arange(len(mag_list))
    
    fig, axs = plt.subplots((mag_list[0].shape[1] + 1) // 2, 2, figsize=(20,15))
    
    for s_idx in range(mag_list[0].shape[1]):
        if s_idx >= mag_list[0].shape[1]:
            break
        for chip_idx in plot_chips:
            axs[s_idx//2, s_idx%2].plot(x[chip_idx], mag_list[chip_idx][0, s_idx], label=str(labels[chip_idx]))
        axs[s_idx//2, s_idx%2].set_title(sparam_str_list[s_idx])
        axs[s_idx//2, s_idx%2].legend()
        if ylims:
            axs[s_idx//2, s_idx%2].set_ylim(0, 1)
        
    fig.tight_layout()

class Measurement_MSE_Loss(nn.Module):
    """
    Given a signal x, observed measurements y, and observed indices kept_inds, 
        return the mse over the measurements of x vs true measurements y.
    
    Args:
        kept_inds: Array with the indices of the kept measurements.
        per_param: Whether to reduce the MSE for each S-param individually
                    before reducing the loss.
                   This can help with robustly fitting each S-param well.
        reduction: ["mean", "sum"]  
    """

    def __init__(self, kept_inds, per_param=False, reduction="mean"):
        super().__init__()

        self.kept_inds = kept_inds
        self.per_param = per_param
        self.reduction = reduction

        if not self.per_param:
            self.mse_loss = nn.MSELoss(reduction=self.reduction)
    
    def forward(self, x, y):
        if not self.per_param:
            return self.mse_loss(x[:, :, self.kept_inds], y)
        
        else:
            square_error = torch.square(x[:, :, self.kept_inds] - y) #[1, 2 * N_sparams, m]

            if self.reduction == "mean":
                mse_per_chan = torch.mean(square_error, dim=2) #[1, 2 * N_sparams]
                rmse_per_chan = torch.sqrt(mse_per_chan) #[1, 2 * N_sparams]
                
                return torch.mean(rmse_per_chan) 
            
            elif self.reduction == "sum":
                sse_per_chan = torch.sum(square_error, dim=2) #[1, 2 * N_sparams]
                rsse_per_chan = torch.sqrt(sse_per_chan) #[1, 2 * N_sparams]
                
                return torch.sum(rsse_per_chan) 

class Smoothing_Loss(nn.Module):
    """
    Loss function that penalizes nth-order differences in a time series.

    Args:
        per_param: Whether to reduce the loss for each S-param individually
                    before reducing the loss.
                   This can help with robustly fitting each S-param well.
        reduction: ["mean", "sum"]
    """
    def __init__(self, per_param=False, reduction="mean", order=2, norm=2):
        super().__init__()

        self.per_param = per_param
        self.reduction = reduction
        self.order = order
        self.norm = norm
    
    def forward(self, x):
        if self.order == 2:
            diffs = torch.diff(torch.diff(x, dim=2), dim=2)
        elif self.order == 3:
            diffs = torch.diff(torch.diff(torch.diff(x, dim=2), dim=2), dim=2)
        
        if self.norm == 1:
            diffs = torch.abs(diffs)
        elif self.norm == 2:
            diffs = torch.square(diffs)

        if not self.per_param:
            if self.reduction == "mean":
                return torch.mean(diffs)
            elif self.reduction == "sum":
                return torch.sum(diffs)
        
        else:
            if self.reduction == "mean":
                loss_per_chan = torch.mean(diffs, dim=2) #[1, 2 * N_sparams]
                rloss_per_chan = torch.sqrt(loss_per_chan) #[1, 2 * N_sparams]
                
                return torch.mean(rloss_per_chan) 
            
            elif self.reduction == "sum":
                loss_per_chan = torch.sum(diffs, dim=2) #[1, 2 * N_sparams]
                rloss_per_chan = torch.sqrt(loss_per_chan) #[1, 2 * N_sparams]
                
                return torch.sum(rloss_per_chan) 
