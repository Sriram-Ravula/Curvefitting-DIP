import torch
import torch.nn as nn

import utils

class DoubleConv(nn.Module):
    """(convolution => [BN] => LeakyReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super().__init__()

        pad = (kernel_size - 1) // 2
        
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, padding=pad, padding_mode='reflect', bias=False),
            nn.BatchNorm1d(mid_channels, affine=False),
            nn.LeakyReLU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, padding=pad, padding_mode='reflect', bias=False),
            nn.BatchNorm1d(out_channels, affine=False),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class SingleConv(nn.Module):
    """(convolution => [BN] => LeakyReLU)"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        pad = (kernel_size - 1) // 2
        
        self.single_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=pad, padding_mode='reflect', bias=False),
            nn.BatchNorm1d(out_channels, affine=False),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.single_conv(x)

class Down(nn.Module):
    """Conv, Down with avgpool, BN, LeakyReLU, then single conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        pad = (kernel_size - 1) // 2
        
        self.down_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=pad, padding_mode='reflect', bias=False),
            nn.AvgPool1d(2), 
            nn.BatchNorm1d(out_channels, affine=False),
            nn.LeakyReLU(),
            SingleConv(out_channels, out_channels, kernel_size=kernel_size)
        )

    def forward(self, x):
        return self.down_conv(x)

class Up_NoCat(nn.Module):
    """Upscaling then double conv, with no concatenation"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='linear')
        self.conv = DoubleConv(in_channels, out_channels, in_channels, kernel_size=kernel_size)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        
        return x

class OutConv(nn.Module):
    """1x1 convolutions to get correct output channels"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv(x)

class InputResidualConv(nn.Module):
    """
    Takes the input through 2 paths and sums the output.
    Path 1: Conv -> BN -> LeakyReLU -> Conv
    Path 2: 1x1 Conv  
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, use_skip=True):
        super().__init__()

        self.use_skip = use_skip

        pad = (kernel_size - 1) // 2

        self.input_layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=pad, padding_mode='reflect', bias=False),
            nn.BatchNorm1d(out_channels, affine=False),
            nn.LeakyReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=pad, padding_mode='reflect', bias=False)
        )

        if self.use_skip:
            self.input_skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            )
    
    def forward(self, x):
        if self.use_skip:
            return self.input_layer(x) + self.input_skip(x)
        else:
            return self.input_layer(x)

class ResidualConv(nn.Module):
    """
    Takes the input through 2 paths and sums the output.
    Path 1: BN -> LeakyReLU -> Conv -> AvgPool -> BN -> LReLU -> Conv
    Path 2: 1x1 Conv -> AvgPool.
    If argument "downsample" is false, then no avg pooling    

    AvgPool uses ceil_mode=True so for odd-sized inputs, output_len = (input_len + 1)/2.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, downsample=False, use_skip=True):
        super().__init__()

        self.use_skip = use_skip

        pad = (kernel_size - 1) // 2

        if mid_channels is None:
            mid_channels = out_channels

        if downsample:
            self.conv_block = nn.Sequential(
                nn.BatchNorm1d(in_channels, affine=False),
                nn.LeakyReLU(),
                nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, padding=pad, padding_mode='reflect', bias=False),
                nn.AvgPool1d(2, ceil_mode=True), 
                nn.BatchNorm1d(mid_channels, affine=False),
                nn.LeakyReLU(),
                nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, padding=pad, padding_mode='reflect', bias=False)
            )

            if self.use_skip:
                self.conv_skip = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.AvgPool1d(2, ceil_mode=True) 
                )

        else:
            self.conv_block = nn.Sequential(
                nn.BatchNorm1d(in_channels, affine=False),
                nn.LeakyReLU(),
                nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, padding=pad, padding_mode='reflect', bias=False),
                nn.BatchNorm1d(mid_channels, affine=False),
                nn.LeakyReLU(),
                nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, padding=pad, padding_mode='reflect', bias=False)
            )

            if self.use_skip:
                self.conv_skip = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False) 
                )
    
    def forward(self, x):
        if self.use_skip:
            return self.conv_block(x) + self.conv_skip(x)
        else:
            return self.conv_block(x)

class UpConv(nn.Module):
    """
     Linear Upsampling -> 1D conv
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        pad = (kernel_size - 1) // 2

        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear'),
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=pad, padding_mode='reflect', bias=False) 
        )

    def forward(self, x):
        return self.up_conv(x)

def crop_and_cat(x1, x2):
    """
    Crops x1 to match x2 in length, then cats the two and return.
    """
    diff = x1.size()[-1] - x2.size()[-1]

    if diff > 0:
        return torch.cat([x1[...,:-diff], x2], dim=1)
    else:
        return torch.cat([x1, x2], dim=1)

class CausalityLayer(nn.Module):
    """
    Layer that enforces causality. 
    Takes a [N, C, FM + 1] real-valued input and returns a [N, 2*C, FK + 1] real output.
        F is the number of frequency points, M is the extrapolation factor, K is upsampling factor.

    Has no trainable parameters.
    """

    def __init__(self, K=1):
        super().__init__()

        # self.F = F
        self.K = K
    
    def forward(self, x, F=None):
        #L = FM + 1
        N, C, L = x.shape

        #If output size is not specific, assume upsampling factor is 2
        if F is None:
            F = (L-1)//2

        #(1) make the double-sided frequency spectrum
        #output is real and has shape [N, C, 2L-1 = 2FM + 1]
        double_x = torch.zeros(N, C, 2*L - 1, device=x.device, dtype=x.dtype)
        double_x[..., 0:L] = x
        double_x[..., L:] = x.flip(-1)[..., 1:]

        #(2) take the FFT
        #output is complex and has shape [N, C, 2FM + 1]
        FFT_double_x = torch.fft.fft(double_x) 

        #(3)upsample and make the signal analytic
        #output is complex and has shape [N, C, 2FMK + K]
        analytic_x = torch.zeros(N, C, self.K*(FFT_double_x.shape[-1]), device=FFT_double_x.device, dtype=FFT_double_x.dtype)
        analytic_x[..., 0] = FFT_double_x[..., 0]
        analytic_x[..., 1:L] = 2 * FFT_double_x[..., 1:L]

        #(3) Take the IFFT and truncate
        #output is complex and has shape [N, C, FK + 1]
        #NOTE there might need to be a "+1" in the parenthesis at the end of the second line
        #   right now this returns a FK-len last dimension, but I think that's actually better
        IFFT_analytic = torch.fft.ifft(analytic_x)
        truncated_IFFT_analytic = IFFT_analytic[..., 0:(self.K*self.F)] 

        #(4) Split the signal into real and imaginary parts and return
        #output is real and has shape [N, 2C, FK + 1]
        #NOTE see above - current length of the output is FK
        evens = [i for i in range(2*C) if i%2 == 0]
        odds = [i for i in range(2*C) if i%2 != 0]

        output = torch.zeros(N, 2*C, truncated_IFFT_analytic.shape[-1], device=x.device, dtype=x.dtype)
        output[:, evens, :] = self.K * truncated_IFFT_analytic.real
        output[:, odds, :] = -1 * self.K * truncated_IFFT_analytic.imag

        return output
    
class NewCausalityLayer(nn.Module):
    def __init__(self, K=1):
        super().__init__()

        # self.F = F #number of output frequencies
        self.K = K
    
    def forward(self, x, F=None):
        #x is the extrapolated real part of the frequency response
        #L = FM, where M is the extrapolation factor
        #C is the number of unique s-parameters
        N, C, L = x.shape
    
        #If output size is not specific, assume upsampling factor is 2
        if F is None:
            F = L//2
        
        #(1) Make the double-sided frequency spectrum
        #output is real and has size [N, C, 2L-1 = 2FM - 1]
        #NOTE different from paper - we do [x[0,...,FM-1], x[FM-1,...,1]] - this makes signal really even!
        double_x = torch.zeros(N, C, 2*L - 1, device=x.device, dtype=x.dtype)
        double_x[..., 0:L] = x
        double_x[..., L:] = x.flip(-1)[..., 0:-1] #exclude the x[0] term at the end - we only want one x[0]!

        #(2) Take the FFT of the double-sided spectrum
        #output is complex and has the shape [N, C, 2L-1 = 2FM-1]
        FFT_double_x = torch.fft.fft(double_x) 

        #(3) Upsample and make the signal analytic
        #output is complex and has shape [N, C, 2LK-K = 2FMK-K]
        analytic_x = torch.zeros(N, C, self.K*(FFT_double_x.shape[-1]), device=FFT_double_x.device, dtype=FFT_double_x.dtype)
        analytic_x[..., 0] = FFT_double_x[..., 0]
        analytic_x[..., 1:L] = 2 * FFT_double_x[..., 1:L]

        #(4) Take the IFFT and truncate
        #output is complex and has shape [N, C, FK]
        IFFT_analytic = torch.fft.ifft(analytic_x)
        truncated_IFFT_analytic = IFFT_analytic[..., 0:(self.K*F)]

        #(5) Split the signal into the real and imaginary components and return
        #output is real and has the shape [N, C, FK]
        evens = [i for i in range(2*C) if i%2 == 0]
        odds = [i for i in range(2*C) if i%2 != 0]

        output = torch.zeros(N, 2*C, truncated_IFFT_analytic.shape[-1], device=x.device, dtype=x.dtype)
        output[:, evens, :] = self.K * truncated_IFFT_analytic.real
        output[:, odds, :] = -1 * self.K * truncated_IFFT_analytic.imag

        return output

class PassivityLayer(nn.Module):
    """
    Layer that enforces passivity. 
        Filters the input to have singular value <= 1 at all frequencies.
        
    Takes a [N, 2C, L] real-valued input and returns a [N, 2C, L] real output.

    Has no trainable parameters.
    """

    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        N, C2, L = x.shape

        #(1) Grab the [L, NUM_PORTS] singular values and isolate the largest [L] - one from each frequency 
        sing_vals = utils.sparams_to_sing_vals(x)[:, 0]

        #(2) Find the [L] magnitude spectrum 
        filter_mag = 1. / (1. + nn.functional.relu(sing_vals - 1)) 

        #(3) Get the Phase: Find the Hilbert transform of the log-magnitudes

        #(3a) make the log-magnitude
        log_mag = torch.log(filter_mag) #[L]

        #(3b) make the double-sided frequency spectrum
        #output is real and has shape [2L-1]
        double_x = torch.zeros(2*L - 1, device=log_mag.device, dtype=log_mag.dtype)
        double_x[0:L] = log_mag
        double_x[L:] = log_mag.flip(-1)[1:]

        #(3c) take the FFT
        #output is complex and has shape [2L-1]
        FFT_double_x = torch.fft.fft(double_x) 

        #(3d) make the signal analytical
        #output is complex and has shape [2L-1]
        analytic_x = torch.zeros(FFT_double_x.shape[-1], device=FFT_double_x.device, dtype=FFT_double_x.dtype)
        analytic_x[0] = FFT_double_x[0]
        analytic_x[1:L] = 2 * FFT_double_x[1:L]

        #(3e) Take the IFFT and truncate
        #output is complex and has shape [L]
        IFFT_analytic = torch.fft.ifft(analytic_x)
        truncated_IFFT_analytic = IFFT_analytic[0:L] 

        #(3f) Grab just the imaginary part of the output to obtain Hilbert
        #output is real and has shape [L]
        filter_phase = truncated_IFFT_analytic.imag

        #(4) Make the filter with the magnitude and phase
        #complex with shape [L]
        passive_filter = torch.polar(filter_mag, filter_phase) #NOTE check hz vs radians for phase

        #(5) Filter and return
        #output is [N, 2C, L] 
        evens = [i for i in range(C2) if i%2 == 0]
        odds = [i for i in range(C2) if i%2 != 0]

        complex_output = torch.complex(x[:, evens, :], x[:, odds, :]) * passive_filter #[N, C, L] complex times [L] complex

        output = torch.zeros(N, C2, L, device=x.device, dtype=x.dtype)
        output[:, evens, :] = complex_output.real
        output[:, odds, :] = complex_output.imag

        return output
