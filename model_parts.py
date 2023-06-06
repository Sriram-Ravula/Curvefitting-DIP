import torch
import torch.nn as nn

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

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        pad = (kernel_size - 1) // 2

        self.input_layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=pad, padding_mode='reflect', bias=False),
            nn.BatchNorm1d(out_channels, affine=False),
            nn.LeakyReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=pad, padding_mode='reflect', bias=False)
        )
    
    def forward(self, x):
        return self.input_layer(x)

class ResidualConv(nn.Module):
    """
    Takes the input through 2 paths and sums the output.
    Path 1: BN -> LeakyReLU -> Conv -> AvgPool -> BN -> LReLU -> Conv
    Path 2: 1x1 Conv -> AvgPool.
    If argument "downsample" is false, then no avg pooling    

    AvgPool uses ceil_mode=True so for odd-sized inputs, output_len = (input_len + 1)/2.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, downsample=False):
        super().__init__()

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
        else:
            self.conv_block = nn.Sequential(
                nn.BatchNorm1d(in_channels, affine=False),
                nn.LeakyReLU(),
                nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, padding=pad, padding_mode='reflect', bias=False),
                nn.BatchNorm1d(mid_channels, affine=False),
                nn.LeakyReLU(),
                nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, padding=pad, padding_mode='reflect', bias=False)
            )
    
    def forward(self, x):
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
    def __init__(self, K=1):
        super().__init__()

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
