import torch.nn as nn
from model_parts import * 

class RESNET_BACKBONE(nn.Module):
    def __init__(self, ngf, ngf_in_out, kernel_size, num_layers):
        """
        Class that acts as a backbone generator. Has downsampling and upsampling layers, but no
            input or output layers.
        
        Args:
            ngf: base number of filters per layer. can be a list - then must be length num_layers, symmetric, ordered from encoder.
            ngf_in_out: the number of channels in the input/output of this network
            kernel_size: can be a list - then must be length num_layers, symmetric, ordered from encoder.
            num_layers: the number of layers in the encoder/decoder. same as the numebr of resolution scales
        """
        super().__init__()

        ###########
        #  PARAMS #
        ###########
        self.ngf = ngf
        self.ngf_in_out = ngf_in_out
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        ###########
        #NET STUFF#
        ###########
        if not isinstance(self.kernel_size, list):
            self.kernel_size = [self.kernel_size] * self.num_layers
        
        if not isinstance(self.ngf, list):
            self.ngf = [self.ngf] * self.num_layers
        
        #the first encoder layer is not a downsampling layer
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for l in range(self.num_layers):
            self.encoder.append(ResidualConv(in_channels=self.ngf_in_out if l==0 else self.ngf[l-1], 
                                             out_channels=self.ngf[l], 
                                             kernel_size=self.kernel_size[l], 
                                             downsample=True))
            
            #decoder at depth l takes input from encoder at depth l-1
            #this is because the decoder at depth 0 takes input from an external source not defined here
            self.decoder.append(ResidualConv(in_channels=2*self.ngf_in_out if l==0 else 2*self.ngf[l-1], 
                                             out_channels=self.ngf_in_out if l==0 else self.ngf[l-1], 
                                             kernel_size=self.kernel_size[l], 
                                             downsample=False))

            self.upsamples.append(UpConv(in_channels=self.ngf[l], 
                                         out_channels=self.ngf_in_out if l==0 else self.ngf[l-1]))

    def forward(self, x):
        #encode
        out = x
        intermediate_outs = [out]
        for enc_layer in self.encoder:
            out = enc_layer(out)
            intermediate_outs.append(out)

        #decode
        i = -2 #start from -2 since we don't want to use the last encoder layer's output twice
        for up_layer, dec_layer in zip(self.upsamples[::-1], self.decoder[::-1]):
            out = up_layer(out)
            out = crop_and_cat(out, intermediate_outs[i])
            out = dec_layer(out)
            i -= 1

        return out
    
class RESNET_HEAD(nn.Module):
    def __init__(self, nz, ngf_in_out, nc, kernel_size, causal):
        """
        Acts as the input and output layers for a Resnet generator.

        Args:
            nz: the channel depth of the initial random seed.
            ngf_in_out: number of filters in the input and output layers. should match the backbone. 
            nc: number of channels in the output.
            kernel_size: length of the convolutional kernel in the input and output.
            causal: if True, adds a causality layer at the end of the network.
        """
        super().__init__()
        
        ###########
        #  PARAMS #
        ###########
        self.nz = nz
        self.ngf_in_out = ngf_in_out
        self.nc = nc
        self.kernel_size = kernel_size
        self.causal = causal
        
        ###########
        #NET STUFF#
        ###########
        self.input = InputResidualConv(in_channels=self.nz, 
                                       out_channels=self.ngf_in_out, 
                                       kernel_size=self.kernel_size)
        
        if self.causal:
            self.output = nn.Sequential(
                UpConv(in_channels=self.ngf_in_out, out_channels=self.ngf_in_out),
                ResidualConv(in_channels=self.ngf_in_out, 
                            out_channels=self.nc//2, 
                            mid_channels=self.nc, 
                            kernel_size=self.kernel_size, 
                            downsample=False),
                CausalityLayer()
            )
        else:
            self.output = OutConv(self.ngf_in_out, self.nc)
    
    def forward(self, x):
        """
        NOTE this method is a dummy, not meant to be used.
        The network must send the result of its input layer to another network
            and get the argument of its output layer from that network. 
        """
        out = self.input(x)
        return self.output(out)

    def forward_input(self, x):
        return self.input(x)
    
    def forward_output(self, x):
        return self.output(x)

class MODULAR_RESNET(nn.Module):
    def __init__(self, backbone, head):
        """
        Class that glues the backbone and head resnets to make one generator. 

        Args:
            backbone: the backbone of the resnet.
            head: the input and output layers of the glued generator. 
        """
        super().__init__()

        self.backbone = backbone
        self.head = head
    
    def forward(self, x):
        out = self.head.forward_input(x)
        out = self.backbone.forward(out)
        out = self.head.forward_output(out)

        return out
        