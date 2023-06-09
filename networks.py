from collections import OrderedDict
import torch
from typing import Union
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DownBlock(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(inChannels, outChannels, kernel_size=3, padding="same")
        self.batchNorm = torch.nn.BatchNorm2d(outChannels)
        self.ReLU = torch.nn.ReLU()

    def forward(self, x):
        # apply CONV => RELU => CONV block to the inputs and return it
        y = self.conv1(x)
        y = self.batchNorm(y)
        y = self.ReLU(y)
        return y


class Encoder(nn.Module):
    def __init__(self, channels=(3, 16, 32, 64, 128)):
        super().__init__()
        self.encodeBlocks = nn.ModuleList(
			[DownBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        blockOutputs = []
        for block in self.encodeBlocks:
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)
        return blockOutputs
        
class Decoder(nn.Module):
    def __init__(self, channels=(128, 64, 32, 16, 3)):
        super().__init__()
        self.channels = channels
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(channels[i], channels[i+1], 3, stride=2, padding=1, dilation=1, output_padding=1) for i in range(len(channels) - 2)]
        )
        self.decodeBlocks = nn.ModuleList(
            [DownBlock(channels[i], channels[i+1]) for i in range(len(channels) - 1)]
        )
        self.out_layer = nn.Conv2d(3, 3, kernel_size=3, padding="same")
    
    def forward(self, encodedFeatures):
        x = encodedFeatures[-1]
        for i in range(len(self.channels) - 2):
			# Pass the inputs through upsampler block.
            y = self.upconvs[i](x)
			# Crop current features from encoder blocks, concatenate with the current upsampled features.
            y = torch.cat([y, encodedFeatures[-(i+2)]], dim=1)
			# Pass the concatenated output through the current decoder block.
            x = self.decodeBlocks[i](y)

        # Apply last block (no upsampling or concatenation)
        x = self.decodeBlocks[-1](x)

        # Apply last layer (no nonlinearity)
        x = self.out_layer(x)

        return x

class UNet(nn.Module):
    """
    Basic U-Net Architecture.
    https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/
    Old version was I think needlyless complicated!
    """

    def __init__(self, channels=(3, 16, 32, 64, 128), normalize=True):
        super().__init__()
        decodeChannels = channels[::-1]
        # Initialize Encoder and Decoder
        self.encoder = Encoder(channels)
        self.decoder = Decoder(decodeChannels)

        self.normalize = normalize

    def forward(self, x):
        encodedFeatures = self.encoder(x)
        # Pass encoder features through decoder, making sure dimensions are suited for concatenation
        # The first one is the final output (bottom of the U), and the rest are the features to work with
        output = self.decoder(encodedFeatures)

        if self.normalize:
            output = torch.sigmoid(output)
        
        return output 
    

class Conv2dBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int):
        super(Conv2dBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding="same"),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=out_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ConvClassifier(nn.Module):
    def __init__(self, in_channels: int, out_dim: int, num_filters: int = 32, kernel_size: int = 3, num_hidden: int=3):
        super(ConvClassifier, self).__init__()
        
        layer_dict = OrderedDict()

        layer_dict["conv_0"] = Conv2dBlock(in_channels, num_filters, kernel_size=kernel_size)

        for i in range(num_hidden):
            layer_dict[f"conv_{i+1}"] = Conv2dBlock((1 if i == 0 else 2) * num_filters, 2 * num_filters, kernel_size)
        
        layer_dict["flatten"] = nn.Flatten()
        layer_dict["FC"] = nn.LazyLinear(out_dim)

        self.layers = nn.Sequential(layer_dict)
    
    def forward(self, x):
        return self.layers(x)

