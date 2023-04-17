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
		# store the convolution and RELU layers
        self.sequence = nn.Sequential(
            torch.nn.Conv2d(inChannels, outChannels, 3),
            torch.nn.BatchNorm2d(outChannels),
            torch.nn.Conv2d(inChannels, outChannels, 3),
            torch.nn.ReLU()
            # torch.nn.BatchNorm2d(out_channels),
            # torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
    def forward(self, x):
        # apply CONV => RELU => CONV block to the inputs and return it
        return self.sequence(x)

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
    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        self.channels = channels
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2) for i in range(len(channels) - 1)]
        )
        self.decodeBlocks = nn.ModuleList(
            [DownBlock(channels[i], channels[i+1]) for i in range(len(channels) - 1)]
        )
    
    def crop(self, encodedFeatures, x):
        # grab the dimensions of the inputs, and crop the encoder
		# features to match the dimensions
        (_, _, H, W) = x.shape
        encodedFeatures = nn.CenterCrop([H, W])(encodedFeatures)
        # return the cropped features
        return encodedFeatures
    
    def forward(self, x, encodedFeatures):
        for i in range(len(self.channels) - 1):
            # Pass thru upsampler blocks
            x = self.upconvs[i](x)

        # loop through the number of channels
        for i in range(len(self.channels) - 1):
			# Pass the inputs through upsampler block.
            x = self.upconvs[i](x)
			# Crop current features from encoder blocks, concatenate with the current upsampled features.
            encFeat = self.crop(encodedFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
			# Pass the concatenated output through the current decoder block.
            x = self.dec_blocks[i](x)
        return x

class UNet(nn.Module):
    """
    Basic U-Net Architecture.
    https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/
    Old version was I think needlyless complicated!
    """

    def __init__(self, encodeChannels=(3, 16, 32, 64), decodeChannels=(64, 32, 16),
                 numClasses=1, retainDimension=True, outSize=(32, 32)):
        super().__init__()
        # Initialize Encoder and Decoder
        self.encoder = Encoder(encodeChannels)
        self.decoder = Decoder(decodeChannels)
            # BELOW: from the version of this that's segmentation-oriented.
        # self.head = nn.Conv2d(decodeChannels[-1], numClasses, 1)
        # self.retainDimension = retainDimension
        # self.outSize = outSize

    def forward(self, x):
        encodedFeatures = self.encoder(x)
        
        # Pass encoder features through decoder, making sure dimensions are suited for concatenation
        # The first one is the final output (bottom of the U), and the rest are the features to work with
        decodedFeatures = self.decoder(encodedFeatures[::-1][0], encodedFeatures[::-1][1:])
        
        return decodedFeatures # Despite the plural, this is just one item.

            # BELOW: from the version of this that's segmentation-oriented.
        # # Pass  decoder features through regression head, obtain the segmentation mask
        # map = self.head(decodedFeatures)

        # # If we are retaining the original output dimensions then resize the output to match them
        # if self.retainDimension:
        #     map = F.interpolate(map, self.outSize)
        # return map
        
    

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



# The following is the pre-trained model for GTSRB:
nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(3, 100, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(100)
        self.conv2 = nn.Conv2d(100, 150, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 250, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(250)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(250*2*2, 350)
        self.fc2 = nn.Linear(350, nclasses)

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
            )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
            )
   
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform forward pass
        x = self.bn1(F.max_pool2d(F.leaky_relu(self.conv1(x)),2))
        x = self.conv_drop(x)
        x = self.bn2(F.max_pool2d(F.leaky_relu(self.conv2(x)),2))
        x = self.conv_drop(x)
        x = self.bn3(F.max_pool2d(F.leaky_relu(self.conv3(x)),2))
        x = self.conv_drop(x)
        x = x.view(-1, 250*2*2)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)