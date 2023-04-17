import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Using this for guidance. I think it has a nice way of doing things...our depth should definitely be flexible
# https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-model-building-6ab09d6a0862

class ConcatLayer(nn.Module):
    def __init__(self):
        super(ConcatLayer, self).__init__()
    
    def forward(self, layer1, layer2):
        return torch.cat((layer1, layer2), 1)

class DownBlock(nn.Module):
    """
    Convolution -> BatchNorm2d -> ReLU
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                #  pooling: bool = True,
                #  normalization: str = None,
                #  dim: str = 2,
                #  conv_mode: str = 'same'
                 ):
        super.__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.pooling = pooling
        # self.normalization = normalization
        self.padding = 1
        # self.dim = dim
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=self.padding, bias=True)
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        mid = self.conv(x)
     

class UNet(nn.Module):
    """
    Basic U-Net Architecture.
    """

    stride = 1

    def __init__(self, convKernel, num_in_channels, num_filters, num_colours):
        super(UNet, self).__init__()

        stride = 2
        padding = convKernel // 2
        output_padding = 1

        self.DownBlock2 = nn.Sequential(
            nn.Conv2d(num_in_channels, num_filters, kernel_size=convKernel, stride=stride, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )

        self.DownBlock1 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=kernel, stride=stride, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )

        self.BottomBlock = nn.Sequential(
            nn.ConvTranspose2d(num_filters, num_filters, kernel_size=kernel, stride=stride, dilation=1, padding=1, output_padding=output_padding),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )

        self.UpBlock1 = nn.Sequential(
            nn.ConvTranspose2d(num_filters * 2, num_filters, kernel_size=kernel, stride=stride, dilation=1, padding=1, output_padding=output_padding),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )

        self.UpBlock2 = nn.Sequential(
            nn.Conv2d(num_filters + num_in_channels, num_colours, kernel_size=kernel, stride=1, padding=padding),
        )
    
    def forward(self, input):
        out1 = self.DownBlock2(input)
        out2 = self.DownBlock1(out1)
        out3 = self.BottomBlock(out2)

        upCat1 = torch.cat((out3, out1), dim=1)
        out4 = self.UpBlock1(upCat1)
        
        upCat2 = torch.cat((out4, x), dim=1)
        outFinal = self.UpBlock2(upCat2)

        return outFinal

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
