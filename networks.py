from collections import OrderedDict
import torch
from torch import nn
from typing import Union

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


class ConvClassifier(nn.module):
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
