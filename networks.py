from collections import OrderedDict
import torch
from torch import nn
from typing import Union

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UNet(nn.Module):
    """
    Basic U-Net Architecture.
    https://medium.com/analytics-vidhya/creating-a-very-simple-u-net-model-with-pytorch-for-semantic-segmentation-of-satellite-images-223aa216e705
    """

    def __init__(self, depth, convKernel, num_in_channels, num_filters, num_colours):
        super(UNet, self).__init__()

        stride = 2
        padding = convKernel // 2
        output_padding = 1

        conv1 = self.contract_block(in_channels=num_in_channels, out_channels=num_filters, kernel_size=convKernel, padding=1)
        conv2 = self.contract_block(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=convKernel, padding=padding)
        conv3 = self.contract_block(in_channels=num_filters * 2, out_channels=num_filters * 4, kernel_size=convKernel, padding=padding)
        
        upconv3 = self.expand_block(in_channels=num_filters * 4, out_channels=num_filters * 2, kernel_size=convKernel, padding=padding)
        upconv2 = self.expand_block(in_channels=num_filters * 2 * 2, out_channels=num_filters, kernel_size=convKernel, padding=padding)
        upconv2 = self.expand_block(in_channels=num_filters * 2, out_channels=num_in_channels, kernel_size=convKernel, padding=padding)
        
            # VERSION I RIPPED OFF FROM THE INTERNET -- keeping these for now for the numbers.
        # self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        # self.conv2 = self.contract_block(32, 64, 3, 1)
        # self.conv3 = self.contract_block(64, 128, 3, 1)

        # self.upconv3 = self.expand_block(128, 64, 3, 1)
        # self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        # self.upconv1 = self.expand_block(32*2, out_channels, 3, 1)

            # OLD VERSION
        # self.DownBlock2 = nn.Sequential(
        #     nn.Conv2d(num_in_channels, num_filters, kernel_size=convKernel, stride=stride, padding=1),
        #     nn.BatchNorm2d(num_filters),
        #     nn.ReLU()
        # )

        # self.DownBlock1 = nn.Sequential(
        #     nn.Conv2d(num_filters, num_filters, kernel_size=kernel, stride=stride, padding=1),
        #     nn.BatchNorm2d(num_filters),
        #     nn.ReLU()
        # )

        # self.BottomBlock = nn.Sequential(
        #     nn.ConvTranspose2d(num_filters, num_filters, kernel_size=kernel, stride=stride, dilation=1, padding=1, output_padding=output_padding),
        #     nn.BatchNorm2d(num_filters),
        #     nn.ReLU()
        # )

        # self.UpBlock1 = nn.Sequential(
        #     nn.ConvTranspose2d(num_filters * 2, num_filters, kernel_size=kernel, stride=stride, dilation=1, padding=1, output_padding=output_padding),
        #     nn.BatchNorm2d(num_filters),
        #     nn.ReLU()
        # )

        # self.UpBlock2 = nn.Sequential(
        #     nn.Conv2d(num_filters + num_in_channels, num_colours, kernel_size=kernel, stride=1, padding=padding),
        # )
    
    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        return nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
            # torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            # torch.nn.BatchNorm2d(out_channels),
            # torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
    
    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        return nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1) 
            )   
    
    def forward(self, input):
        # downsample
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        # upsample
        upconv3 = self.upconv3(conv3)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1
    

# I think we could delete this -will
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
