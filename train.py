import os
import numpy as np

import torch
import torchvision
from torch import nn
from torch.optim import Adam
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets
from torchvision import transforms as transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader


import matplotlib.pyplot as plt

from networks import ConvClassifier
import gtsrb_utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            

def trainUNet(epochs: int, starting_epoch: int):
    # Hyperparameters
    learning_rate = 10e-4
    batch_size = 512
    alpha = 10e-6

    # Load model and set weights
    from networks import Net
    compareModel = gtsrb_utils.load_pretrained()

    from networks import UNet
    model = UNet().to(device)
    model.train()
    # def __init__(self, convKernel, num_in_channels, num_filters, num_colours):


    # Load dataset
    data_loader = gtsrb_utils.load_gtsrb_dataloader()

    # Make Train/Test Split
    # TODO: Fill this in. Use https://github.com/jfilter/split-folders potentially.
    # Dataloader should be JUST for train data.
    data_loader_train = data_loader

    # Loss Function
    loss_fn = nn.MSELoss()

    # Optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Train loop
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        if not model.training:
            model.train()

        for batch, (X, _) in enumerate(data_loader_train):
            X = X.to(device)
            pred = model(X)
            loss = loss_fn(pred, X) # Learns the identity function

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 50 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"train loss: {loss:>7f}  [{current:>7d}/{len(data_loader_train.dataset):>7d}]")
