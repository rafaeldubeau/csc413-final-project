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

def train_epoch(model, data, optimizer, loss_fn, device):
    """
    Trains a given model for one epoch

    Args:
        model: the model being trained
        data: a DataLoader containing the training data
        optimizer: the optimizer training the model's parameters
        loss_fn: a function to compute the loss to be minimized

    Returns:
        Nothing
    
    """
    if not model.training:
        model.train()

    for batch, (X, y) in enumerate(data):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"train loss: {loss:>7f}  [{current:>7d}/{len(data.dataset):>7d}]")
            

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
    total_dataset = gtsrb_utils.load_gtsrb_dataset()
    print("Length:", len(total_dataset))

    train_set, val_set = torch.utils.data.random_split(total_dataset, (22644, 3996))

    # Make Train/Test Split
    # TODO: Fill this in. Use https://github.com/jfilter/split-folders potentially.

    # Dataloader should be JUST for train data.
    data_loader_train = DataLoader(train_set, shuffle=True)
    
    # Loss functions
    base_loss = nn.MSELoss()
    
    # TODO: Make this more involved.
    def UNetLoss(input, target):
        print('compareModel')
        return base_loss(input, target)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Train loop
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_epoch(model, data_loader_train, optimizer, UNetLoss, device)
        if (t+1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join("data", "models", f"UNetTrain_{starting_epoch+t+1}.pth"))


def train_copycat():
    
    # Hyperparameters
    epochs = 50
    learning_rate = 1e-3
    batch_size = 512
    weight_decay = 1e-2
    
    # Model Definition
    model = ConvClassifier(3, 1000)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Loss Function
    loss_fn = nn.CrossEntropyLoss()

    # Initialize Pretrained ResNet Model
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    # Load & Preprocess Data
    preprocess = weights.transforms()

    prediction = model(batch).softmax(dim=-1)
    class_ids = prediction.argmax(dim=-1)
    resnet_labels = nn.functional.one_hot(class_ids, num_classes=1000)

    # Training
    for e in range(epochs):
        train_epoch(model, train_dataloader, optimizer, loss_fn)
        if (e+1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join("data", "models", f"copycat_{e+1}.pth"))