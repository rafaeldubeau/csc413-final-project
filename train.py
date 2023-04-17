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
            

def trainUNet(names: list[str], epochs: int, starting_epoch: int, lipschitz: bool):
    # Hyperparameters
    learning_rate = 10e-4
    batch_size = 512
    alpha = 10e-6

    # Load dataset
    dataset = datasets.GTSRB(root= "./model", download=True, transform=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True)

    # Loss function.
    base_loss = nn.MSELoss()
    
    def UNetLoss(input, target):
        pass

    # loss_fn = lambda pred, y, model: base_loss(pred.squeeze(), y.squeeze()) + alpha * model.get_lipschitz_bound()

    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Train loop
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_epoch(model, )
        train_loop(model, train_dataloader, optimizer, loss_fn)
        eval_loop(model, test_dataloader, loss_fn)
        if (t+1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join("data", "models", f"{filename}_{starting_epoch+t+1}.pth"))


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