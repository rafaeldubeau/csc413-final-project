import os
import random
import numpy as np

import torch
import torchvision
from torch import nn
from torch.optim import Adam
from torchvision import transforms as transforms
from torch.utils.data import DataLoader

from gtsrb_utils import label_map

from networks import UNet

import matplotlib.pyplot as plt

import gtsrb_utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
# LOSS FUNCTION ZONE!

def UNetLoss_simple(originalImage, newImage, gtLabel, adversaryNetwork, beta=5):
    L_x = nn.functional.mse_loss(originalImage, newImage)

    pred = nn.functional.softmax(adversaryNetwork(newImage), dim=-1) # Tensor of size batch_size x nClasses
    L_y = pred[:, gtLabel].mean()

    L = beta * L_x + L_y

    return L


def evaluate(model, data, adversaryNetwork, device):
    if model.training:
        model.eval()

    with torch.no_grad():
        # loss = 0
        acc = 0
        for batch, (X, y) in enumerate(data):
            X, y = X.to(device), y.to(device)
            gen = model(X)
            pred = adversaryNetwork(gen)
            # loss += loss_fn(pred, y).item()
            
            acc += torch.count_nonzero(y == pred.argmax(dim=-1))
    
    # loss = loss / (len(data.dataset) / batch_size)
    acc = acc / len(data.dataset)

    print(f"Val Accuracy: {acc}")


def demo(model, adversary, device):
    path = os.path.join("data")
    dataset = torchvision.datasets.GTSRB(root=path, download=True, split="test")

    if model.training:
        model.eval()
    if adversary.training:
        adversary.eval()

    n = 3

    fig = plt.figure()
    for i in range(n):
        img, label = dataset[random.randint(0, len(dataset)-1)]
        
        X = gtsrb_utils.data_transforms(img).unsqueeze(0).to(device)

        og_label = adversary(X).argmax(dim=-1).squeeze()
        
        gen = model(X)
        pred_label = adversary(gen).argmax(dim=-1).squeeze()

        fig.add_subplot(n, 3, 3*i+1)
        plt.title(label_map[label])
        plt.axis("off")
        plt.imshow(img)

        fig.add_subplot(n, 3, 3*i+2)
        plt.title(label_map[og_label.item()])
        plt.axis("off")
        X = X.squeeze().detach().cpu()
        X = torch.transpose(X, 1, 2)
        X = torch.transpose(X, 0, 2)
        plt.imshow(X)

        fig.add_subplot(n, 3, 3*i+3)
        plt.title(label_map[pred_label.item()])
        plt.axis("off")
        gen = gen.squeeze().detach().cpu()
        perturbed = torch.transpose(gen, 1, 2)
        perturbed = torch.transpose(perturbed, 0, 2)
        plt.imshow(perturbed)
    plt.show()


def trainUNet(epochs: int, starting_epoch: int):
    # Hyperparameters
    learning_rate = 10e-4
    batch_size = 64
    alpha = 10e-6

    # Load model and set weights
    compareModel = gtsrb_utils.load_pretrained().to(device)

    model = UNet().to(device)
    model.train()

    # Load dataset
    total_dataset = gtsrb_utils.load_gtsrb_dataset()
    print("Length:", len(total_dataset))

    # Load Datasets
    train_set, val_set = torch.utils.data.random_split(total_dataset, (22644, 3996))
    data_loader_train = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    data_loader_val = DataLoader(val_set, shuffle=True, batch_size=batch_size)

    # Loss functions
    loss_fn = UNetLoss_simple

    # Optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Train loop
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        if not model.training:
            model.train()
        if not compareModel.training:
            compareModel.train()

        if (t+1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join("data", "models", f"UNetTrain_{starting_epoch+t+1}.pth"))

        for batch, (X, y) in enumerate(data_loader_train):
            X, y = X.to(device), y.to(device)
            X.requires_grad = True
            gen = model(X)

            loss = loss_fn(X, gen, y, compareModel)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 50 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"train loss: {loss:>7f}  [{current:>7d}/{len(data_loader_train.dataset):>7d}]")

        evaluate(model, data_loader_val, compareModel, device)
        demo(model, compareModel, device)
        


if __name__ == "__main__":
    trainUNet(10, 0)
