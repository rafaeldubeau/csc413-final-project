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

def UNetLoss_simple(originalImage, newImage, gtLabel, adversaryNetwork, beta=0.1):
    L_x = nn.functional.mse_loss(originalImage, newImage)

    pred = nn.functional.softmax(adversaryNetwork(newImage), dim=-1) # Tensor of size batch_size x nClasses
    L_y = pred[:, gtLabel].sum()

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


def demo(model, adversary, epsilon, device):
    path = os.path.join("data")
    dataset = torchvision.datasets.GTSRB(root=path, download=True, split="test")

    if model.training:
        model.eval()
    if adversary.training:
        adversary.eval()

    n = 4

    fig = plt.figure()
    for i in range(n):
        img, label = dataset[random.randint(0, len(dataset)-1)]
        resized = gtsrb_utils.just_resize(img).unsqueeze(0).to(device)

        X = gtsrb_utils.data_transforms(img).unsqueeze(0).to(device)
        og_label = adversary(X).argmax(dim=-1).squeeze()

        gen = model(X)
        noised = noisify(X, gen, epsilon)
        pred_label = adversary(noised).argmax(dim=-1).squeeze()

        untransformed = gtsrb_utils.inverse_transforms(noised).unsqueeze(0).to(device)

        fig.add_subplot(n, 4, 4*i+1)
        plt.title(label_map[label])
        plt.axis("off")
        resized = resized.squeeze().detach().cpu()
        resized = torch.transpose(resized, 1, 2)
        resized = torch.transpose(resized, 0, 2)
        plt.imshow(resized)

        fig.add_subplot(n, 4, 4*i+2)
        plt.title(label_map[og_label.item()])
        plt.axis("off")
        X = X.squeeze().detach().cpu()
        X = torch.transpose(X, 1, 2)
        X = torch.transpose(X, 0, 2)
        plt.imshow(X)

        fig.add_subplot(n, 4, 4*i+3)
        plt.title(label_map[pred_label.item()])
        plt.axis("off")
        noised = noised.squeeze().detach().cpu()
        noised = torch.transpose(noised, 1, 2)
        noised = torch.transpose(noised, 0, 2)
        plt.imshow(noised)

        fig.add_subplot(n, 4, 4*i+4)
        plt.title(label_map[pred_label.item()])
        plt.axis("off")
        untransformed = untransformed.squeeze().detach().cpu()
        untransformed = torch.transpose(untransformed, 1, 2)
        untransformed = torch.transpose(untransformed, 0, 2)
        plt.imshow(untransformed)
    plt.show()

def noisify(originalImage, generatedImage, epsilon): 
    # return generatedImage
    normalized = nn.functional.tanh(generatedImage)
    return originalImage + epsilon * normalized

def trainUNet(epochs: int, starting_epoch: int):
    # Hyperparameters
    learning_rate = 10e-4
    batch_size = 64
    alpha = 10e-6
    epsilon = 0.5 # Max influence the noise can have in the generated image
    beta = 0.1 # How much influence image closeness has on total loss

    # Load model and set weights
    compareModel = gtsrb_utils.load_pretrained().to(device)

    model = UNet().to(device)
    model.train()

    # Load dataset
    total_dataset = gtsrb_utils.load_gtsrb_dataset()
    print("Length:", len(total_dataset))

    # Load Datasets
    train_size = int(len(total_dataset) * 0.5)
    test_size = int(len(total_dataset) * 0.3)
    val_size = int(len(total_dataset) * 0.2)
    train_set, test_set, val_set = torch.utils.data.random_split(total_dataset, (train_size, test_size, val_size))
    data_loader_train = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    data_loader_val = DataLoader(val_set, shuffle=True, batch_size=batch_size)

    # Loss functions
    loss_fn = UNetLoss_simple

    # Optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)

    demo(model, compareModel, epsilon, device)
    # return

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
            noised_gen = noisify(X, gen, epsilon)

            loss = loss_fn(X, noised_gen, y, compareModel, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 25 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"train loss: {loss:>7f}  [{current:>7d}/{len(data_loader_train.dataset):>7d}]")

        evaluate(model, data_loader_val, compareModel, device)
        demo(model, compareModel, epsilon, device)
        


if __name__ == "__main__":
    trainUNet(10, 0)
