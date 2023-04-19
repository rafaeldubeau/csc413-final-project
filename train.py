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

def ImageLoss(originalImage, newImage):
    difference = torch.abs(originalImage - newImage)
    # Now we have pixel-wise difference between the two images. 
    gamma = 10
    # We will find the mean of the top gamma values and take that as the loss.
    values, _ = difference.topk(gamma, dim=-1)
    mean = torch.mean(values)
    return mean
    return nn.functional.mse_loss(originalImage, newImage)

def UNetLoss_baluja(originalImage, newImage, gtLabel, adversaryNetwork, beta=0.5, target=0):
    with torch.no_grad():
        # Compute reranking
        alpha = [5, 4, 3, 2]
        y = nn.functional.softmax(adversaryNetwork(originalImage), dim=-1)
        ymax, max_indices = y.max(dim=-1) # B x 1
        if target == -1:
            values, indices = y.topk(2, dim=-1) # B x 2
            t2 = indices[:, 1] # B x 1 of indices -- indicates the one with the second-highest probability.
            t3 = indices[:, 2] # B x 1 of indices -- indicates the one with the third-highest probability.
            t4 = indices[:, 2] # B x 1 of indices -- indicates the one with the fourth-highest probability.
            t5 = indices[:, 2] # B x 1 of indices -- indicates the one with the fifth-highest probability.
        else:
            t2 = target
        
        y[:, t2] = alpha[0] * ymax # This might break
        if target == -1:
            y[:, t3] = alpha[1] * ymax
            y[:, t4] = alpha[2] * ymax
            y[:, t5] = alpha[3] * ymax
        # print(y[0])
        y_prime = nn.functional.softmax(y, dim=-1)

    # Get prediction on on new image
    y = nn.functional.softmax(adversaryNetwork(newImage), dim=-1)

    # Compute losses
    L_x = ImageLoss(originalImage, newImage) # TODO: maybe we need to change this to before normalization
    L_y = nn.functional.mse_loss(y, y_prime)

    L = beta * L_x + L_y

    return L

def evaluate(model, data, adversaryNetwork, device, target=0):
    if model.training:
        model.eval()

    with torch.no_grad():
        # loss = 0
        ranking_acc = 0.0
        target_acc = 0.0
        for batch, (X, y) in enumerate(data):
            X, y = X.to(device), y.to(device)
            gen = model(X)
            gen = noisify(X, gen, 0)
            pred = adversaryNetwork(gen)
            # loss += loss_fn(pred, y).item()

            values, indices = pred.topk(2, dim=-1)
            
            ranking_acc += torch.count_nonzero(y == indices[:, 1])
            target_acc += torch.count_nonzero(target == indices[:, 0])
    
    # loss = loss / (len(data.dataset) / batch_size)
    ranking_acc = ranking_acc / len(data.dataset)
    target_acc = target_acc / len(data.dataset)

    print(f"Target Accuracy: {target_acc}, Ranking Accuracy: {ranking_acc}")
    return target_acc


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
        X = torch.sigmoid(X)
        plt.imshow(X)

        fig.add_subplot(n, 4, 4*i+3)
        plt.title(label_map[pred_label.item()])
        plt.axis("off")
        noised = noised.squeeze().detach().cpu()
        noised = torch.transpose(noised, 1, 2)
        noised = torch.transpose(noised, 0, 2)
        noised = torch.sigmoid(noised)
        plt.imshow(noised)

        fig.add_subplot(n, 4, 4*i+4)
        plt.title(label_map[pred_label.item()])
        plt.axis("off")
        gen = gen.squeeze().detach().cpu()
        gen = torch.transpose(gen, 1, 2)
        gen = torch.transpose(gen, 0, 2)
        plt.imshow(gen)
    plt.show()

def noisify(originalImage, generatedImage, epsilon): 
    return gtsrb_utils.just_normalize(generatedImage)
    # normalized = nn.functional.tanh(gtsrb_utils.just_normalize(generatedImage))
    # return originalImage + epsilon * normalized

def trainUNet(epochs: int, starting_epoch: int):
    # Hyperparameters
    learning_rate = 10e-4
    batch_size = 128
    alpha = 10e-6
    epsilon = 0.3 # Max influence the noise can have in the generated image
    beta = 0.4 # How much influence image closeness has on total loss
    target = 0

    # For measuring stats, measured per epoch
    train_losses = []
    validation_losses = []

    # Load model and set weights
    compareModel = gtsrb_utils.load_pretrained().to(device)

    model = UNet().to(device)
    model.train()

    # Load dataset
    total_dataset = gtsrb_utils.load_gtsrb_dataset()
    print("Length:", len(total_dataset))

    # Load Datasets
    train_size = int(len(total_dataset) * 0.6)
    val_size = int(len(total_dataset) * 0.15)
    train_set, val_set = torch.utils.data.random_split(total_dataset, 
                                (train_size, val_size)) 
    # For testing -- faster processing with a smaller training dataset
    # train_set, _, val_set = torch.utils.data.random_split(total_dataset, 
    #                             (int(len(total_dataset) * 0.4), int(len(total_dataset) * 0.3), int(len(total_dataset) * 0.3)))
    data_loader_train = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    data_loader_val = DataLoader(val_set, shuffle=True, batch_size=batch_size)

    # Load weights, if starting_epochs is above 1
    if starting_epoch > 0:
        model.load_state_dict(torch.load(os.path.join("data", "models", f"UNetTrain_{starting_epoch}.pth")))

    # Loss functions
    loss_fn = UNetLoss_baluja

    # Optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # demo(model, compareModel, epsilon, device)
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

            loss = loss_fn(X, noised_gen, y, compareModel)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"train loss: {loss:>7f}  [{current:>7d}/{len(data_loader_train.dataset):>7d}]")

        validate_loss = evaluate(model, data_loader_val, compareModel, device)
        train_losses.append(loss.item())
        validation_losses.append(validate_loss)
        
        # if (t+1) % 1 == 0:
            # demo(model, compareModel, epsilon, device)
        
    for i in range(epochs):
        plt.plot(range(epochs), train_losses, label="Train Loss", linestyle="-.")
        plt.plot(range(epochs), validation_losses, label="Validation Loss", linestyle=".")
    plt.legend()
    plt.show()
        


if __name__ == "__main__":
    trainUNet(10, 0)
