import os
import random
import numpy as np
import pickle

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

from piqa import PSNR, SSIM, VSI, GMSD
    # mdsi: Too similar, maybe even a bit better than, MSE. So not gonna use.
    # VSI: Seems to mess up and give us bad errors which lead to problems.

def ImageLossMaxSquareError(originalImage, newImage, autoAdjust=True):
    difference = torch.square(originalImage - newImage)
    # Now we have pixel-wise difference between the two images. 
    gamma = 10
    # We will find the mean of the top gamma values and take that as the loss.
    values, _ = difference.topk(gamma, dim=-1)
    mean = torch.mean(values)
    return mean

def ImageLossPIQA(originalImage, newImage, autoAdjust=True):
    gmsd = GMSD().cuda()    # Gradient Magnitude Similarity Deviation: Image gets noisy but it works well.
    # psnr = PSNR()           # Signal-To-Noise Ratio: seems to detect features in the image
    # ssim = SSIM().cuda()    # Structural similarity: seems to do best detecting texture/pattern
    # print('PSNR:', psnr(originalImage, newImage))
    # print('SSIM:', ssim(originalImage, newImage))
    # ssim_error = torch.abs(ssim(torch.sigmoid(originalImage), torch.sigmoid(newImage)))
    # psnr_error = psnr(torch.sigmoid(originalImage), torch.sigmoid(newImage))
    gmsd_error = gmsd(torch.sigmoid(originalImage), torch.sigmoid(newImage))
    # mse_error = ImageLossMSE(originalImage, newImage)
    # # stacked = torch.stack([0.5 * psnr_error, 5 * mse_error, 10 * ssim_error])
    # stacked = torch.stack([50 * gmsd_error, 0.08 * psnr_error, 0.5 * mse_error])
    # return torch.mean(stacked)
    return gmsd_error


# Alone, does a good job replicating the original image and modifying a small region to misclassify.
def ImageLossMSE(originalImage, newImage, autoAdjust=True):
    # betaLocal = 5 if autoAdjust else 1  # This number has seemed to work well for MSE.
    betaLocal = 1
    return betaLocal * nn.functional.mse_loss(originalImage, newImage)

def ImageLoss(originalImage, newImage, autoAdjust=True):
    return ImageLossPIQA(originalImage, newImage, autoAdjust=True)

def UNetLoss_baluja(originalImage, newImage, gtLabel, adversaryNetwork, beta=0.5, target=-1):
    with torch.no_grad():
        # Compute reranking
        alpha = [5, 4, 3, 2]
        y = nn.functional.softmax(adversaryNetwork(originalImage), dim=-1)
        ymax, max_indices = y.max(dim=-1) # B x 1
        if target == -1:
            values, indices = y.topk(5, dim=-1) # B x 2
            t2 = indices[:, 1] # B x 1 of indices -- indicates the one with the second-highest probability.
            t3 = indices[:, 2] # B x 1 of indices -- indicates the one with the third-highest probability.
            t4 = indices[:, 3] # B x 1 of indices -- indicates the one with the fourth-highest probability.
            t5 = indices[:, 4] # B x 1 of indices -- indicates the one with the fifth-highest probability.
            target = torch.mode(t2)[0].item()
        else:
            t2 = target
        
        y[:, t2] = alpha[0] * ymax # This might break
        if target == -1:
            y[:, t3] = alpha[1] * ymax
            y[:, t4] = alpha[2] * ymax
            y[:, t5] = alpha[3] * ymax
        y_prime = nn.functional.softmax(y, dim=-1)

    # Get prediction on on new image
    y = nn.functional.softmax(adversaryNetwork(newImage), dim=-1)

    # Compute losses
    L_x = ImageLoss(originalImage, newImage) # TODO: maybe we need to change this to before normalization
    L_y = nn.functional.mse_loss(y, y_prime)

    L = beta * L_x + L_y

    return L, target

def evaluate(model, data, adversaryNetwork, loss_fn, device, target=-1):
    if model.training:
        model.eval()

    with torch.no_grad():
        loss = 0
        ranking_acc = 0.0
        target_acc = 0.0
        for batch, (X, y) in enumerate(data):
            X, y = X.to(device), y.to(device)
            gen = model(X)
            gen = noisify(X, gen, 0)
            pred = adversaryNetwork(gen)
            l, _ = loss_fn(X, gen, y, adversaryNetwork)
            loss += l.item()

            values, indices = pred.topk(2, dim=-1)
            
            ranking_acc += torch.count_nonzero(y == indices[:, 1])
            # if target != -1:
            target_acc += torch.count_nonzero(target == indices[:, 0])
            # else:
                # target_acc += torch.count_nonzero(target != indices[:, 0])
    
    # loss = loss / (len(data.dataset) / batch_size)
    ranking_acc = ranking_acc / len(data.dataset)
    target_acc = target_acc / len(data.dataset)

    print(f"Target Accuracy: {target_acc}, Ranking Accuracy: {ranking_acc}")
    return target_acc, ranking_acc

def demo(model, adversary, epsilon, device, epoch, target, saveLocation):
    path = os.path.join("data")
    dataset = torchvision.datasets.GTSRB(root=path, download=True, split="test")

    if model.training:
        model.eval()
    if adversary.training:
        adversary.eval()

    n = 4

    fig = plt.figure(figsize=(10,7))
    for i in range(n):
        img, label = dataset[random.randint(0, len(dataset)-1)]
        resized = gtsrb_utils.just_resize(img).unsqueeze(0).to(device)

        X = gtsrb_utils.data_transforms(img).unsqueeze(0).to(device)
        og_label = adversary(X).argmax(dim=-1).squeeze()

        gen = model(X)
        noised = noisify(X, gen, epsilon)
        pred_label = adversary(noised).argmax(dim=-1).squeeze()

        untransformed = gtsrb_utils.inverse_transforms(noised).unsqueeze(0).to(device)

        fig.suptitle(f'U-Net Generation - Epoch {epoch+1} - Target {target}', fontsize=12)
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
    fig.savefig(f"{saveLocation}/epoch{epoch+1}.png")
    # plt.show()
    # plt.close()

def noisify(originalImage, generatedImage, epsilon): 
    return gtsrb_utils.just_normalize(generatedImage)
    # normalized = nn.functional.tanh(gtsrb_utils.just_normalize(generatedImage))
    # return originalImage + epsilon * normalized

def trainUNet(epochs: int, starting_epoch: int = 0, beta: int=1, trainCycleName="DefaultCycle"):

    # Hyperparameters
    learning_rate = 1e-3
    batch_size = 256
    alpha = 1e-4
    epsilon = 0.3 # Max influence the noise can have in the generated image
    # beta = 5 # How much influence image closeness has on total loss
    target = -1

    original_target = target

    # For measuring stats, measured per epoch
    train_losses = []
    validation_accs = []
    ranking_accs = []

    # Load model and set weights
    from train_copycat import load_copycat
    # compareModel = gtsrb_utils.load_pretrained().to(device)
    compareModel = load_copycat().to(device)

    model = UNet().to(device)
    model.train()

    model_path = os.path.join("data", "models", f"UNetTrain_{trainCycleName}_{starting_epoch}.pth")
    if starting_epoch > 0 and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    # Load dataset
    total_dataset = gtsrb_utils.load_gtsrb_dataset()
    print("Length:", len(total_dataset))

    # Load Datasets
    train_size = int(len(total_dataset) * 0.85)
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
        model.load_state_dict(torch.load(os.path.join("data", f"{trainCycleName}_t{original_target}_{alpha}_{beta}", f"UNetTrain_{trainCycleName}_{starting_epoch}.pth")))
        dict = pickle.load(open(f"data/{trainCycleName}_t{original_target}_{alpha}_{beta}/statistics.txt", 'rb'))
        train_losses = dict['train_loss']
        validation_accs = dict['val_accuracy']
        ranking_accs = dict['rank_accuracy']

    # Loss functions
    loss_fn = UNetLoss_baluja

    # Optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # demo(model, compareModel, epsilon, device, 0, target, f"data/{trainCycleName}_t{original_target}_{alpha}_{beta}")
    # return

    # Create a folder in which to store our data.
    dirExists = os.path.exists(f"data/{trainCycleName}_t{original_target}_{alpha}_{beta}")
    if not dirExists: os.makedirs(f"data/{trainCycleName}_t{original_target}_{alpha}_{beta}")

    # Train loop
    for t in range(epochs):
        print(f"Epoch {starting_epoch+t+1} -- Beta {beta}\n-------------------------------")
        if not model.training:
            model.train()
        if not compareModel.training:
            compareModel.train()

        if (t+1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(f"data/{trainCycleName}_t{original_target}_{alpha}_{beta}", f"UNetTrain_{trainCycleName}_{starting_epoch+t+1}.pth"))

        for batch, (X, y) in enumerate(data_loader_train):
            X, y = X.to(device), y.to(device)
            X.requires_grad = True

            gen = model(X)
            noised_gen = noisify(X, gen, epsilon)

            loss, sub_tar = loss_fn(X, noised_gen, y, compareModel, beta=beta, target=target)
            if target == -1: target = sub_tar

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"train loss: {loss:>7f}  [{current:>7d}/{len(data_loader_train.dataset):>7d}]")

        validate_acc, ranking_acc = evaluate(model, data_loader_val, compareModel, loss_fn, device, target)
        train_losses.append(loss.item())
        validation_accs.append(validate_acc.item())
        ranking_accs.append(ranking_acc.item())
        
        if (t+1) % 5 == 0:
            demo(model, compareModel, epsilon, device, t, target, f"data/{trainCycleName}_t{original_target}_{alpha}_{beta}")
    
    figFinal = plt.figure()
    plt.plot(range(epochs), train_losses, label="Train Loss", linestyle="-.")
    plt.plot(range(epochs), validation_accs, label="Validation Accuracy", linestyle="-")
    plt.plot(range(epochs), ranking_accs, label="Ranking Accuracy", linestyle="-")
    plt.legend()
    if t > 0:
        figFinal.savefig(f"data/{trainCycleName}_t{original_target}_{alpha}_{beta}/FinalGraph.png")
        dict = {}
        dict['train_loss'] = train_losses
        dict['val_accuracy'] = validation_accs
        dict['rank_accuracy'] = ranking_accs

        file1 = open(f"data/{trainCycleName}_t{original_target}_{alpha}_{beta}/statistics.txt", "wb") 
        pickle.dump(dict, file1)
        file1.close
    
    return train_losses, validation_accs, ranking_accs, f"data/{trainCycleName}_t{original_target}_{alpha}_{beta}"
    # # plt.show()
    # return

import matplotlib.patches as mpatches
import matplotlib.lines as mlines

if __name__ == "__main__":
    trainCycleName ="GMSDErrorCopycat" # "MaxSquareError" # "BasicMSE", "GMSDError"
    
    metrics = []
    colours = ["green", "red", "orange", "yellow", "brown", "black"]
    linestyles = ["dotted", "dashed"]
    betas = [0.5, 1, 5]
    for b in betas:
        train_losses, validation_accs, ranking_accs, path = trainUNet(20, 0, beta=b, trainCycleName=trainCycleName)
        metrics.append(train_losses)
        metrics.append(validation_accs)
        metrics.append(ranking_accs)
    epochs = len(train_losses)
    
    figFinal = plt.figure()
    figFinal.suptitle(f"Training Metrics on {trainCycleName}", fontsize=12)
    handles = []
    for i in range(len(betas)):
        plt.plot(range(epochs), metrics[i], linestyle=linestyles[0], color=colours[i])
        plt.plot(range(epochs), metrics[i+1], linestyle=linestyles[1], color=colours[i])
        # plt.plot(range(epochs), metrics[i+2], linestyle=linestyles[2], color=colours[i])
        handles.append(mpatches.Patch(color=colours[i], label=f'Beta={betas[i]}'))
    handles.append(mlines.Line2D([], [], color='black', linestyle=linestyles[0], label=f'Training Loss'))
    handles.append(mlines.Line2D([], [], color='black', linestyle=linestyles[1], label=f'Validation Accuracy'))
        
    plt.legend(handles=handles)
    
    figFinal.savefig(f"{path}/CompiledGraph.png")
