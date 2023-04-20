import os

from tqdm import tqdm

import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

from torch import nn
from torch.optim import Adam

from networks import ConvClassifier

from gtsrb_utils import load_gtsrb_dataloader, load_pretrained, data_transforms


class InferrenceDataset(Dataset):
    def __init__(self, split="train"):
        data_file = os.path.join("data", f"inferred_labels_{split}.pt")

        # Load GTSRB Dataset
        path = os.path.join("data")
        self.gtsrb = torchvision.datasets.GTSRB(root=path, download=True, transform=data_transforms, split=split)


        if os.path.exists(data_file):
            self.probs = torch.load(data_file).long()
        else:
            print(f"Generating labels for {split} dataset from pretrained model...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            batch_size = 64
            gtsrb_loader = DataLoader(self.gtsrb, shuffle=False, batch_size=batch_size)

            # Load pretrained model
            pretrained = load_pretrained().to(device)
            pretrained.eval()

            # Run the pretrained model on the entire dataset
            self.probs = torch.zeros(len(self.gtsrb)).long()
            self.probs.requires_grad = False
            for batch_num, (imgs, _) in tqdm(enumerate(gtsrb_loader), total=len(gtsrb_loader)):
                imgs = imgs.to(device)
                pred_probs = pretrained(imgs).detach().cpu()

                self.probs[batch_size * batch_num: batch_size * (batch_num + 1)] = pred_probs.argmax(dim=-1)

            # Save the resulting labels
            torch.save(self.probs, data_file)

    def __len__(self):
        return len(self.gtsrb)

    def __getitem__(self, idx):
        return self.gtsrb[idx][0], self.probs[idx]




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

        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"train loss: {loss:>7f}  [{current:>7d}/{len(data.dataset):>7d}]")


def evaluate(model, data, loss_fn, device):
    if model.training:
        model.eval()

    with torch.no_grad():
        loss = 0
        acc = 0
        for batch, (X, y) in enumerate(data):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y).item()

            acc += torch.count_nonzero(y == pred.argmax(dim=-1))

    loss = loss / (len(data.dataset) / batch_size)
    acc = acc / len(data.dataset)

    print(f"Val Loss: {loss}, Val Accuracy: {acc}")


def load_copycat(epoch=10) -> ConvClassifier:
    path = os.path.join("data", f"copycat_{epoch}.pth")
    copycat =  ConvClassifier(3, 43)
    if torch.cuda.is_available():
        copycat.load_state_dict(torch.load(path))
    else:
        copycat.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    copycat = copycat.eval()

    return copycat


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    epochs = 5
    starting_epoch = 5
    learning_rate = 1e-3
    batch_size = 128
    weight_decay = 1e-2

    # Load Dataset

    train_dataset = InferrenceDataset()
    train_size = int(len(train_dataset) * 0.85)
    val_size = int(len(train_dataset) * 0.15)
    train_set, val_set = torch.utils.data.random_split(train_dataset, (train_size, val_size))
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # Initialize Model
    model = ConvClassifier(3, 43).to(device)
    if starting_epoch > 0:
        model.load_state_dict(torch.load(os.path.join("data", f"copycat_{starting_epoch}.pth")))
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Loss Function
    loss_fn = nn.CrossEntropyLoss()

    for e in range(epochs):
        print(f"Epoch {starting_epoch+e+1}\n-------------------------------")
        train_epoch(model, train_dataloader, optimizer, loss_fn, device)
        evaluate(model, val_dataloader, loss_fn, device)

        if (e+1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join("data", f"copycat_{starting_epoch+e+1}.pth"))





