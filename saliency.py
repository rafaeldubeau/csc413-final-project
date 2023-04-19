import torch
from torch import nn
from torch import Tensor, LongTensor, BoolTensor

import numpy as np

from gtsrb_utils import load_pretrained, load_gtsrb_dataset, just_normalize, label_map

from matplotlib import pyplot as plt

import random


def show_img(img: Tensor):
    img = img.detach().numpy().astype(np.float32)

    if img.shape[0] == 3:
        img = img.transpose((1, 2, 0))

    plt.figure()
    plt.imshow(img)
    plt.show()


def linear_map(X: Tensor, low = 0.0, high = 1.0):
    X = X + (low - X.min())
    X = X * ((high - low) / (X.max() - X.min()))

    return X


def compute_jacobian(X: Tensor, y: Tensor, model):
    J = torch.zeros((y.size(0), X.size(0), X.size(1), X.size(2)))
    for i in range(y.size(0)):
        # Compute gradient
        model.zero_grad()
        y[i].backward(retain_graph=True)

        J[i] = X.grad.data
    
    return J


def compute_adversarial_saliency_map(J: Tensor, t: int, Gamma: BoolTensor, alt=False):
    Sigma = J.sum(dim=0) - J[t]
    if alt:
        mask = torch.logical_and(torch.logical_and(J[t] < 0, Sigma > 0), Gamma)
        S = torch.where(mask, J[t].abs() * Sigma, 0.0)
    else:
        mask = torch.logical_and(torch.logical_and(J[t] >= 0, Sigma <= 0), Gamma)
        S = torch.where(mask, J[t] * Sigma.abs(), 0.0)
    return S


# def compute_untargeted_adversarial_saliency_map(J: Tensor, y: int, Gamma: BoolTensor):
#     Sigma = J.sum(dim=0) - J[y]
#     mask = torch.logical_and(torch.logical_and(J[y] < 0, Sigma > 0), Gamma)
#     S = torch.where(mask, J[y].abs()*Sigma, 0.0)
#     return S

def compute_alt_adversarial_saliency_map(J: Tensor, t: int, Gamma: BoolTensor):
    Sigma = J.sum(dim=0) - J[t]
    mask = torch.logical_and(torch.logical_and(J[t] < 0, Sigma > 0), Gamma)
    S = torch.where(mask, J[t].abs()-Sigma)
    return S


def CraftingAlg(img, t, model, theta, upsilon=torch.inf, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    img = img.to(device)
    X = img
    X.requires_grad = True

    delta = 0
    Gamma = torch.ones_like(X, dtype=torch.bool).to(device)
    alt = theta < 0

    y = model(just_normalize(X).unsqueeze(0)).squeeze()
    guess = y.argmax(dim=-1)
    k = 0
    while guess != t and delta < upsilon and Gamma.sum() > 0:
        if k % 50 == 0 and verbose:
            print(f"{k}: {y}")
        k += 1
        # Get index of largest value in adversarial saliency map
        J = compute_jacobian(X, y, model).to(device)
        S = compute_adversarial_saliency_map(J, t, Gamma, alt=alt).to(device)
        i_max = (S==S.max()).nonzero()[0]
        # print(f"i_max: {i_max} -> {S[i_max[0], i_max[1], i_max[2]]}")

        # Remove X from gradient tree
        X.detach()
        X.requires_grad = False

        # Update X
        diff = torch.zeros_like(X, dtype=torch.float).to(device)
        diff[i_max[0], i_max[1], i_max[2]] = theta
        X = torch.clamp(X + diff, 0, 1)
        if X[i_max[0], i_max[1], i_max[2]] == 0 or X[i_max[0], i_max[1], i_max[2]] == 1:
            Gamma[i_max[0], i_max[1], i_max[2]] = False

        delta = (X - img).square().sum()

        # Reset X as leaf node
        X.requires_grad = True

        y = model(just_normalize(X).unsqueeze(0)).squeeze()
        guess = y.argmax(dim=-1)
        # print(f"guess: {guess}={y[guess]}\tt: {t}={y[t]}\t{y}")

    return X.cpu()


def CraftingAlg_untargeted(img, model, theta, allow_stacking=True, upsilon=torch.inf, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    img = img.to(device)
    X = img
    X.requires_grad = True

    delta = 0
    Gamma = torch.ones_like(X, dtype=torch.bool).to(device)
    alt = theta > 0

    y = model(just_normalize(X).unsqueeze(0)).squeeze()
    initial_guess = y.argmax(dim=-1)
    guess = initial_guess
    k = 0
    while guess == initial_guess and delta < upsilon and Gamma.sum() > 0:
        if k % 50 == 0 and verbose:
            print(f"{k} (Gamma={Gamma.sum()}): guess={y[guess.item()]}, initial_guess={y[initial_guess.item()]}")
            show_img(X.cpu())
            X = X.to(device)
        k += 1
        # Get index of largest value in adversarial saliency map
        J = compute_jacobian(X, y, model).to(device)
        S = compute_adversarial_saliency_map(J, initial_guess, Gamma, alt=alt).to(device)
        i_max = (S==S.max()).nonzero()[0]
        # print(f"i_max: {i_max} -> {S[i_max[0], i_max[1], i_max[2]]}")

        # Remove X from gradient tree
        X.detach()
        X.requires_grad = False

        # Update X
        diff = torch.zeros_like(X, dtype=torch.float).to(device)
        diff[i_max[0], i_max[1], i_max[2]] = theta
        X = torch.clamp(X + diff, 0, 1)
        if allow_stacking:
            if X[i_max[0], i_max[1], i_max[2]] == 0 or X[i_max[0], i_max[1], i_max[2]] == 1:
                Gamma[i_max[0], i_max[1], i_max[2]] = False
        else:
            Gamma[i_max[0], i_max[1], i_max[2]] = False

        delta = (X - img).abs().sum()

        # Reset X as leaf node
        X.requires_grad = True

        y = model(just_normalize(X).unsqueeze(0)).squeeze()
        guess = y.argmax(dim=-1)
        # print(f"guess: {guess}={y[guess]}\tt: {t}={y[t]}\t{y}")

    return X.cpu(), delta



def demo(img, label, theta, upsilon=torch.inf, n=1):
    model = load_pretrained()

    classes = np.random.randint(0, 43, size=n)

    # modded_img = CraftingAlg1_untargeted(img, model, 0.1, verbose=True)
    # Sometimes works: modded_img = CraftingAlg1(img, classes[i], model, 1, verbose=True)
    
    plt.figure()
    for i in range(n):

        # Additive
        modded_img, delta = CraftingAlg_untargeted(img, model, theta, upsilon=upsilon, verbose=False)
        model = model.cpu()
        with torch.no_grad():
            y = model(just_normalize(modded_img).unsqueeze(0)).squeeze()
            guess = y.argmax(dim=-1)

        plt.subplot(2*n,2,2*i+1)
        modded_img = modded_img.detach().numpy().astype(np.float32)
        if modded_img.shape[0] == 3:
            modded_img = modded_img.transpose((1, 2, 0))
        plt.imshow(modded_img)
        plt.title(f"{label_map[label]}\n{label_map[guess.item()]} - {delta}")
        plt.axis("off")

        print(f"Done with Additive ({'positive' if delta < upsilon else 'negative'})")

        # Reductive
        modded_img, delta = CraftingAlg_untargeted(img, model, -theta, upsilon=upsilon, verbose=False)
        model = model.cpu()
        with torch.no_grad():
            y = model(just_normalize(modded_img).unsqueeze(0)).squeeze()
            guess = y.argmax(dim=-1)
        
        plt.subplot(2*n,2,2*i+2)
        modded_img = modded_img.detach().numpy().astype(np.float32)
        if modded_img.shape[0] == 3:
            modded_img = modded_img.transpose((1, 2, 0))
        plt.imshow(modded_img)
        plt.title(f"{label_map[label]}\n{label_map[guess.item()]} - {delta}")
        plt.axis("off")
    plt.show()



if __name__ == "__main__":
    
    model = load_pretrained()

    dataset = load_gtsrb_dataset(split="test", normalize=False)
    img, label = dataset[0]
    # img = torch.zeros_like(img)

    print(f"{label}: {label_map[label]}")
    show_img(img)
    
    demo(img, label, 0.3, upsilon=50, n=1)



    # img = torch.zeros((3, 32, 32))
    # demo(img)



    