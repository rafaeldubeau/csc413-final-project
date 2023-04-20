import os
from tqdm import tqdm

import numpy as np

import torch
from torch import Tensor
from sklearn.metrics import f1_score

from torch.utils.data import DataLoader, RandomSampler, random_split
from torch import nn

from train_copycat import load_copycat
from gtsrb_utils import load_gtsrb_dataloader, load_gtsrb_dataset, load_pretrained, just_normalize

from fgsm import fgsm
from networks import UNet
from saliency import CraftingAlg_untargeted


def get_batch_metrics(original_imgs: Tensor, original_labels: Tensor, X: Tensor, y: Tensor):

    # Compute accuracy
    accuracy = torch.count_nonzero(y == original_labels).item()

    original_imgs = original_imgs.flatten(start_dim=1)
    X = X.flatten(start_dim=1)
    diff = X - original_imgs
    l2 = diff.square().sum(dim=-1).sqrt()
    # Compute avg L2 difference
    avg_l2 = l2.sum()
    # Compute avg l2 difference for successful adversarial imgs
    succ = y != original_labels
    avg_l2_succ = l2[succ].sum()
    num_succ = torch.count_nonzero(succ).item()

    return accuracy, avg_l2, avg_l2_succ, num_succ


def evaluate_method(method, dataloader: DataLoader, use_copycat: bool = False):
    accuracy = 0
    avg_l2 = 0
    avg_l2_succ = 0
    num_succ = 0

    label_all = None
    y_all = None

    no_succ = 0

    classifier = load_pretrained()
    if use_copycat:
        adversary = load_copycat()
    else:
        adversary = classifier

    for batch, (imgs, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
        X = method(imgs, labels, adversary).detach().cpu() # X = batch of adversarial images
        y = classifier(X).detach().cpu().argmax(dim=-1) # y = batch of classifications of adversarial images

        metrics = get_batch_metrics(imgs, labels, X, y)

        if label_all is None:
            label_all = labels
        else:
            label_all = torch.cat((label_all, labels), dim=0)

        if y_all is None:
            y_all = labels
        else:
            y_all = torch.cat((y_all, y), dim=0)


        accuracy += metrics[0]
        avg_l2 += metrics[1]
        avg_l2_succ += metrics[2]
        num_succ += metrics[3]

    accuracy = accuracy / len(dataloader.dataset)
    avg_l2 = avg_l2 / len(dataloader.dataset)
    if num_succ > 0:
        avg_l2_succ = avg_l2_succ / num_succ
    else:
        avg_l2_succ = float("nan")

    f1 = f1_score(label_all, y_all, average='macro')

    return accuracy, f1, avg_l2, avg_l2_succ


def evaluate_fgsm(dataloader: DataLoader, use_copycat: bool = False):
    def fgsm_method(epsilon: float):
        return lambda imgs, labels, classifier : just_normalize(fgsm(imgs, labels, classifier, epsilon))

    eps_list = [0.01, 0.05, 0.1, 0.2, 0.3]
    for eps in eps_list:
        print(f"Testing epsilon = {eps}")
        method = fgsm_method(eps)
        accuracy, f1_score, avg_l2, avg_l2_succ = evaluate_method(method, dataloader, use_copycat=use_copycat)
        print(f"eps: {eps}  -  accuracy: {accuracy}, f1_score: {f1_score}, avg_l2: {avg_l2}, avg_l2_succ: {avg_l2_succ}")


def evaluate_Unet(dataloader: DataLoader, use_copycat: bool = False):
    def Unet_method(Unet: UNet):
        return lambda imgs, labels, classifier : Unet(imgs)

    image_methods = ["BasicMSE", "GMSDError", "MaxSquareError"]
    copycat_str = ["", "Copycat"]
    betas = ["0.5", "1", "5"]

    path_list = []
    for im in image_methods:
        for cs in copycat_str:
            for b in betas:
                epoch = "30" if cs == "" else "20"
                path_list.append(
                    os.path.join("data", f"{im}{cs}_t-1_0.0001_{b}", f"UnetTrain_{im}{cs}_{epoch}.pth")
                    )

    for path in path_list:
        print(f"Test model at {path}")
        Unet = UNet().eval()
        Unet.load_state_dict(torch.load(path))
        method = Unet_method(Unet)
        accuracy, f1_score, avg_l2, avg_l2_succ = evaluate_method(method, dataloader, use_copycat=False)
        print(f"model: {path}  -  accuracy: {accuracy}, f1_score: {f1_score}, avg_l2: {avg_l2}, avg_l2_succ: {avg_l2_succ}")


def evaluate_crafting(dataloader: DataLoader, use_copycat: bool = False):
    assert dataloader.batch_size == 1 # Crafting Algorithm is not batched

    def crafting_method(theta, upsilon, allow_stacking):
        return lambda imgs, labels, classifier : just_normalize(
                CraftingAlg_untargeted(imgs.squeeze(), classifier, theta, allow_stacking, upsilon)[0]
            ).unsqueeze(0)

    theta_list = [0.5]
    upsilon_list = [10]
    stacking_list = [True]

    for theta, upsilon, allow_stacking in zip(theta_list, upsilon_list, stacking_list):
        print(f"Testing with theta={theta}, upsilon={upsilon}, allow_stacking={allow_stacking}")
        method = crafting_method(theta, upsilon, allow_stacking)
        accuracy, f1_score, avg_l2, avg_l2_succ = evaluate_method(method, dataloader, use_copycat=use_copycat)
        print(f"Crafting ({theta}, {upsilon}, {allow_stacking})  -  accuracy: {accuracy}, f1_score: {f1_score}, avg_l2: {avg_l2}, avg_l2_succ: {avg_l2_succ}")


def final_fgsm_eval():
    dataset = load_gtsrb_dataset(split="test", normalize=False)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    print("---------------Using Pretrained Model------")
    evaluate_fgsm(dataloader, use_copycat=False)
    print("---------------Using Copycat---------------")
    evaluate_fgsm(dataloader, use_copycat=True)


def final_Unet_eval():
    dataset = load_gtsrb_dataset(split="test", normalize=True)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    evaluate_Unet(dataloader, use_copycat=False)


if __name__ == "__main__":
    final_fgsm_eval()
