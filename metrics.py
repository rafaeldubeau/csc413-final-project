import torch
from torch import Tensor
from sklearn import metrics

from torch.utils.data import DataLoader
from torch import nn

from train_copycat import load_copycat
from gtsrb_utils import load_gtsrb_dataloader, load_pretrained
from tqdm import tqdm

from fgsm import fgsm


def get_batch_metrics(original_imgs: Tensor, original_labels: Tensor, X: Tensor, y: Tensor):
    
    # Compute accuracy 
    accuracy = (torch.count_nonzero(y == original_labels) / original_labels.size(0)).item()

    # Compute F1-score
    f1_score = metrics.f1_score(original_labels, y, average='macro')
    
    original_imgs = original_imgs.flatten(start_dim=1)
    X = X.flatten(start_dim=1)
    diff = X - original_imgs
    l2 = diff.square().sum(dim=-1).sqrt()
    # Compute avg L2 difference
    avg_l2 = l2.mean()
    # Compute avg l2 difference for successful adversarial imgs
    if (y != original_labels).sum() > 1:
        avg_l2_succ = l2[y != original_labels].mean()
    else:
        avg_l2_succ = 0

    return accuracy, f1_score, avg_l2, avg_l2_succ


def evaluate_method(method, dataloader: DataLoader, use_copycat: bool = False):
    accuracy = 0
    f1_score = 0
    avg_l2 = 0
    avg_l2_succ = 0

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

        accuracy += metrics[0]
        f1_score += metrics[1]
        avg_l2 += metrics[2]
        avg_l2_succ += metrics[3]
        if metrics[3] == 0:
            no_succ += 1
    
    accuracy /= len(dataloader)
    f1_score /= len(dataloader)
    avg_l2 /= len(dataloader)
    avg_l2_succ /= (len(dataloader) - no_succ)

    return accuracy, f1_score, avg_l2, avg_l2_succ


def evaluate_fgsm(dataloader: DataLoader, use_copycat: bool = False):
    def fgsm_method(epsilon: float):
        return lambda imgs, labels, classifier : fgsm(imgs, labels, classifier, epsilon)
    
    eps_list = [0.05, 0.1, 0.2, 0.3]
    for eps in eps_list:
        print(f"Testing epsilon = {eps}")
        method = fgsm_method(eps)
        accuracy, f1_score, avg_l2, avg_l2_succ = evaluate_method(method, dataloader, use_copycat=use_copycat)
        print(f"eps: {eps}  -  accuracy: {accuracy}, f1_score: {f1_score}, avg_l2: {avg_l2}, avg_l2_succ: {avg_l2_succ}")


if __name__ == "__main__":
    dataloader = load_gtsrb_dataloader(split="test")

    evaluate_fgsm(dataloader, use_copycat=True)
