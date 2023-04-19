from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import gtsrb_utils


# FGSM attack code
def fgsm_attack(image, epsilon):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = image.grad.data.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def test( model, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []
    y_true = []
    y_pred = []
    # Loop over all examples in test set
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        data.requires_grad = True
        # Forward pass the data through the model
        output = F.log_softmax(model(data), dim=-1)
        init_pred = output.max(dim=-1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, don't bother attacking, just move on
        # if init_pred.item() != target.item():
        #     continue
        y_true.extend(target.tolist())

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect ``datagrad``
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(dim=-1, keepdim=True)[1] # get the index of the max log-probability
        correct += torch.count_nonzero(torch.tensor(target.tolist()) == torch.tensor(final_pred.squeeze().tolist()))

        y_pred.extend(final_pred.tolist())

        if len(adv_examples) < 5:
            adv_ex = torch.clamp(perturbed_data[0], 0, 1)
            adv_ex = adv_ex.squeeze().detach().cpu().numpy()
            adv_ex = np.einsum('kij->ijk', adv_ex)
            adv_examples.append( (init_pred.squeeze().tolist()[0], final_pred.squeeze().tolist()[0], adv_ex) )

    f1_score = metrics.f1_score(y_true, y_pred, average='macro')
    precision = metrics.precision_score(y_true, y_pred, average='macro')
    recall = metrics.recall_score(y_true, y_pred, average='macro')
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    final_acc = correct/len(test_loader.dataset)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader.dataset), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, f1_score, precision, recall, mcc, adv_examples


if __name__ == "__main__":
    epsilons = [0, 0.0001, 0.001, 0.01, 0.05, .1, .15, .2, .25, .3]

    accuracies = []
    f1s = []
    precs = []
    recalls = []
    mccs = []
    examples = []

    use_cuda = True
    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")


    model = gtsrb_utils.load_pretrained().to(device)

    test_dataset = gtsrb_utils.load_gtsrb_dataset(split="test", normalize=False)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=512)

    # Run test for each epsilon
    for eps in epsilons:
        acc, f1, prec, recall, mcc, ex = test(model, device, test_loader, eps)
        accuracies.append(acc)
        f1s.append(f1)
        precs.append(prec)
        recalls.append(recall)
        mccs.append(mcc)
        examples.append(ex)

    plt.figure(figsize=(5, 5))
    plt.plot(epsilons, accuracies, "*-", label="Accuracy")
    plt.plot(epsilons, f1s, label="F1-Score")
    plt.plot(epsilons, precs, label="Precision")
    plt.plot(epsilons, recalls, label="Recall")
    plt.plot(epsilons, mccs, label="MCC")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Metrics Over Different Epsilons")
    plt.xlabel("Epsilon")
    plt.ylabel("Score")
    plt.legend()
    plt.show()

    # Plot several examples of adversarial samples at each epsilon
    cnt = 0
    plt.figure(figsize=(8, 10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig, adv, ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex)
    plt.tight_layout()
    plt.show()
