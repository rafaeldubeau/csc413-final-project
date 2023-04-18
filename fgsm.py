from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)



# FGSM attack code
def fgsm_attack(image, epsilon):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = image.grad.data.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
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
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, don't bother attacking, just move on
        if init_pred.item() != target.item():
            continue
        y_true.append(target.item())
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
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        y_pred.append(final_pred.item())
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    f1_score = metrics.f1_score(y_true, y_pred, average='macro')
    precision = metrics.precision_score(y_true, y_pred, average='macro')
    recall = metrics.recall_score(y_true, y_pred, average='macro')
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, f1_score, precision, recall, mcc, adv_examples


if __name__ == "__main__":
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    accuracies = []
    f1s = []
    precs = []
    recalls = []
    mccs = []
    examples = []

    import gtsrb_utils
    model = gtsrb_utils.load_pretrained()
    use_cuda = True
    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")


    total_dataset = gtsrb_utils.load_gtsrb_dataset()
    train_set, val_set = torch.utils.data.random_split(total_dataset, (22644, 3996))
    data_loader_train = DataLoader(train_set, shuffle=True)
    test_loader = DataLoader(val_set, shuffle=True)

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

