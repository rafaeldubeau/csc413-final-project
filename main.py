import os
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms as transforms
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights

import matplotlib.pyplot as plt


def test_pretrained():
    img = read_image(os.path.join("data", "puppy.jpg"))

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    preprocess = weights.transforms()

    batch = preprocess(img).unsqueeze(0)

    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")



if __name__ == "__main__":
    from train import trainUNet
    trainUNet(10, 0)

    test_pretrained()