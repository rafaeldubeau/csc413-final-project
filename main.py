import os
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as transforms
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights


def test_pretrained():
    img = read_image(os.path.join("data", "puppy.jpg"))

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    preprocess = weights.transforms()

    batch = preprocess(img).unsqueeze(0)

    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    for id in class_ids:
        item = id.item()
        score = prediction[item].item()
        category_name = weights.meta["categories"][class_id]
        print(f"{category_name}: {100 * score:.1f}%")
    # score = prediction[class_id].item()
    # category_name = weights.meta["categories"][class_id]
    # print(f"{category_name}: {100 * score:.1f}%")

def external_test():
    dataset = torchvision.datasets.GTSRB(root= "./model", download=True, transform=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    print(dataset)
    weights = torch.load('pretrained/gtsrb-pytorch-master/model/model_40.pth')

    model = GTSRB(weights=weights)
    # print(model)
    model.eval()

    model.load_state_dict(weights)


if __name__ == "__main__":
    # test_pretrained()
    external_test()