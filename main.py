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
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True)

    # train_features, train_labels = next(iter(data_loader))
    # while (train_features is not None):
    #     train_features, train_labels = next(iter(data_loader))
    #     print(f"Feature batch shape: {train_features.size()}")
    #     print(f"Labels batch shape: {train_labels.size()}")
    #     img = train_features[0].squeeze()
    #     label = train_labels[0]
    #     plt.imshow(img[0], cmap="gray")
    #     plt.show()
    #     print(f"Label: {label}")

    weights = torch.load('pretrained/gtsrb-pytorch-master/model/model_40.pth')
    from networks import Net
    model = Net()
    model.eval()
    # print(model.bn1.weight.data)
    print(list(weights.items())[0])
    # model.bn1.weight.data = weights
    # print(model.bn1.weight.data)

    # model = GTSRB(weights=weights)
    # # print(model)
    # model.eval()

    # model.load_state_dict(weights)


if __name__ == "__main__":
    # test_pretrained()
    external_test()