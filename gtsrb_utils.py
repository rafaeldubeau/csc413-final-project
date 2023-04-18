import os
import random

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as transforms

import matplotlib.pyplot as plt

from gtsrb.model import Net



# From https://github.com/magnusja/GTSRB-caffe-model/blob/master/labeller/main.py
label_map = {
    0: '20_speed',
    1: '30_speed',
    2: '50_speed',
    3: '60_speed',
    4: '70_speed',
    5: '80_speed',
    6: '80_lifted',
    7: '100_speed',
    8: '120_speed',
    9: 'no_overtaking_general',
    10: 'no_overtaking_trucks',
    11: 'right_of_way_crossing',
    12: 'right_of_way_general',
    13: 'give_way',
    14: 'stop',
    15: 'no_way_general',
    16: 'no_way_trucks',
    17: 'no_way_one_way',
    18: 'attention_general',
    19: 'attention_left_turn',
    20: 'attention_right_turn',
    21: 'attention_curvy',
    22: 'attention_bumpers',
    23: 'attention_slippery',
    24: 'attention_bottleneck',
    25: 'attention_construction',
    26: 'attention_traffic_light',
    27: 'attention_pedestrian',
    28: 'attention_children',
    29: 'attention_bikes',
    30: 'attention_snowflake',
    31: 'attention_deer',
    32: 'lifted_general',
    33: 'turn_right',
    34: 'turn_left',
    35: 'turn_straight',
    36: 'turn_straight_right',
    37: 'turn_straight_left',
    38: 'turn_right_down',
    39: 'turn_left_down',
    40: 'turn_circle',
    41: 'lifted_no_overtaking_general',
    42: 'lifted_no_overtaking_trucks'
}

# Resize all images to 32 * 32 and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from the training set
data_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
    ])


def load_pretrained() -> Net:
    model = Net()
    weights = torch.load(os.path.join("gtsrb", "model", "model_40.pth"))
    model.load_state_dict(weights)

    return model


def load_gtsrb_dataloader(batch_size=64, split="train") -> DataLoader:
    path = os.path.join("data")

    dataset = torchvision.datasets.GTSRB(root=path, download=True, transform=data_transforms, split=split)
    data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    return data_loader


def load_gtsrb_dataset(split="train") -> Dataset:
    path = os.path.join("data")

    dataset = torchvision.datasets.GTSRB(root=path, download=True, transform=data_transforms, split=split)

    return dataset


def demo_pretrained():
    path = os.path.join("data")
    dataset = torchvision.datasets.GTSRB(root=path, download=True, split="train")
    pretrained = load_pretrained()

    fig = plt.figure()
    for i in range(9):
        img, label = dataset[random.randint(0, len(dataset)-1)]
        
        pred_label = pretrained(data_transforms(img).unsqueeze(0)).argmax(dim=-1).squeeze()

        fig.add_subplot(3, 3, i+1)
        plt.title(label_map[pred_label.item()])
        plt.axis("off")
        plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    demo_pretrained()