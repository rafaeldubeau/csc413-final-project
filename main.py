import torch

from tqdm import tqdm

from gtsrb_utils import load_gtsrb_dataset, load_gtsrb_dataloader, load_pretrained
from train_copycat import load_copycat


if __name__ == "__main__":

    dataloader = load_gtsrb_dataloader(split="test")
    model = load_copycat().to("cuda")
    acc = 0
    for X, y in tqdm(dataloader, total=len(dataloader)):
        X, y = X.to("cuda"), y.to("cuda")
        pred = model(X).argmax(dim=-1)

        acc += torch.count_nonzero(y == pred)
    acc = acc / len(dataloader.dataset)

    print(f"accuracy: {acc.item()}")
