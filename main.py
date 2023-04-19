from train import trainUNet

from gtsrb_utils import load_gtsrb_dataset


if __name__ == "__main__":
    # trainUNet(10, 0)

    dataset = load_gtsrb_dataset(split="train", normalize=False)
    img, label = dataset[0]
    print(f"({img.min(), img.max()})")
