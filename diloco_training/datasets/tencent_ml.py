import os

import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class TencentMLDataset(Dataset):
    """Tencent ML Images dataset loader for BigGAN"""

    def __init__(self, root_dir, label_csv, transform=None):
        self.root_dir = root_dir
        self.labels = pd.read_csv(label_csv)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels.iloc[idx, 1:].values.astype(float)

        return image, label


def get_tencent_ml(root="./datasets/images/tencent_ml", split="train", batch_size=64):
    """Loads Tencent ML Images dataset for BigGAN training"""

    csv_path = f"./datasets/metadata/{split}_labels.csv"
    img_path = os.path.join(root, split)

    transform = transforms.Compose(
        [
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    dataset = TencentMLDataset(
        root_dir=img_path, label_csv=csv_path, transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return dataset, dataloader
