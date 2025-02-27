import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader, Dataset


class HuggingFaceDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = self.transform(item["image"])  # Apply transformation here
        label = torch.tensor(item["label"], dtype=torch.long)  # Convert label to tensor
        return {"image": image, "label": label}


def get_imagenet(world_size, local_rank, per_device_train_batch_size, split="train"):
    transform = transforms.Compose(
        [
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = load_dataset("zh-plus/tiny-imagenet", split=split).select(range(100))

    # Split dataset across nodes
    dataset = split_dataset_by_node(dataset, rank=local_rank, world_size=world_size)

    # Convert dataset to PyTorch Dataset
    dataset = HuggingFaceDataset(dataset, transform)

    # DataLoader
    dataloader = DataLoader(
        dataset, batch_size=per_device_train_batch_size, shuffle=True
    )

    return dataset, dataloader


if __name__ == "__main__":
    dataset, dataloader = get_imagenet(
        world_size=1, local_rank=0, per_device_train_batch_size=32
    )
    print("Dataset length:", len(dataset))
    print("Dataloader length:", len(dataloader))
    print("Dataloader batch size:", dataloader.batch_size)

    for step, batch in enumerate(dataloader):
        print(batch["image"].shape)  # Should now work without errors
        break
