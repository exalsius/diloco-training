import torch
from pytorch_pretrained_biggan import one_hot_from_int, truncated_noise_sample
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class BigGANDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self._distributed = False

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        noise = torch.tensor(
            truncated_noise_sample(truncation=0.4, batch_size=1)
        ).squeeze(0)
        class_vector = torch.tensor(one_hot_from_int([207], batch_size=1)).squeeze(
            0
        )  # Example class vector for "golden retriever"
        real_images = torch.randn(1, 3, 256, 256)  # Placeholder for real images
        return {"noise": noise, "class_vector": class_vector, "real_image": real_images}


def get_tencent_ml(world_size, local_rank, per_device_train_batch_size, split="train"):
    """Loads Tencent ML Images dataset for BigGAN training"""

    dataset = BigGANDataset(num_samples=1000)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank)
    dataloader = DataLoader(
        dataset, batch_size=per_device_train_batch_size, sampler=sampler
    )

    return dataset, dataloader


if __name__ == "__main__":
    dataset, dataloader = get_tencent_ml(1, 0, 4)
    print(f"Loaded {len(dataset)} samples.")
    for i, batch in enumerate(dataloader):
        print(
            f"Batch {i+1}: {batch["noise"].shape, batch["class_vector"].shape, batch["real_image"].shape}"
        )
        if i >= 2:  # Print only the first 3 batches
            break
