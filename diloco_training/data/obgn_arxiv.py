import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data.data import (  # Import the required class
    DataEdgeAttr,
    DataTensorAttr,
    GlobalStorage,
)
from torch_geometric.loader import NeighborLoader

torch.serialization.add_safe_globals([DataEdgeAttr])
torch.serialization.add_safe_globals([DataTensorAttr])
torch.serialization.add_safe_globals([GlobalStorage])


def get_ogbn_arxiv(world_size, local_rank, per_device_train_batch_size, split="train"):
    """Loads ogbn-arxiv dataset for GCN training."""
    dataset = PygNodePropPredDataset(name="ogbn-arxiv")
    data = dataset[0]  # Get the single graph object

    # Get train/val/test masks from split_idx
    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"]
    split_idx["valid"]
    split_idx["test"]
    # Split the train_idx for each process
    train_idx = train_idx.split(train_idx.size(0) // world_size)[local_rank]

    # Set up NeighborLoader
    dataloader = NeighborLoader(
        data,
        num_neighbors=[10, 10],
        batch_size=per_device_train_batch_size,
        input_nodes=train_idx,  # Correct way to specify training nodes
    )

    # Wrap the batch in a dictionary with key 'data'
    def wrap_batch(batch):
        return {"data": batch}

    # Apply the wrapper to each batch
    dataloader = map(wrap_batch, dataloader)

    return dataset, dataloader


if __name__ == "__main__":
    world_size = 1
    local_rank = 0
    per_device_train_batch_size = 32

    dataset, dataloader = get_ogbn_arxiv(
        world_size, local_rank, per_device_train_batch_size
    )

    for batch in dataloader:
        print(batch)
        break
