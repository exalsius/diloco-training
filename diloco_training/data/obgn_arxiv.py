from torch_geometric.transforms import ToUndirected
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import NeighborLoader
from torch.utils.data.distributed import DistributedSampler


def infinite_dataloader(dataloader):
    """Wrap a dataloader to yield batches indefinitely."""
    while True:
        for batch in dataloader:
            yield batch


def get_ogbn_arxiv(world_size, local_rank, per_device_train_batch_size, split="train"):
    """Loads ogbn-arxiv dataset for GCN training."""
    # Load and preprocess data
    dataset = PygNodePropPredDataset(
        name="ogbn-arxiv", transform=ToUndirected(), root="/workspace/datasets/obgn"
    )
    data = dataset[0]

    # Normalize features
    data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)

    split_idx = dataset.get_idx_split()
    input_nodes = split_idx["train"]
    # Distributed sampler
    train_sampler = DistributedSampler(
        input_nodes, num_replicas=world_size, rank=local_rank, shuffle=True
    )

    # NeighborLoader for mini-batch training
    train_loader = NeighborLoader(
        data,
        input_nodes=input_nodes,
        num_neighbors=[15, 10],
        batch_size=2048,
        shuffle=False,
        sampler=train_sampler,
        num_workers=4,
    )

    val_loader = NeighborLoader(
        data,
        input_nodes=split_idx["valid"],
        num_neighbors=[15, 10],
        batch_size=2048,
        shuffle=False,
        num_workers=2,
    )

    # Wrap the train loader to loop indefinitely
    train_loader = infinite_dataloader(train_loader)

    # Wrap the batch in a dictionary with key 'data'
    def wrap_batch(batch):
        return {"data": batch}

    # Apply the wrapper to each batch
    train_loader = map(wrap_batch, train_loader)
    val_loader = map(wrap_batch, val_loader)

    return train_loader, val_loader


if __name__ == "__main__":
    world_size = 1
    local_rank = 0
    per_device_train_batch_size = 32

    train_loader, val_loader = get_ogbn_arxiv(
        world_size, local_rank, per_device_train_batch_size
    )

    # print the first 3 batches
    for batch, i in zip(train_loader, range(3)):
        print(
            f"Batch {i+1}: {batch['data'].x.shape}, {batch['data'].y.shape}, {batch['data'].batch_size}"
        )
