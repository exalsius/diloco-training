from ogb.nodeproppred import PygNodePropPredDataset
from torch.serialization import safe_globals
from torch.utils.data import IterableDataset
from torch.utils.data.distributed import DistributedSampler

# Allow PyG Data object to be safely unpickled
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import ToUndirected
from torchdata.stateful_dataloader import StatefulDataLoader


class OGBNArxivIterable(IterableDataset):
    """
    Iterable dataset that internally instantiates a NeighborLoader each epoch.
    When repeat=True it loops forever (for continuous training).
    """

    def __init__(
        self,
        data,
        input_nodes,
        num_neighbors,
        batch_size,
        sampler=None,
        num_workers=4,
        repeat=True,
    ):
        self.data = data
        self.input_nodes = input_nodes
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.sampler = sampler
        self.num_workers = num_workers
        self.repeat = repeat

    def _make_loader(self):
        return NeighborLoader(
            self.data,
            input_nodes=self.input_nodes,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            shuffle=False,  # Shuffling handled by DistributedSampler if desired
            sampler=self.sampler,
            num_workers=self.num_workers,
        )

    def __iter__(self):
        while True:
            loader = self._make_loader()
            for batch in loader:
                yield {"data": batch}
            if not self.repeat:
                break


def get_ogbn_arxiv(
    world_size,
    local_rank,
    per_device_train_batch_size,
    split="train",
    cache_dir="/workspace/datasets/obgn_arxiv",
):
    """
    Loads ogbn-arxiv dataset returning StatefulDataLoader for train & val.

    Args:
        world_size: Number of processes in distributed training
        local_rank: Rank of current process
        per_device_train_batch_size: Batch size per device
        split: Dataset split to use
        cache_dir: Directory for caching datasets. If None, uses default location

    Returns:
        Tuple of train and validation dataloaders
    """
    # Use cache_dir if provided, otherwise use None for default
    root_dir = "/workspace/datasets/obgn_arxiv"  # str(cache_dir) if cache_dir else None
    print(f"root: {root_dir}")
    with safe_globals([Data, HeteroData, DataEdgeAttr, DataTensorAttr, GlobalStorage]):
        dataset = PygNodePropPredDataset(
            name="ogbn-arxiv", transform=ToUndirected(), root=root_dir
        )
    data = dataset[0]

    # Normalize features
    data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"]

    # Distributed sampler for node indices (affects NeighborLoader sampling order)
    train_sampler = DistributedSampler(
        train_idx, num_replicas=world_size, rank=local_rank, shuffle=True
    )

    # Training iterable (infinite / repeat)
    train_dataset = OGBNArxivIterable(
        data=data,
        input_nodes=train_idx,
        num_neighbors=[15, 10],
        batch_size=per_device_train_batch_size,
        sampler=train_sampler,
        num_workers=4,
        repeat=True,
    )

    # Validation iterable (single pass)
    val_dataset = OGBNArxivIterable(
        data=data,
        input_nodes=split_idx["valid"],
        num_neighbors=[15, 10],
        batch_size=per_device_train_batch_size,
        sampler=None,
        num_workers=2,
        repeat=False,
    )

    # StatefulDataLoader: batch_size=None -> yield elements as produced (already dicts)
    train_loader = StatefulDataLoader(train_dataset, batch_size=None)
    val_loader = StatefulDataLoader(val_dataset, batch_size=None)

    return train_loader, val_loader


if __name__ == "__main__":
    world_size = 1
    local_rank = 0
    per_device_train_batch_size = 2048

    train_loader, val_loader = get_ogbn_arxiv(
        world_size, local_rank, per_device_train_batch_size
    )

    # print first 3 training batches
    for i, batch in zip(range(3), train_loader):
        b = batch["data"]
        print(
            f"Train Batch {i+1}: x={b.x.shape}, y={b.y.shape}, batch_size={b.batch_size}"
        )

    # print first 2 validation batches
    for i, batch in zip(range(2), val_loader):
        b = batch["data"]
        print(
            f"Val Batch {i+1}: x={b.x.shape}, y={b.y.shape}, batch_size={b.batch_size}"
        )
