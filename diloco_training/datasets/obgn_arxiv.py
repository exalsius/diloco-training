from torch_geometric.datasets import OGBNArxiv
from torch_geometric.loader import DataLoader


def get_ogbn_arxiv(root="./datasets/graphs", batch_size=64):
    """Loads ogbn-arxiv dataset for GCN training."""

    dataset = OGBNArxiv(root=root)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataset, dataloader
