import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


def get_gcn(in_channels=128, hidden_channels=256, out_channels=40):
    """Return a GCN model"""
    return GCN(in_channels, hidden_channels, out_channels)


# Example usage
if __name__ == "__main__":
    model = get_gcn()
    print(model)
