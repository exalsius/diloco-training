import torch.nn.functional as F
from torch_geometric.nn.models import GCN  # Pre-built GCN model
import torch.nn as nn
from torch_geometric.nn.models import GCN

class GCNWithLoss(nn.Module):
    def __init__(self, in_channels=128, hidden_channels=256, out_channels=40, num_layers=2):
        super().__init__()
        self.model = GCN(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers, out_channels=out_channels)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, data):
        logits = self.model(data.x, data.edge_index)
        if data.y is not None:
            # Reshape target tensor if needed (important fix)
            if data.y.dim() > 1:
                # If y is [N, 1], convert to [N]
                y = data.y.squeeze() if data.y.dim() > 1 else data.y
            else:
                y = data.y
            loss = self.criterion(logits, y)
            return type("Output", (object,), {"loss": loss, "logits": logits})()
        return logits

def get_gcn(in_channels=128, hidden_channels=256, out_channels=40, num_layers=2):
    """Return GCN model with a loss calculation interface"""
    return GCNWithLoss(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers)


# Example usage
if __name__ == "__main__":
    model = get_gcn()
    print(model)
