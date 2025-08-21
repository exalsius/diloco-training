from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch_geometric.nn.models import GCN


@dataclass
class GCNOutput:
    """Structured output from GCN model."""

    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


class GCNWithLoss(nn.Module):
    """Graph Convolutional Network with integrated loss calculation."""

    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 1024,  # Increased hidden channels
        out_channels: int = 40,
        num_layers: int = 4,  # Increased number of layers
        dropout: float = 0.0,
    ):
        """
        Initialize GCN model with loss function.

        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output classes
            num_layers: Number of GCN layers
            dropout: Dropout probability
        """
        super().__init__()
        self.model = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            dropout=dropout,
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, data) -> Union[torch.Tensor, GCNOutput]:
        """
        Forward pass through the GCN model.

        Args:
            data: PyG data object containing x, edge_index, and optionally y

        Returns:
            GCNOutput with logits and loss if labels are provided, otherwise just logits
        """
        logits = self.model(data.x, data.edge_index)

        if data.y is not None:
            # Ensure target has correct shape [N]
            y = data.y.squeeze() if data.y.dim() > 1 else data.y
            loss = self.criterion(logits, y)
            return GCNOutput(logits=logits, loss=loss)

        return GCNOutput(logits=logits)

    def print_num_parameters(self) -> None:
        """Print the total number of trainable parameters in the model."""
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {num_params}")


def get_gcn(
    in_channels: int = 128,
    hidden_channels: int = 1024,  # Increased hidden channels
    out_channels: int = 40,
    num_layers: int = 4,  # Increased number of layers
    dropout: float = 0.0,
) -> Tuple[None, GCNWithLoss]:
    """
    Factory function to create a GCN model with loss calculation.

    Args:
        in_channels: Number of input features
        hidden_channels: Number of hidden features
        out_channels: Number of output classes
        num_layers: Number of GCN layers
        dropout: Dropout probability

    Returns:
        Configured GCNWithLoss model
    """
    return None, GCNWithLoss(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        dropout=dropout,
    )


# Example usage
if __name__ == "__main__":
    _, model = get_gcn()  # Updated parameters
    model.print_num_parameters()
    print(model)
