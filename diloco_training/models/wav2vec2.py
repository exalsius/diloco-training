from typing import List, Optional, Union

import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC


class Wav2Vec2Output:
    """Class to hold model outputs in a structured way."""

    def __init__(
        self, loss: Optional[torch.Tensor] = None, logits: torch.Tensor = None
    ):
        self.loss = loss
        self.logits = logits


class Wav2Vec2Model(nn.Module):
    """Wav2Vec2 model wrapper with CTC loss integration."""

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-large-960h",
        hidden_size: int = 128,
        intermediate_size: int = 512,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 4,
    ):
        """
        Initialize the Wav2Vec2 model with custom configuration.

        Args:
            model_name: Pretrained model identifier (not used in current implementation)
            hidden_size: Size of hidden layers
            intermediate_size: Size of intermediate (feed-forward) layers
            num_hidden_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
        """
        super(Wav2Vec2Model, self).__init__()

        self.config = Wav2Vec2Config(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
        )
        self.model = Wav2Vec2ForCTC(self.config)
        self.criterion = nn.CTCLoss()

    def forward(
        self, input_values: torch.Tensor, labels: Optional[List[torch.Tensor]] = None
    ) -> Union[torch.Tensor, Wav2Vec2Output]:
        """
        Forward pass through the model.

        Args:
            input_values: Audio input tensor of shape (batch_size, sequence_length)
            labels: Optional list of label tensors for loss calculation

        Returns:
            Either logits tensor or Wav2Vec2Output object containing loss and logits
        """
        outputs = self.model(input_values)
        logits = outputs.logits

        if labels is not None:
            # For CTC loss, input_lengths should be the length of the logits sequence
            batch_size = input_values.shape[0]
            logits_length = logits.shape[1]

            input_lengths = torch.full(
                size=(batch_size,),
                fill_value=logits_length,
                dtype=torch.long,
                device=input_values.device,
            )

            # Ensure labels are on the same device as input
            if isinstance(labels[0], torch.Tensor):
                target_lengths = torch.tensor(
                    [label.size(0) for label in labels],
                    dtype=torch.long,
                    device=input_values.device,
                )
            else:
                target_lengths = torch.tensor(
                    [len(label) for label in labels],
                    dtype=torch.long,
                    device=input_values.device,
                )

            # CTC loss expects (time, batch, num_classes)
            loss = self.criterion(
                logits.transpose(0, 1), labels, input_lengths, target_lengths
            )

            return Wav2Vec2Output(loss=loss, logits=logits)

        return logits


def get_wav2vec2(
    hidden_size: int = 128,
    intermediate_size: int = 512,
    num_hidden_layers: int = 6,
    num_attention_heads: int = 4,
) -> Wav2Vec2Model:
    """
    Factory function to create a Wav2Vec2 model with specified parameters.

    Args:
        hidden_size: Size of hidden layers
        intermediate_size: Size of intermediate layers
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads

    Returns:
        Configured Wav2Vec2Model instance
    """
    return Wav2Vec2Model(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
    )


# Example usage
if __name__ == "__main__":
    model = get_wav2vec2()
    print(model)
