from transformers import Wav2Vec2ForCTC, Wav2Vec2Config
import torch
import torch.nn as nn


class Wav2Vec2Model(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-large-960h", ):
        super(Wav2Vec2Model, self).__init__()
        self.config = Wav2Vec2Config(
            hidden_size=128,  # Very small hidden size
            intermediate_size=512,  # Typically 4x hidden size
            num_hidden_layers=6,  # Fewer transformer layers
            num_attention_heads=4,  # Fewer attention heads
        )
        self.model = Wav2Vec2ForCTC(self.config)
        self.criterion = nn.CTCLoss()


    def forward(self, input_values, labels=None):
        outputs = self.model(input_values)
        logits = outputs.logits

        if labels is not None:
            input_lengths = torch.full(
                size=(input_values.shape[0],), 
                fill_value=logits.shape[1],  # Use logits length, NOT input_values length
                dtype=torch.long
            )
            target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)

            loss = self.criterion(logits.transpose(0, 1), labels, input_lengths, target_lengths)
            return type("Output", (object,), {"loss": loss, "logits": logits})()

        return logits

def get_wav2vec2():
    """Return ResNet model with a loss calculation interface"""
    return Wav2Vec2Model()


# Example usage
if __name__ == "__main__":
    model = Wav2Vec2Model()
    print(model)