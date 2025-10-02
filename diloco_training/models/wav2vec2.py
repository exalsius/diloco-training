from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from diloco_training.data.librispeech import get_librispeech


def get_wav2vec2(
    hidden_size: int = 128,
    intermediate_size: int = 512,
    num_hidden_layers: int = 6,
    num_attention_heads: int = 4,
):
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
    # Load model config (same as pretrained one)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    config = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base",
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    ).config

    # Reset model with random weights
    model_class = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base",
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    return config, model_class


if __name__ == "__main__":
    train_loader, val_loader = get_librispeech(1, 0, 16, "train.clean.100")
    config, model = get_wav2vec2()
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")
