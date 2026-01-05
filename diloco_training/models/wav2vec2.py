from pathlib import Path
from typing import Optional

from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Processor

from diloco_training.data.librispeech import get_librispeech
from diloco_training.utils.hf_download import set_hf_timeout


def get_wav2vec2(
    hidden_size: int = 128,
    intermediate_size: int = 512,
    num_hidden_layers: int = 6,
    num_attention_heads: int = 4,
    cache_dir: Optional[Path] = None,
):
    """
    Factory function to create a Wav2Vec2 model with specified parameters.

    Args:
        hidden_size: Size of hidden layers
        intermediate_size: Size of intermediate layers
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        cache_dir: Directory for caching models. If None, uses HuggingFace default

    Returns:
        Configured Wav2Vec2Model instance
    """
    # Set timeout for HuggingFace Hub downloads
    set_hf_timeout()

    # Load processor and config from pretrained model
    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base", cache_dir=cache_dir
    )
    config = Wav2Vec2Config.from_pretrained(
        "facebook/wav2vec2-base",
        cache_dir=cache_dir,
    )

    # Set CTC-specific config
    config.ctc_loss_reduction = "mean"
    config.pad_token_id = processor.tokenizer.pad_token_id
    config.ctc_zero_infinity = True

    # Create model from config with random weights (training from scratch)
    model = Wav2Vec2ForCTC(config)

    return config, model


if __name__ == "__main__":
    train_loader, val_loader = get_librispeech(1, 0, 16, "train.clean.100")
    config, model = get_wav2vec2()
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")
