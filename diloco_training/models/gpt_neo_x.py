"""
Llama model implementation module.

This module provides functionality to create and configure Llama models
for causal language modeling tasks.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from diloco_training.utils.hf_download import set_hf_timeout

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "EleutherAI/gpt-neox-20b"
DEFAULT_CONFIG: Dict[str, Any] = {
    "hidden_size": 896,
    "num_hidden_layers": 3,
    "num_attention_heads": 16,
    "intermediate_size": 896,
    "torch_dtype": "bfloat16",
    "attn_implementation": "flash_attention_2",
}
TINY_LLAMA_CONFIG: Dict[str, Any] = {
    "vocab_size": 100,
    "hidden_size": 8,
    "max_position_embeddings": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 1,
    "intermediate_size": 32,
}


def get_tiny_llama(
    cache_dir: Optional[Path] = None,
) -> Tuple[GPTNeoXConfig, GPTNeoXForCausalLM]:
    """Returns a tiny Llama model suitable for testing purposes.

    This is a convenience function that creates a minimal Llama model
    with a small configuration, making it useful for testing and development.

    Args:
        cache_dir: Directory for caching models (not used for models created from config)
    """
    config = GPTNeoXConfig(**TINY_LLAMA_CONFIG)
    return config, GPTNeoXForCausalLM(config)


def get_cpu_llama(
    cache_dir: Optional[Path] = None,
) -> Tuple[GPTNeoXConfig, GPTNeoXForCausalLM]:
    """Returns a Llama model with default configuration for CPU usage.

    This is a convenience function that creates a Llama model
    with the default configuration, making it suitable for CPU-based applications.

    Args:
        cache_dir: Directory for caching models (not used for models created from config)
    """
    config = GPTNeoXConfig(**DEFAULT_CONFIG)
    return config, GPTNeoXForCausalLM(config)


def get_gpt_neo_x(
    model_name: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[Path] = None,
) -> Tuple[Optional[GPTNeoXConfig], GPTNeoXForCausalLM]:
    """
    Create and return a Llama model for causal language modeling.

    Args:
        model_name: Name or path of a pretrained model to load.
                   If None, creates a model with custom configuration.
        config_overrides: Dictionary of parameters to override in the default configuration.
        cache_dir: Directory for caching models. If None, uses HuggingFace default.

    Returns:
        A LlamaForCausalLM model instance.

    Raises:
        ValueError: If configuration parameters are invalid.
    """
    if model_name and config_overrides:
        logger.warning(
            "Both model_name and config_overrides provided. "
            "Using custom configuration instead of pretrained model."
        )

    if model_name and not config_overrides:
        try:
            # Set timeout for HuggingFace Hub downloads
            set_hf_timeout()
            logger.info(f"Loading pretrained model: {model_name}")
            return None, GPTNeoXForCausalLM.from_pretrained(
                model_name, cache_dir=cache_dir
            )
        except Exception as e:
            logger.error(f"Failed to load pretrained model: {e}")
            raise

    # Use default config with any overrides
    config_params = DEFAULT_CONFIG.copy()
    if config_overrides:
        config_params.update(config_overrides)

    logger.info(f"Creating GPT Neo X with custom configuration: {config_params}")
    config = GPTNeoXConfig(**config_params)
    model = GPTNeoXForCausalLM(config)
    return config, model


if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Example usage with default configuration
    config, model = get_gpt_neo_x()
    print(config)
    print(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {total_params - trainable_params}")
