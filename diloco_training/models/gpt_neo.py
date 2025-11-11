"""
GPT-Neo model implementation module.

This module provides functionality to create and configure GPT-Neo models
for causal language modeling tasks.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from transformers import GPTNeoConfig, GPTNeoForCausalLM

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
DEFAULT_CONFIG: Dict[str, Any] = {
    "hidden_size": 512,
    "num_hidden_layers": 3,
    "num_heads": 16,
}
TINY_GPT_NEO_CONFIG: Dict[str, Any] = {
    "vocab_size": 100,
    "hidden_size": 8,
    "max_position_embeddings": 512,
    "num_layers": 2,
    "num_heads": 1,
    "attention_types": [[["global", "local"], 1]],
}


def get_tiny_gpt_neo(
    cache_dir: Optional[Path] = None,
) -> Tuple[GPTNeoConfig, GPTNeoForCausalLM]:
    """Returns a tiny GPT-Neo model suitable for testing purposes.

    This is a convenience function that creates a minimal GPT-Neo model
    with a small configuration, making it useful for testing and development.

    Args:
        cache_dir: Directory for caching models (not used for models created from config)
    """
    config = GPTNeoConfig(**TINY_GPT_NEO_CONFIG)
    return config, GPTNeoForCausalLM(config)


def get_cpu_gpt_neo(
    cache_dir: Optional[Path] = None,
) -> Tuple[GPTNeoConfig, GPTNeoForCausalLM]:
    """Returns a tiny GPT-Neo model suitable for testing purposes.

    This is a convenience function that creates a minimal GPT-Neo model
    with a small configuration, making it useful for testing and development.

    Args:
        cache_dir: Directory for caching models (not used for models created from config)
    """
    config = GPTNeoConfig(**DEFAULT_CONFIG)
    return config, GPTNeoForCausalLM(config)


def get_gpt_neo(
    model_name: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[Path] = None,
) -> Tuple[Optional[GPTNeoConfig], GPTNeoForCausalLM]:
    """
    Create and return a GPT-Neo model for causal language modeling.

    Args:
        model_name: Name or path of a pretrained model to load.
                   If None, creates a model with custom configuration.
        config_overrides: Dictionary of parameters to override in the default configuration.
        cache_dir: Directory for caching models. If None, uses HuggingFace default.

    Returns:
        A GPTNeoForCausalLM model instance.

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
            logger.info(f"Loading pretrained model: {model_name}")
            return None, GPTNeoForCausalLM.from_pretrained(
                model_name, cache_dir=cache_dir
            )
        except Exception as e:
            logger.error(f"Failed to load pretrained model: {e}")
            raise

    # Use default config with any overrides
    config_params = DEFAULT_CONFIG.copy()
    if config_overrides:
        config_params.update(config_overrides)

    logger.info(f"Creating GPT-Neo with custom configuration: {config_params}")
    config = GPTNeoConfig(**config_params)
    model = GPTNeoForCausalLM(config)
    return config, model


if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Example usage with default configuration
    config, model = get_gpt_neo()
    print(config)
    print(model)
    print(sum(p.numel() for p in model.parameters()))

    # Example with configuration overrides
    custom_config, custom_model = get_gpt_neo(
        config_overrides={
            "hidden_size": 256,
            "num_hidden_layers": 4,
        }
    )
    print(custom_config)
    print(custom_model)

    tiny_config, tiny_model = get_tiny_gpt_neo()
    print(tiny_config)
    print(tiny_model)
