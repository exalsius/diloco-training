"""
Llama model implementation module.

This module provides functionality to create and configure Llama models
for causal language modeling tasks.
"""

import logging
from typing import Any, Dict, Optional, Tuple

from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "EleutherAI/gpt-neox-20b"
DEFAULT_CONFIG: Dict[str, Any] = {
    "hidden_size": 896,
    "num_hidden_layers": 3,
    "num_attention_heads": 16,
    "intermediate_size": 896,
}
TINY_LLAMA_CONFIG: Dict[str, Any] = {
    "vocab_size": 100,
    "hidden_size": 8,
    "max_position_embeddings": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 1,
    "intermediate_size": 32,
}


def get_tiny_llama() -> Tuple[GPTNeoXConfig, GPTNeoXForCausalLM]:
    """Returns a tiny Llama model suitable for testing purposes.

    This is a convenience function that creates a minimal Llama model
    with a small configuration, making it useful for testing and development.
    """
    config = GPTNeoXConfig(**TINY_LLAMA_CONFIG)
    return config, GPTNeoXForCausalLM(config)


def get_cpu_llama() -> Tuple[GPTNeoXConfig, GPTNeoXForCausalLM]:
    """Returns a Llama model with default configuration for CPU usage.

    This is a convenience function that creates a Llama model
    with the default configuration, making it suitable for CPU-based applications.
    """
    config = GPTNeoXConfig(**DEFAULT_CONFIG)
    return config, GPTNeoXForCausalLM(config)


def get_gpt_neo_x(
    model_name: Optional[str] = None, config_overrides: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[GPTNeoXConfig], GPTNeoXForCausalLM]:
    """
    Create and return a Llama model for causal language modeling.

    Args:
        model_name: Name or path of a pretrained model to load.
                   If None, creates a model with custom configuration.
        config_overrides: Dictionary of parameters to override in the default configuration.

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
            logger.info(f"Loading pretrained model: {model_name}")
            return None, GPTNeoXForCausalLM.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Failed to load pretrained model: {e}")
            raise

    # Use default config with any overrides
    config_params = DEFAULT_CONFIG.copy()
    if config_overrides:
        config_params.update(config_overrides)

    logger.info(f"Creating Llama with custom configuration: {config_params}")
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
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
