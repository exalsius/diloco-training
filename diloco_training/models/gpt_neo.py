"""
GPT-Neo model implementation module.

This module provides functionality to create and configure GPT-Neo models
for causal language modeling tasks.
"""

from typing import Dict, Any, Optional
import logging
from transformers import GPTNeoConfig, GPTNeoForCausalLM

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
DEFAULT_CONFIG = {
    "hidden_size": 128,
    "intermediate_size": 512,
    "num_hidden_layers": 6,
    "num_attention_heads": 4,
}


def get_gpt_neo(
    model_name: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> GPTNeoForCausalLM:
    """
    Create and return a GPT-Neo model for causal language modeling.
    
    Args:
        model_name: Name or path of a pretrained model to load.
                   If None, creates a model with custom configuration.
        config_overrides: Dictionary of parameters to override in the default configuration.
    
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
            return GPTNeoForCausalLM.from_pretrained(model_name)
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
    return model


if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Example usage with default configuration
    model = get_gpt_neo()
    print(model)
    
    # Example with configuration overrides
    custom_model = get_gpt_neo(config_overrides={
        "hidden_size": 256,
        "num_hidden_layers": 4,
    })
    print(custom_model)
