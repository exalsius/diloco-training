"""
HuggingFace download utilities for robust dataset and model loading.

This module provides helper functions to configure timeouts and retries
for HuggingFace Hub downloads, making training more resilient to
intermittent network issues.
"""

import os

from datasets import DownloadConfig

# Default timeout and retry settings for HuggingFace downloads
DEFAULT_HF_TIMEOUT = 120
DEFAULT_HF_MAX_RETRIES = 5


def create_download_config(
    timeout: float = DEFAULT_HF_TIMEOUT, max_retries: int = DEFAULT_HF_MAX_RETRIES
) -> DownloadConfig:
    """Create a robust download configuration for HuggingFace datasets.

    Args:
        timeout: Timeout in seconds for HTTP requests (default: 120)
        max_retries: Maximum number of retries for failed downloads (default: 5)

    Returns:
        DownloadConfig with increased timeout and retry settings
    """
    return DownloadConfig(
        max_retries=max_retries,
    )


def set_hf_timeout(timeout: int = DEFAULT_HF_TIMEOUT) -> None:
    """Set HuggingFace Hub download timeout via environment variable.

    This affects model and tokenizer downloads via transformers library.

    Args:
        timeout: Timeout in seconds for HTTP requests
    """
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(timeout)
