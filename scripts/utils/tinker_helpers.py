"""
Tinker API helper utilities.

Centralizes model names, client setup, and renderer/tokenizer access
using tinker_cookbook's infrastructure.
"""

import os
from typing import Optional

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Model identifiers on Tinker
STUDENT_MODEL = "Qwen/Qwen3-4B"
TEACHER_MODEL = "Qwen/Qwen3-235B-A22B-Instruct"

# Default LoRA config
DEFAULT_LORA_RANK = 64


def get_service_client(base_url: Optional[str] = None) -> tinker.ServiceClient:
    """Create a Tinker ServiceClient."""
    if base_url is None:
        base_url = os.environ.get("TINKER_BASE_URL", None)
    return tinker.ServiceClient(base_url=base_url)


def get_renderer_for_model(model_name: str) -> tuple:
    """Get tokenizer and renderer for a model.

    Returns:
        (tokenizer, renderer, renderer_name)
    """
    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    return tokenizer, renderer, renderer_name
