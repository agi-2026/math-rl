"""
Tinker API helper utilities.

Wraps common Tinker patterns: client setup, datum construction,
chat rendering, and checkpoint management.
"""

import logging
import os
from typing import Optional

import tinker
from tinker import types
from tinker.types.tensor_data import TensorData
import torch

logger = logging.getLogger(__name__)

# Model identifiers on Tinker
STUDENT_MODEL = "qwen3-4b-instruct-2507"
TEACHER_MODEL = "qwen3-235b-a22b-instruct-2507"

# Default LoRA config
DEFAULT_LORA_RANK = 64


def get_service_client(base_url: Optional[str] = None) -> tinker.ServiceClient:
    """Create a Tinker ServiceClient.

    Uses TINKER_BASE_URL env var if base_url not provided.
    """
    if base_url is None:
        base_url = os.environ.get("TINKER_BASE_URL", None)
    return tinker.ServiceClient(base_url=base_url)


def create_training_client(
    service_client: tinker.ServiceClient,
    model_name: str = STUDENT_MODEL,
    lora_rank: int = DEFAULT_LORA_RANK,
) -> tinker.LoraTrainingClient:
    """Create a LoRA training client for the given model."""
    return service_client.create_lora_training_client(
        base_model=model_name,
        rank=lora_rank,
    )


def get_tokenizer(model_name: str):
    """Get tokenizer for the given model.

    Uses tinker_cookbook's tokenizer utils if available,
    falls back to transformers.
    """
    try:
        from tinker_cookbook.tokenizer_utils import get_tokenizer as _get_tokenizer
        return _get_tokenizer(model_name)
    except ImportError:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def get_renderer(model_name: str, tokenizer):
    """Get the appropriate chat renderer for the model.

    Uses tinker_cookbook's renderer system if available.
    """
    try:
        from tinker_cookbook import model_info, renderers
        renderer_name = model_info.get_recommended_renderer_name(model_name)
        return renderers.get_renderer(renderer_name, tokenizer)
    except ImportError:
        logger.warning("tinker_cookbook not available, using basic renderer")
        return None


def build_sft_datum(
    messages: list[dict],
    tokenizer,
    renderer=None,
    max_seq_length: int = 2048,
) -> Optional[types.Datum]:
    """Build a Tinker Datum for supervised fine-tuning.

    Tokenizes the conversation, masks system+user tokens,
    trains only on assistant tokens.

    Args:
        messages: List of {role, content} message dicts.
        tokenizer: Model tokenizer.
        renderer: Optional Tinker renderer for chat formatting.
        max_seq_length: Maximum sequence length.

    Returns:
        Tinker Datum or None if sequence too long.
    """
    if renderer is not None:
        # Use Tinker's native rendering
        model_input = renderer.build_training_input(messages)
        if model_input.length > max_seq_length:
            return None

        # Build target tokens with masking
        # In Tinker SFT, we pass the full sequence and mask non-assistant tokens
        tokens = model_input.tokens
        target_tokens = list(tokens)

        # Create loss mask: 1 for assistant tokens, 0 for others
        loss_mask = [0.0] * len(tokens)
        # The renderer should handle identifying assistant token spans
        # For now, use a simple heuristic based on the messages structure
        _apply_assistant_mask(messages, tokenizer, tokens, loss_mask)

        return types.Datum(
            model_input=model_input,
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                "loss_mask": TensorData.from_torch(torch.tensor(loss_mask)),
            },
        )
    else:
        # Fallback: use tokenizer directly
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) > max_seq_length:
            return None

        target_tokens = list(tokens)
        loss_mask = [0.0] * len(tokens)
        _apply_assistant_mask(messages, tokenizer, tokens, loss_mask)

        model_input = types.ModelInput.from_tokens(tokens)
        return types.Datum(
            model_input=model_input,
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                "loss_mask": TensorData.from_torch(torch.tensor(loss_mask)),
            },
        )


def _apply_assistant_mask(
    messages: list[dict],
    tokenizer,
    tokens: list[int],
    loss_mask: list[float],
) -> None:
    """Set loss_mask to 1.0 for assistant response tokens.

    Uses the tokenizer to find where assistant content starts/ends.
    """
    # Tokenize up to each assistant message to find boundaries
    prefix_text = ""
    for msg in messages:
        if msg["role"] == "assistant":
            # Tokenize everything before this assistant message
            prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
            prefix_len = len(prefix_tokens)

            # Tokenize including this assistant message
            full_text = prefix_text + msg["content"]
            full_tokens = tokenizer.encode(full_text, add_special_tokens=False)

            # Mark assistant tokens
            for i in range(prefix_len, min(len(full_tokens), len(loss_mask))):
                loss_mask[i] = 1.0

        # Build up the prefix (approximate — chat templates vary)
        prefix_text += msg.get("content", "")


def build_rl_datum(
    prompt_input: types.ModelInput,
    sampled_tokens: list[int],
    logprobs: list[float],
    advantage: float,
) -> types.Datum:
    """Build a Tinker Datum for RL (importance sampling loss).

    Following the Tinker cookbook rl_loop.py pattern.
    """
    ob_len = prompt_input.length - 1
    model_input = prompt_input.append(
        types.EncodedTextChunk(tokens=sampled_tokens[:-1])
    )
    target_tokens = [0] * ob_len + sampled_tokens
    padded_logprobs = [0.0] * ob_len + logprobs
    padded_advantages = [0.0] * ob_len + [advantage] * (model_input.length - ob_len)

    return types.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
            "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
            "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
        },
    )
