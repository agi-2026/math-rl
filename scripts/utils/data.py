"""
Data loading and formatting utilities.

Handles GSM8K, MATH-500, and custom trace datasets.
"""

import json
import os
from typing import Optional

import datasets


def load_gsm8k_train() -> datasets.Dataset:
    """Load GSM8K training split."""
    dataset = datasets.load_dataset("openai/gsm8k", "main")
    return dataset["train"]


def load_gsm8k_test() -> datasets.Dataset:
    """Load GSM8K test split."""
    dataset = datasets.load_dataset("openai/gsm8k", "main")
    return dataset["test"]


def load_math500() -> datasets.Dataset:
    """Load MATH-500 benchmark.

    Uses HuggingFace's MATH-500 subset (500 competition-level problems).
    """
    dataset = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
    return dataset


def build_chat_messages(
    problem: str,
    system_prompt: str = "You are a helpful math assistant. Solve the problem step by step, then provide your final answer inside \\boxed{}.",
    response: Optional[str] = None,
) -> list[dict]:
    """Build chat messages in standard format.

    Args:
        problem: The math problem text.
        system_prompt: System instruction.
        response: Optional assistant response (for SFT data).

    Returns:
        List of message dicts with 'role' and 'content' keys.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    return messages


def save_traces_jsonl(traces: list[dict], output_path: str) -> None:
    """Save reasoning traces to JSONL format.

    Each line: {"messages": [...], "problem": str, "ground_truth": str, "source": str}
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for trace in traces:
            f.write(json.dumps(trace) + "\n")


def load_traces_jsonl(path: str) -> list[dict]:
    """Load reasoning traces from JSONL."""
    traces = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


def load_public_sft_data(
    dataset_name: str = "open-thoughts/OpenThoughts-114k",
    max_samples: Optional[int] = None,
    filter_math: bool = True,
) -> list[dict]:
    """Load public SFT data for comparison experiments.

    Args:
        dataset_name: HuggingFace dataset identifier.
        max_samples: Optional cap on number of samples.
        filter_math: If True, filter to math-related samples only.

    Returns:
        List of trace dicts in our standard format.
    """
    dataset = datasets.load_dataset(dataset_name, split="train")

    traces = []
    for row in dataset:
        # OpenThoughts format: has 'conversations' field
        if "conversations" in row:
            messages = row["conversations"]
        elif "messages" in row:
            messages = row["messages"]
        else:
            continue

        if filter_math:
            # Check if problem mentions math-related keywords
            content = str(messages).lower()
            math_keywords = [
                "calculate", "solve", "equation", "number", "math",
                "sum", "product", "find the value", "compute", "how many",
                "what is", "fraction", "percent", "ratio", "area",
                "volume", "probability", "integer",
            ]
            if not any(kw in content for kw in math_keywords):
                continue

        trace = {
            "messages": messages,
            "source": dataset_name,
        }
        traces.append(trace)

        if max_samples and len(traces) >= max_samples:
            break

    return traces
