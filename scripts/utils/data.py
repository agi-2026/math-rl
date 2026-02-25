"""
Data loading utilities.

Wraps HuggingFace dataset loading for GSM8K and MATH-500.
"""

import json
import os
from typing import cast

import datasets


def load_gsm8k_train() -> datasets.Dataset:
    """Load GSM8K training split."""
    ds = datasets.load_dataset("openai/gsm8k", "main", split="train")
    return cast(datasets.Dataset, ds)


def load_gsm8k_test() -> datasets.Dataset:
    """Load GSM8K test split."""
    ds = datasets.load_dataset("openai/gsm8k", "main", split="test")
    return cast(datasets.Dataset, ds)


def load_math500() -> datasets.Dataset:
    """Load MATH-500 benchmark (500 competition-level problems)."""
    ds = datasets.load_dataset("HuggingFaceH4/MATH-500", name="default", split="test")
    return cast(datasets.Dataset, ds)


def save_traces_jsonl(traces: list[dict], output_path: str) -> None:
    """Save reasoning traces to JSONL format."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
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
