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


def load_aime_2024() -> datasets.Dataset:
    """Load AIME 2024 (30 problems: AIME I + II).

    Fields: problem_idx, problem, answer (int64).
    AIME answers are integers 0-999.
    """
    ds_i = datasets.load_dataset("MathArena/aime_2024_I", split="train")
    ds_ii = datasets.load_dataset("MathArena/aime_2024_II", split="train")
    combined = datasets.concatenate_datasets([cast(datasets.Dataset, ds_i), cast(datasets.Dataset, ds_ii)])
    return cast(datasets.Dataset, combined)


def load_aime_2025() -> datasets.Dataset:
    """Load AIME 2025 (30 problems: AIME I + II combined).

    Fields: problem_idx, problem, answer (int64), problem_type.
    AIME answers are integers 0-999.
    """
    ds = datasets.load_dataset("MathArena/aime_2025", split="train")
    return cast(datasets.Dataset, ds)


def load_olympiadbench() -> datasets.Dataset:
    """Load OlympiadBench (674 olympiad-level math problems).

    Fields: question, solution (list[str]), final_answer (list[str]),
            answer_type, subfield, is_multiple_answer.
    Answer types: Numerical (572), Expression (63), Tuple (33), Interval (6).
    """
    ds = datasets.load_dataset("math-ai/olympiadbench", split="test")
    return cast(datasets.Dataset, ds)


def load_math_train_hard() -> datasets.Dataset:
    """Load MATH training set filtered to Level 4-5 (3,994 problems).

    These are competition-level problems suitable for teacher distillation.
    Fields: problem, level, type, solution.
    Answers are in \\boxed{} format within the solution field.
    """
    full = load_math_train()
    hard = full.filter(lambda row: row["level"] in ("Level 4", "Level 5"))
    return cast(datasets.Dataset, hard)


def load_math_train() -> datasets.Dataset:
    """Load full MATH training set (all subjects, ~7.5k problems).

    Fields: problem, level, type, solution.
    Answers are in \\boxed{} format within the solution field.
    """
    all_splits = []
    for config_name in [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
    ]:
        ds = datasets.load_dataset("EleutherAI/hendrycks_math", config_name, split="train")
        all_splits.append(cast(datasets.Dataset, ds))
    combined = datasets.concatenate_datasets(all_splits)
    return cast(datasets.Dataset, combined)


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
