"""
Phase 5: Prepare OpenR1-Math-220k for SFT

Downloads OpenR1-Math-220k (default subset, ~93k rows), filters for quality
and decontaminates against AIME 2024/2025 eval benchmarks, then converts
to our JSONL trace format.

Usage:
    python scripts/5_prepare_openr1.py
    python scripts/5_prepare_openr1.py --max-samples 5000
    python scripts/5_prepare_openr1.py --output data/openr1_filtered.jsonl
"""

import argparse
import json
import logging
import os
import re
import sys

import datasets
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils.data import load_aime_2024, load_aime_2025, save_traces_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# System prompt matching our distillation format
SYSTEM_PROMPT = (
    "You are an expert math olympiad solver. Solve the following problem step by step. "
    "Show your reasoning clearly, then provide your final answer inside \\boxed{}."
)
QUESTION_SUFFIX = " Write your answer in \\boxed{} format."


def normalize_for_dedup(text: str) -> str:
    """Normalize a problem string for deduplication matching."""
    text = text.lower().strip()
    # Remove LaTeX commands that vary in formatting
    text = re.sub(r"\s+", " ", text)
    # Remove common prefixes
    text = re.sub(r"^(problem|question|exercise)\s*\d*[.:]\s*", "", text)
    return text


def build_aime_fingerprints() -> set[str]:
    """Build a set of normalized AIME 2024/2025 problem fingerprints for decontamination."""
    fingerprints = set()

    for loader, name in [(load_aime_2024, "AIME 2024"), (load_aime_2025, "AIME 2025")]:
        ds = loader()
        for row in ds:
            problem = row["problem"]
            norm = normalize_for_dedup(problem)
            # Use multiple substring fingerprints for fuzzy matching
            fingerprints.add(norm)
            # Also add first 200 chars as a shorter fingerprint
            fingerprints.add(norm[:200])
        logger.info(f"Loaded {len(ds)} {name} problems for decontamination")

    return fingerprints


def is_contaminated(problem: str, aime_fingerprints: set[str]) -> bool:
    """Check if a problem matches any AIME eval problem."""
    norm = normalize_for_dedup(problem)

    # Exact match
    if norm in aime_fingerprints:
        return True

    # Prefix match (200 chars)
    if norm[:200] in aime_fingerprints:
        return True

    # Substring containment check - check if any AIME problem is substantially
    # contained in this problem or vice versa
    for fp in aime_fingerprints:
        if len(fp) > 50:  # Only check meaningful-length fingerprints
            # Check if the first 100 chars of either match
            if fp[:100] in norm or norm[:100] in fp:
                return True

    return False


def strip_think_tags(text: str) -> str:
    """Strip <think>...</think> wrapper from DeepSeek R1 responses."""
    # Remove opening <think> tag
    text = re.sub(r"^<think>\s*", "", text)
    # Remove closing </think> tag
    text = re.sub(r"\s*</think>\s*", "\n\n", text)
    return text.strip()


def select_best_generation(row: dict) -> str | None:
    """Select the best correct generation from a row.

    Prefers: shortest correct generation that contains \\boxed{}.
    """
    generations = row.get("generations", [])
    correctness = row.get("correctness_math_verify", [])
    is_complete = row.get("is_reasoning_complete", [])

    if not generations:
        return None

    candidates = []
    for i, gen in enumerate(generations):
        # Check correctness
        is_correct = correctness[i] if i < len(correctness) else False
        if not is_correct:
            continue

        # Check reasoning completeness
        complete = is_complete[i] if i < len(is_complete) else True
        if not complete:
            continue

        # Must have boxed answer
        if "\\boxed{" not in gen:
            continue

        candidates.append(gen)

    if not candidates:
        return None

    # Select shortest correct generation
    return min(candidates, key=len)


def main():
    parser = argparse.ArgumentParser(description="Prepare OpenR1-Math-220k for SFT")
    parser.add_argument("--output", default="data/openr1_filtered.jsonl")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to include (None = all)")
    parser.add_argument("--min-length", type=int, default=100,
                        help="Minimum response length in chars")
    parser.add_argument("--max-length", type=int, default=12000,
                        help="Maximum response length in chars")
    args = parser.parse_args()

    # Build AIME decontamination fingerprints
    logger.info("Building AIME decontamination fingerprints...")
    aime_fps = build_aime_fingerprints()
    logger.info(f"Built {len(aime_fps)} fingerprints")

    # Load OpenR1-Math-220k (default subset)
    logger.info("Loading OpenR1-Math-220k (default subset)...")
    ds = datasets.load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")
    logger.info(f"Loaded {len(ds)} rows")

    traces = []
    stats = {
        "total": 0,
        "no_correct_gen": 0,
        "contaminated": 0,
        "too_short": 0,
        "too_long": 0,
        "no_reasoning": 0,
        "accepted": 0,
    }

    for row in tqdm(ds, desc="Processing"):
        stats["total"] += 1

        problem = row["problem"]

        # Decontamination check
        if is_contaminated(problem, aime_fps):
            stats["contaminated"] += 1
            continue

        # Select best generation
        best_gen = select_best_generation(row)
        if best_gen is None:
            stats["no_correct_gen"] += 1
            continue

        # Strip think tags (DeepSeek R1 format -> clean reasoning)
        response_text = strip_think_tags(best_gen)

        # Length filter
        char_len = len(response_text)
        if char_len < args.min_length:
            stats["too_short"] += 1
            continue
        if char_len > args.max_length:
            stats["too_long"] += 1
            continue

        # Must have step-by-step reasoning
        if response_text.count(".") < 2 and response_text.count("\n") < 2:
            stats["no_reasoning"] += 1
            continue

        trace = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem + QUESTION_SUFFIX},
                {"role": "assistant", "content": response_text},
            ],
            "problem": problem,
            "ground_truth": row.get("answer", ""),
            "source": "openr1_math_220k",
            "dataset": "openr1",
            "teacher_model": "deepseek-r1",
            "trace_length": char_len,
            "num_valid": row.get("correctness_count", 1),
            "num_sampled": len(row.get("generations", [])),
        }
        traces.append(trace)
        stats["accepted"] += 1

        if args.max_samples and stats["accepted"] >= args.max_samples:
            logger.info(f"Reached max_samples limit ({args.max_samples})")
            break

    # Save
    save_traces_jsonl(traces, args.output)

    # Print summary
    print(f"\n{'='*60}")
    print("OPENR1-MATH-220K PREPARATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total rows processed:    {stats['total']}")
    print(f"  No correct generation:   {stats['no_correct_gen']}")
    print(f"  Contaminated (AIME):     {stats['contaminated']}")
    print(f"  Too short (<{args.min_length} chars):  {stats['too_short']}")
    print(f"  Too long (>{args.max_length} chars):  {stats['too_long']}")
    print(f"  No reasoning:            {stats['no_reasoning']}")
    print(f"  Accepted:                {stats['accepted']}")
    print(f"  Output:                  {args.output}")
    avg_len = sum(t["trace_length"] for t in traces) / max(len(traces), 1)
    print(f"  Avg trace length:        {avg_len:.0f} chars")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
