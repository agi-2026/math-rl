"""
Phase 4: Unified Evaluation & Ablation Studies

Evaluates all model checkpoints on GSM8K test and MATH-500.
Produces a comparison table for the ablation study.

Usage:
    python scripts/4_eval.py
    python scripts/4_eval.py --checkpoints baseline,sft_custom,sft_public,grpo,direct_rl
    python scripts/4_eval.py --benchmarks gsm8k,math500
"""

import argparse
import json
import logging
import math
import os
from concurrent.futures import Future
from typing import Optional

import tinker
from tqdm import tqdm

from utils.answer_extraction import (
    extract_gsm8k_final_answer,
    extract_number_from_response,
    is_equivalent,
)
from utils.data import load_gsm8k_test, load_math500
from utils.tinker_helpers import (
    STUDENT_MODEL,
    get_renderer,
    get_service_client,
    get_tokenizer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step, "
    "then provide your final answer inside \\boxed{}."
)

# Experiment definitions: name → checkpoint path (None = base model)
EXPERIMENTS = {
    "baseline": None,
    "sft_custom": "checkpoints/sft/final",
    "sft_public": "checkpoints/sft_public/final",
    "grpo": "checkpoints/grpo/final",
    "direct_rl": "checkpoints/direct_rl/final",
}

# Benchmark definitions
BENCHMARKS = {
    "gsm8k": {
        "loader": load_gsm8k_test,
        "get_question": lambda row: row["question"],
        "get_ground_truth": lambda row: extract_gsm8k_final_answer(row["answer"]),
    },
    "math500": {
        "loader": load_math500,
        "get_question": lambda row: row["problem"],
        "get_ground_truth": lambda row: row["answer"],
    },
}


def evaluate_checkpoint(
    service_client,
    tokenizer,
    renderer,
    checkpoint_path: Optional[str],
    benchmark_name: str,
    max_tokens: int = 2048,
) -> dict:
    """Evaluate a single checkpoint on a single benchmark."""

    bench = BENCHMARKS[benchmark_name]
    dataset = bench["loader"]()

    # Create sampling client
    if checkpoint_path and os.path.exists(checkpoint_path):
        training_client = service_client.create_training_client_from_state_with_optimizer(
            checkpoint_path
        )
    else:
        training_client = service_client.create_lora_training_client(
            base_model=STUDENT_MODEL, rank=8
        )

    sampling_client = training_client.save_weights_and_get_sampling_client()

    sampling_params = tinker.types.SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        stop=renderer.get_stop_sequences() if renderer else [],
    )

    # Submit all requests
    futures: list[tuple[Future, dict]] = []
    for row in dataset:
        question = bench["get_question"](row)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        if renderer:
            model_input = renderer.build_generation_prompt(messages)
        else:
            model_input = tinker.types.ModelInput.from_text(
                f"{SYSTEM_PROMPT}\n\n{question}\n\n"
            )

        future = sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )
        futures.append((future, row))

    # Collect results
    correct = 0
    total = 0
    for future, row in tqdm(futures, desc=f"Eval {benchmark_name}"):
        ground_truth = bench["get_ground_truth"](row)
        sample_result = future.result()
        sequence = sample_result.sequences[0]

        if renderer:
            parsed_message, _ = renderer.parse_response(sequence.tokens)
            from tinker_cookbook import renderers as r
            response_text = r.get_text_content(parsed_message)
        else:
            response_text = sequence.text

        extracted = extract_number_from_response(response_text)
        if extracted is not None and is_equivalent(extracted, ground_truth):
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    ci = 1.96 * math.sqrt(accuracy * (1 - accuracy) / total) if total > 0 else 0.0

    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "accuracy_pct": f"{accuracy * 100:.1f}%",
        "ci_95": f"±{ci * 100:.1f}%",
    }


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation & ablations")
    parser.add_argument(
        "--checkpoints",
        default="baseline,sft_custom,grpo",
        help="Comma-separated experiment names",
    )
    parser.add_argument(
        "--benchmarks",
        default="gsm8k,math500",
        help="Comma-separated benchmark names",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--output-dir", default="results/ablation")
    parser.add_argument("--tinker-url", default=None)
    args = parser.parse_args()

    experiment_names = [x.strip() for x in args.checkpoints.split(",")]
    benchmark_names = [x.strip() for x in args.benchmarks.split(",")]

    # Setup
    service_client = get_service_client(args.tinker_url)
    tokenizer = get_tokenizer(STUDENT_MODEL)
    renderer = get_renderer(STUDENT_MODEL, tokenizer)

    # Run all evaluations
    results = {}
    for exp_name in experiment_names:
        checkpoint_path = EXPERIMENTS.get(exp_name)

        # Skip if checkpoint doesn't exist (except baseline)
        if checkpoint_path and not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint for '{exp_name}' not found at {checkpoint_path}, skipping")
            continue

        results[exp_name] = {}
        for bench_name in benchmark_names:
            logger.info(f"Evaluating {exp_name} on {bench_name}...")
            result = evaluate_checkpoint(
                service_client=service_client,
                tokenizer=tokenizer,
                renderer=renderer,
                checkpoint_path=checkpoint_path,
                benchmark_name=bench_name,
                max_tokens=args.max_tokens,
            )
            results[exp_name][bench_name] = result

    # Print results table
    print(f"\n{'='*70}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*70}")

    # Header
    header = f"{'Experiment':<20}"
    for bench in benchmark_names:
        header += f" | {bench:>15}"
    print(header)
    print("-" * 70)

    # Rows
    for exp_name in experiment_names:
        if exp_name not in results:
            continue
        row = f"{exp_name:<20}"
        for bench in benchmark_names:
            if bench in results[exp_name]:
                r = results[exp_name][bench]
                row += f" | {r['accuracy_pct']:>8} {r['ci_95']:>6}"
            else:
                row += f" | {'N/A':>15}"
        print(row)

    print(f"{'='*70}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output_dir}/ablation_results.json")


if __name__ == "__main__":
    main()
