"""
Phase 4: Unified Evaluation & Ablation Studies

Evaluates model checkpoints on GSM8K test and MATH-500.
Produces a comparison table for ablation study.

Usage:
    python scripts/4_eval.py
    python scripts/4_eval.py --experiments baseline,sft_custom,grpo
    python scripts/4_eval.py --benchmarks gsm8k,math500
"""

import argparse
import json
import logging
import math
import os
import sys
from concurrent.futures import Future

import tinker
from tinker import types
from tqdm import tqdm

from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.recipes.math_rl.math_env import extract_gsm8k_final_answer, MathEnv
from tinker_cookbook.tokenizer_utils import get_tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils.data import load_gsm8k_test, load_math500, load_aime_2024, load_aime_2025
from scripts.utils.tinker_helpers import STUDENT_MODEL, get_service_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)

QUESTION_SUFFIX = MathEnv.question_suffix()
CONVO_PREFIX = MathEnv.standard_fewshot_prefix()

# Experiment -> checkpoint log_path mapping
EXPERIMENTS = {
    "baseline": None,  # No checkpoint = base model
    "sft_custom": "/tmp/tinker-math/sft",
    "sft_public": "/tmp/tinker-math/sft_public",
    "grpo": "/tmp/tinker-math/grpo",
    "direct_rl": "/tmp/tinker-math/direct_rl",
}

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
    "aime2024": {
        "loader": load_aime_2024,
        "get_question": lambda row: row["problem"],
        "get_ground_truth": lambda row: str(row["answer"]),
    },
    "aime2025": {
        "loader": load_aime_2025,
        "get_question": lambda row: row["problem"],
        "get_ground_truth": lambda row: str(row["answer"]),
    },
}


def evaluate_checkpoint(
    service_client,
    renderer,
    checkpoint_log_path: str | None,
    benchmark_name: str,
    max_tokens: int = 2048,
) -> dict:
    """Evaluate a single checkpoint on a single benchmark."""
    bench = BENCHMARKS[benchmark_name]
    dataset = bench["loader"]()

    # Create sampling client from checkpoint or base model
    if checkpoint_log_path:
        resume_info = checkpoint_utils.get_last_checkpoint(checkpoint_log_path)
        if resume_info:
            training_client = service_client.create_training_client_from_state_with_optimizer(
                resume_info["state_path"]
            )
        else:
            logger.warning(f"No checkpoint at {checkpoint_log_path}, using base model")
            training_client = service_client.create_lora_training_client(
                base_model=STUDENT_MODEL, rank=8
            )
    else:
        training_client = service_client.create_lora_training_client(
            base_model=STUDENT_MODEL, rank=8
        )

    sampling_client = training_client.save_weights_and_get_sampling_client()

    sampling_params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        stop=renderer.get_stop_sequences(),
    )

    # Submit all requests
    futures: list[tuple[Future, dict]] = []
    for row in dataset:
        question = bench["get_question"](row)
        convo = [
            *CONVO_PREFIX,
            {"role": "user", "content": question + QUESTION_SUFFIX},
        ]
        model_input = renderer.build_generation_prompt(convo)
        future = sampling_client.sample(
            prompt=model_input, num_samples=1, sampling_params=sampling_params,
        )
        futures.append((future, row))

    # Collect
    correct = 0
    total = 0
    for future, row in tqdm(futures, desc=f"Eval {benchmark_name}"):
        ground_truth = bench["get_ground_truth"](row)
        sample_result = future.result()
        sequence = sample_result.sequences[0]

        parsed_message, _ = renderer.parse_response(sequence.tokens)
        response_text = renderers.get_text_content(parsed_message)

        try:
            given_answer = extract_boxed(response_text)
            if grade_answer(given_answer, ground_truth):
                correct += 1
        except ValueError:
            pass
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    ci = 1.96 * math.sqrt(accuracy * (1 - accuracy) / total) if total > 0 else 0.0

    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "accuracy_pct": f"{accuracy * 100:.1f}%",
        "ci_95": f"+/-{ci * 100:.1f}%",
    }


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation & ablations")
    parser.add_argument("--experiments", default="baseline,direct_rl")
    parser.add_argument("--benchmarks", default="gsm8k,math500,aime2024,aime2025")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--output-dir", default="results/ablation")
    parser.add_argument("--tinker-url", default=None)
    args = parser.parse_args()

    exp_names = [x.strip() for x in args.experiments.split(",")]
    bench_names = [x.strip() for x in args.benchmarks.split(",")]

    service_client = get_service_client(args.tinker_url)
    tokenizer = get_tokenizer(STUDENT_MODEL)
    renderer_name = model_info.get_recommended_renderer_name(STUDENT_MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    results = {}
    for exp_name in exp_names:
        checkpoint_path = EXPERIMENTS.get(exp_name)
        results[exp_name] = {}

        for bench_name in bench_names:
            logger.info(f"Evaluating {exp_name} on {bench_name}...")
            result = evaluate_checkpoint(
                service_client=service_client,
                renderer=renderer,
                checkpoint_log_path=checkpoint_path,
                benchmark_name=bench_name,
                max_tokens=args.max_tokens,
            )
            results[exp_name][bench_name] = result

    # Print results table
    print(f"\n{'='*70}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*70}")

    header = f"{'Experiment':<20}"
    for bench in bench_names:
        header += f" | {bench:>15}"
    print(header)
    print("-" * 70)

    for exp_name in exp_names:
        if exp_name not in results:
            continue
        row = f"{exp_name:<20}"
        for bench in bench_names:
            if bench in results[exp_name]:
                r = results[exp_name][bench]
                row += f" | {r['accuracy_pct']:>8} {r['ci_95']:>6}"
            else:
                row += f" | {'N/A':>15}"
        print(row)

    print(f"{'='*70}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output_dir}/ablation_results.json")


if __name__ == "__main__":
    main()
