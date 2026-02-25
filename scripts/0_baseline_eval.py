"""
Phase 0: Baseline Evaluation

Evaluate Qwen3-4B on GSM8K test and MATH-500 before any training.
Establishes baseline accuracy for ablation comparisons.

Usage:
    python scripts/0_baseline_eval.py
    python scripts/0_baseline_eval.py --benchmarks gsm8k
    python scripts/0_baseline_eval.py --benchmarks math500
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from concurrent.futures import Future

import tinker
from tinker import types
from tqdm import tqdm

from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.recipes.math_rl.math_env import extract_gsm8k_final_answer, MathEnv
from tinker_cookbook.tokenizer_utils import get_tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils.data import load_gsm8k_test, load_math500
from scripts.utils.tinker_helpers import STUDENT_MODEL, get_service_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)

QUESTION_SUFFIX = MathEnv.question_suffix()  # " Write your answer in \boxed{} format."
CONVO_PREFIX = MathEnv.standard_fewshot_prefix()


def evaluate_benchmark(
    sampling_client,
    renderer,
    dataset,
    benchmark_name: str,
    get_ground_truth,
    get_question,
    max_tokens: int = 2048,
    output_dir: str = "results/baseline",
) -> dict:
    """Evaluate model on a benchmark dataset."""
    logger.info(f"Evaluating on {benchmark_name} ({len(dataset)} samples)...")

    sampling_params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,  # greedy decoding
        stop=renderer.get_stop_sequences(),
    )

    # Submit all sampling requests
    futures: list[tuple[Future, dict, str]] = []
    for row in dataset:
        question = get_question(row)
        ground_truth = get_ground_truth(row)

        convo = [
            *CONVO_PREFIX,
            {"role": "user", "content": question + QUESTION_SUFFIX},
        ]
        model_input = renderer.build_generation_prompt(convo)

        future = sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )
        futures.append((future, row, ground_truth))

    # Collect results
    correct = 0
    total = 0
    results = []

    for future, row, ground_truth in tqdm(futures, desc=f"Eval {benchmark_name}"):
        sample_result = future.result()
        sequence = sample_result.sequences[0]

        parsed_message, _ = renderer.parse_response(sequence.tokens)
        response_text = renderers.get_text_content(parsed_message)

        try:
            given_answer = extract_boxed(response_text)
            is_correct = grade_answer(given_answer, ground_truth)
        except ValueError:
            is_correct = False

        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question": get_question(row),
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "response": response_text[:500],
        })

    accuracy = correct / total if total > 0 else 0.0
    ci = 1.96 * math.sqrt(accuracy * (1 - accuracy) / total) if total > 0 else 0.0

    summary = {
        "benchmark": benchmark_name,
        "model": STUDENT_MODEL,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "accuracy_pct": f"{accuracy * 100:.1f}%",
        "ci_95": f"+/-{ci * 100:.1f}%",
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{benchmark_name}_results.json"), "w") as f:
        json.dump({"summary": summary, "samples": results}, f, indent=2)

    logger.info(f"{benchmark_name}: {correct}/{total} = {accuracy*100:.1f}% ({ci*100:.1f}% CI)")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation")
    parser.add_argument(
        "--benchmarks", nargs="+", default=["gsm8k", "math500"],
        choices=["gsm8k", "math500"],
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--output-dir", default="results/baseline")
    parser.add_argument("--tinker-url", default=None)
    args = parser.parse_args()

    # Setup Tinker
    service_client = get_service_client(args.tinker_url)
    tokenizer = get_tokenizer(STUDENT_MODEL)
    renderer_name = model_info.get_recommended_renderer_name(STUDENT_MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Model: {STUDENT_MODEL}, Renderer: {renderer_name}")

    # Create sampling client (no training, just inference)
    training_client = service_client.create_lora_training_client(
        base_model=STUDENT_MODEL, rank=8
    )
    sampling_client = training_client.save_weights_and_get_sampling_client()

    summaries = []

    if "gsm8k" in args.benchmarks:
        dataset = load_gsm8k_test()
        summary = evaluate_benchmark(
            sampling_client=sampling_client,
            renderer=renderer,
            dataset=dataset,
            benchmark_name="gsm8k",
            get_ground_truth=lambda row: extract_gsm8k_final_answer(row["answer"]),
            get_question=lambda row: row["question"],
            max_tokens=args.max_tokens,
            output_dir=args.output_dir,
        )
        summaries.append(summary)

    if "math500" in args.benchmarks:
        dataset = load_math500()
        summary = evaluate_benchmark(
            sampling_client=sampling_client,
            renderer=renderer,
            dataset=dataset,
            benchmark_name="math500",
            get_ground_truth=lambda row: row["answer"],
            get_question=lambda row: row["problem"],
            max_tokens=args.max_tokens,
            output_dir=args.output_dir,
        )
        summaries.append(summary)

    print(f"\n{'='*60}")
    print("BASELINE EVALUATION RESULTS")
    print(f"{'='*60}")
    for s in summaries:
        print(f"  {s['benchmark']:>10}: {s['accuracy_pct']} {s['ci_95']}")
    print(f"{'='*60}")

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)


if __name__ == "__main__":
    main()
