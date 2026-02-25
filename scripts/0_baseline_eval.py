"""
Phase 0: Baseline Evaluation

Evaluate Qwen3-4B-Instruct on GSM8K test and MATH-500 before any training.
Establishes baseline accuracy for ablation comparisons.

Usage:
    python scripts/0_baseline_eval.py
    python scripts/0_baseline_eval.py --benchmarks gsm8k
    python scripts/0_baseline_eval.py --benchmarks math500
    python scripts/0_baseline_eval.py --max-tokens 2048
"""

import argparse
import json
import logging
import os
import time
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
    """Evaluate model on a benchmark dataset.

    Args:
        sampling_client: Tinker sampling client.
        renderer: Chat renderer for the model.
        dataset: HuggingFace dataset.
        benchmark_name: Name for logging/saving.
        get_ground_truth: Function to extract ground truth from a row.
        get_question: Function to extract question text from a row.
        max_tokens: Max generation tokens.
        output_dir: Where to save results.

    Returns:
        Dict with accuracy, total, correct count, and per-sample results.
    """
    logger.info(f"Evaluating on {benchmark_name} ({len(dataset)} samples)...")

    sampling_params = tinker.types.SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,  # greedy decoding for deterministic eval
        stop=renderer.get_stop_sequences() if renderer else [],
    )

    # Submit all sampling requests
    futures: list[tuple[Future, dict, str]] = []
    for row in dataset:
        question = get_question(row)
        ground_truth = get_ground_truth(row)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        if renderer:
            model_input = renderer.build_generation_prompt(messages)
        else:
            # Fallback: use raw text
            model_input = tinker.types.ModelInput.from_text(
                f"{SYSTEM_PROMPT}\n\n{question}\n\n"
            )

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

        if renderer:
            parsed_message, _ = renderer.parse_response(sequence.tokens)
            from tinker_cookbook import renderers as r
            response_text = r.get_text_content(parsed_message)
        else:
            response_text = sequence.text

        extracted = extract_number_from_response(response_text)
        is_correct = extracted is not None and is_equivalent(extracted, ground_truth)

        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question": get_question(row),
            "ground_truth": ground_truth,
            "extracted_answer": extracted,
            "is_correct": is_correct,
            "response": response_text[:500],  # truncate for storage
        })

    accuracy = correct / total if total > 0 else 0.0

    # Confidence interval (95%, normal approximation)
    import math
    ci = 1.96 * math.sqrt(accuracy * (1 - accuracy) / total) if total > 0 else 0.0

    summary = {
        "benchmark": benchmark_name,
        "model": STUDENT_MODEL,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "accuracy_pct": f"{accuracy * 100:.1f}%",
        "ci_95": f"±{ci * 100:.1f}%",
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{benchmark_name}_results.json"), "w") as f:
        json.dump({"summary": summary, "samples": results}, f, indent=2)

    logger.info(
        f"{benchmark_name}: {correct}/{total} = {accuracy*100:.1f}% ({ci*100:.1f}% CI)"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["gsm8k", "math500"],
        choices=["gsm8k", "math500"],
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--output-dir", default="results/baseline")
    parser.add_argument("--tinker-url", default=None)
    args = parser.parse_args()

    # Setup Tinker
    service_client = get_service_client(args.tinker_url)
    tokenizer = get_tokenizer(STUDENT_MODEL)
    renderer = get_renderer(STUDENT_MODEL, tokenizer)

    # Create a sampling client (no training, just inference)
    training_client = service_client.create_lora_training_client(
        base_model=STUDENT_MODEL, rank=8  # minimal rank, we won't train
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

    # Print final summary
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION RESULTS")
    print("=" * 60)
    for s in summaries:
        print(f"  {s['benchmark']:>10}: {s['accuracy_pct']} {s['ci_95']}")
    print("=" * 60)

    # Save combined summary
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)


if __name__ == "__main__":
    main()
