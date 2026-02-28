"""
Phase 8: Evaluation for locally-trained HuggingFace checkpoints.

Loads a model from a local HF checkpoint using vLLM for fast inference
and evaluates on MATH-500, AIME 2024, AIME 2025, GSM8K, OlympiadBench.

Usage:
    python scripts/8_eval_local.py --checkpoint /tmp/tinker-math/grpo_local_fullparam/final
    python scripts/8_eval_local.py --checkpoint Qwen/Qwen3-4B-Instruct-2507  # baseline
    python scripts/8_eval_local.py --checkpoint /path/to/checkpoint --benchmarks math500,aime2024
"""

import argparse
import json
import logging
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils.data import (
    load_gsm8k_test, load_math500, load_aime_2024, load_aime_2025, load_olympiadbench
)

from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.recipes.math_rl.math_env import extract_gsm8k_final_answer, MathEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

QUESTION_SUFFIX = MathEnv.question_suffix()
FEWSHOT_PREFIX = MathEnv.standard_fewshot_prefix()

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
    "olympiadbench": {
        "loader": lambda: load_olympiadbench().filter(lambda r: r["answer_type"] == "Numerical"),
        "get_question": lambda row: row["question"],
        "get_ground_truth": lambda row: row["final_answer"][0],
    },
}


def evaluate_vllm(
    llm,
    tokenizer,
    benchmark_name: str,
    max_new_tokens: int = 4096,
    temperature: float = 0.01,
) -> dict:
    """Evaluate using vLLM for fast batched inference."""
    from vllm import SamplingParams

    bench = BENCHMARKS[benchmark_name]
    dataset = bench["loader"]()
    rows = list(dataset)

    # Build all prompts
    prompts = []
    for row in rows:
        question = bench["get_question"](row)
        messages = [
            *FEWSHOT_PREFIX,
            {"role": "user", "content": question + QUESTION_SUFFIX},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    # Generate all at once with vLLM
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
    )
    outputs = llm.generate(prompts, sampling_params)

    # Grade
    correct = 0
    total = 0
    for output, row in zip(outputs, rows):
        ground_truth = bench["get_ground_truth"](row)
        response_text = output.outputs[0].text

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
    parser = argparse.ArgumentParser(description="Evaluate local HF checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to HF checkpoint or model name")
    parser.add_argument("--benchmarks", default="math500,aime2024,aime2025,gsm8k,olympiadbench")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    bench_names = [x.strip() for x in args.benchmarks.split(",")]

    logger.info(f"Loading model with vLLM from: {args.checkpoint}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    from vllm import LLM
    llm = LLM(
        model=args.checkpoint,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.9,
    )
    logger.info("vLLM model loaded")

    results = {}
    for bench_name in bench_names:
        logger.info(f"Evaluating on {bench_name}...")
        result = evaluate_vllm(
            llm=llm,
            tokenizer=tokenizer,
            benchmark_name=bench_name,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        results[bench_name] = result
        logger.info(f"  {bench_name}: {result['accuracy_pct']} ({result['correct']}/{result['total']})")

    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS: {args.checkpoint}")
    print(f"{'='*60}")
    print(f"{'Benchmark':<20} | {'Accuracy':>10} | {'CI 95%':>10} | {'Correct':>8}")
    print("-" * 60)
    for bench_name in bench_names:
        if bench_name in results:
            r = results[bench_name]
            print(f"{bench_name:<20} | {r['accuracy_pct']:>10} | {r['ci_95']:>10} | {r['correct']:>5}/{r['total']}")
    print(f"{'='*60}")

    # Save results
    output_path = args.output or f"/tmp/{os.path.basename(args.checkpoint)}_eval.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
