"""
Phase 1: Teacher Distillation -- Reasoning Trace Generation

Uses Qwen3-235B (via Tinker) to generate high-quality chain-of-thought
reasoning traces on GSM8K training problems. These traces are used for
SFT cold-start in Phase 2.

Usage:
    python scripts/1_generate_traces.py
    python scripts/1_generate_traces.py --num-samples 4 --output data/traces_custom.jsonl
    python scripts/1_generate_traces.py --resume
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import Future

import tinker
from tinker import types
from tqdm import tqdm

from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.recipes.math_rl.math_env import extract_gsm8k_final_answer
from tinker_cookbook.tokenizer_utils import get_tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils.data import load_gsm8k_train, save_traces_jsonl, load_traces_jsonl
from scripts.utils.tinker_helpers import TEACHER_MODEL, get_service_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)

SYSTEM_PROMPT = (
    "You are an expert math tutor. Solve the following problem step by step. "
    "Show your reasoning clearly, then provide your final answer inside \\boxed{}."
)
QUESTION_SUFFIX = " Write your answer in \\boxed{} format."


def generate_traces(
    sampling_client,
    renderer,
    dataset,
    num_samples: int = 4,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    min_trace_chars: int = 100,
    max_trace_chars: int = 8000,
    already_done: set | None = None,
) -> list[dict]:
    """Generate reasoning traces for all problems in the dataset."""
    if already_done is None:
        already_done = set()

    sampling_params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=renderer.get_stop_sequences(),
    )

    traces = []
    no_correct = 0

    problems = [row for row in dataset if row["question"] not in already_done]
    logger.info(
        f"Generating traces for {len(problems)} problems "
        f"({len(already_done)} already done, K={num_samples})"
    )

    # Process in batches to manage concurrent requests
    batch_size = 32
    for batch_start in range(0, len(problems), batch_size):
        batch = problems[batch_start : batch_start + batch_size]

        # Submit sampling requests
        futures: list[tuple[Future, dict]] = []
        for row in batch:
            convo = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["question"] + QUESTION_SUFFIX},
            ]
            model_input = renderer.build_generation_prompt(convo)

            future = sampling_client.sample(
                prompt=model_input,
                num_samples=num_samples,
                sampling_params=sampling_params,
            )
            futures.append((future, row))

        # Collect and filter
        for future, row in tqdm(
            futures,
            desc=f"Batch {batch_start // batch_size + 1}",
            total=len(futures),
        ):
            question = row["question"]
            ground_truth = extract_gsm8k_final_answer(row["answer"])

            sample_result = future.result()
            valid_traces = []

            for sequence in sample_result.sequences:
                parsed_message, _ = renderer.parse_response(sequence.tokens)
                response_text = renderers.get_text_content(parsed_message)

                # Filter: must have \boxed{} and be correct
                try:
                    given_answer = extract_boxed(response_text)
                    if not grade_answer(given_answer, ground_truth):
                        continue
                except ValueError:
                    continue

                # Filter: reasonable length
                char_len = len(response_text)
                if char_len < min_trace_chars or char_len > max_trace_chars:
                    continue

                # Filter: must have step-by-step reasoning
                if response_text.count(".") < 2 and response_text.count("\n") < 2:
                    continue

                valid_traces.append({
                    "text": response_text,
                    "length": char_len,
                })

            if not valid_traces:
                no_correct += 1
                continue

            # Select shortest correct trace
            best = min(valid_traces, key=lambda t: t["length"])

            trace = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question + QUESTION_SUFFIX},
                    {"role": "assistant", "content": best["text"]},
                ],
                "problem": question,
                "ground_truth": ground_truth,
                "source": "teacher_distillation",
                "teacher_model": TEACHER_MODEL,
                "trace_length": best["length"],
                "num_valid": len(valid_traces),
                "num_sampled": num_samples,
            }
            traces.append(trace)

    logger.info(
        f"Generated {len(traces)} traces from {len(problems)} problems "
        f"({no_correct} had no correct completions)"
    )
    return traces


def main():
    parser = argparse.ArgumentParser(description="Generate teacher distillation traces")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output", default="data/traces_custom.jsonl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tinker-url", default=None)
    args = parser.parse_args()

    # Setup Tinker with teacher model
    service_client = get_service_client(args.tinker_url)
    tokenizer = get_tokenizer(TEACHER_MODEL)
    renderer_name = model_info.get_recommended_renderer_name(TEACHER_MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Teacher: {TEACHER_MODEL}, Renderer: {renderer_name}")

    training_client = service_client.create_lora_training_client(
        base_model=TEACHER_MODEL, rank=8
    )
    sampling_client = training_client.save_weights_and_get_sampling_client()

    dataset = load_gsm8k_train()

    # Handle resume
    already_done = set()
    existing_traces = []
    if args.resume and os.path.exists(args.output):
        existing_traces = load_traces_jsonl(args.output)
        already_done = {t["problem"] for t in existing_traces}
        logger.info(f"Resuming: {len(already_done)} problems already processed")

    new_traces = generate_traces(
        sampling_client=sampling_client,
        renderer=renderer,
        dataset=dataset,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        already_done=already_done,
    )

    all_traces = existing_traces + new_traces
    save_traces_jsonl(all_traces, args.output)
    logger.info(f"Saved {len(all_traces)} total traces to {args.output}")

    print(f"\n{'='*60}")
    print("TRACE GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total problems:     {len(dataset)}")
    print(f"  Traces generated:   {len(all_traces)}")
    print(f"  Yield:              {len(all_traces)/len(dataset)*100:.1f}%")
    avg_len = sum(t.get("trace_length", 0) for t in all_traces) / max(len(all_traces), 1)
    print(f"  Avg trace length:   {avg_len:.0f} chars")
    print(f"  Output:             {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
