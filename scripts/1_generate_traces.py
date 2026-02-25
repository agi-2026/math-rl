"""
Phase 1: Teacher Distillation — Reasoning Trace Generation

Uses Qwen3-235B (via Tinker) to generate high-quality chain-of-thought
reasoning traces on GSM8K training problems. These traces are used for
SFT cold-start in Phase 2.

Usage:
    python scripts/1_generate_traces.py
    python scripts/1_generate_traces.py --num-samples 4 --output data/traces.jsonl
    python scripts/1_generate_traces.py --resume  # resume from partial output
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import Future

import tinker
from tqdm import tqdm

from utils.answer_extraction import (
    extract_gsm8k_final_answer,
    extract_number_from_response,
    is_equivalent,
)
from utils.data import build_chat_messages, load_gsm8k_train, save_traces_jsonl
from utils.tinker_helpers import (
    TEACHER_MODEL,
    get_renderer,
    get_service_client,
    get_tokenizer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert math tutor. Solve the following problem step by step. "
    "Show your reasoning clearly, then provide your final answer inside \\boxed{}."
)


def count_tokens_approx(text: str) -> int:
    """Rough token count (~4 chars per token for English)."""
    return len(text) // 4


def generate_traces(
    sampling_client,
    renderer,
    dataset,
    num_samples: int = 4,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    min_trace_length: int = 100,
    max_trace_length: int = 8000,
    already_done: set = None,
) -> list[dict]:
    """Generate reasoning traces for all problems in the dataset.

    Args:
        sampling_client: Tinker sampling client for the teacher model.
        renderer: Chat renderer for the teacher model.
        dataset: GSM8K training dataset.
        num_samples: Number of completions per problem (K).
        max_tokens: Max generation tokens per completion.
        temperature: Sampling temperature.
        min_trace_length: Min chars for a valid trace.
        max_trace_length: Max chars for a valid trace.
        already_done: Set of questions already processed (for resume).

    Returns:
        List of trace dicts.
    """
    if already_done is None:
        already_done = set()

    sampling_params = tinker.types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=renderer.get_stop_sequences() if renderer else [],
    )

    traces = []
    skipped = 0
    no_correct = 0

    # Process in batches to manage memory
    batch_size = 32
    problems = [
        row for row in dataset if row["question"] not in already_done
    ]
    logger.info(
        f"Generating traces for {len(problems)} problems "
        f"({len(already_done)} already done, {num_samples} samples each)"
    )

    for batch_start in range(0, len(problems), batch_size):
        batch = problems[batch_start : batch_start + batch_size]

        # Submit all sampling requests for this batch
        futures: list[tuple[Future, dict]] = []
        for row in batch:
            question = row["question"]
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
                num_samples=num_samples,
                sampling_params=sampling_params,
            )
            futures.append((future, row))

        # Collect and filter results
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
                if renderer:
                    parsed_message, _ = renderer.parse_response(sequence.tokens)
                    from tinker_cookbook import renderers as r
                    response_text = r.get_text_content(parsed_message)
                else:
                    response_text = sequence.text

                # Filter 1: Must have a parseable answer
                extracted = extract_number_from_response(response_text)
                if extracted is None:
                    continue

                # Filter 2: Answer must be correct
                if not is_equivalent(extracted, ground_truth):
                    continue

                # Filter 3: Must be reasonable length (not too short or long)
                char_len = len(response_text)
                if char_len < min_trace_length or char_len > max_trace_length:
                    continue

                # Filter 4: Must contain step-by-step reasoning
                # (heuristic: has multiple sentences/steps)
                if response_text.count(".") < 2 and response_text.count("\n") < 2:
                    skipped += 1
                    continue

                valid_traces.append({
                    "text": response_text,
                    "length": char_len,
                })

            if not valid_traces:
                no_correct += 1
                continue

            # Select the shortest correct trace (prefer concise reasoning)
            best = min(valid_traces, key=lambda t: t["length"])

            trace = {
                "messages": build_chat_messages(
                    problem=question,
                    system_prompt=SYSTEM_PROMPT,
                    response=best["text"],
                ),
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
        f"({no_correct} had no correct completions, {skipped} skipped for quality)"
    )
    return traces


def main():
    parser = argparse.ArgumentParser(description="Generate teacher distillation traces")
    parser.add_argument("--num-samples", type=int, default=4, help="Completions per problem (K)")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output", default="data/traces_custom.jsonl")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    parser.add_argument("--tinker-url", default=None)
    args = parser.parse_args()

    # Setup Tinker with teacher model
    service_client = get_service_client(args.tinker_url)
    tokenizer = get_tokenizer(TEACHER_MODEL)
    renderer = get_renderer(TEACHER_MODEL, tokenizer)

    # Create sampling client for teacher model
    training_client = service_client.create_lora_training_client(
        base_model=TEACHER_MODEL, rank=8
    )
    sampling_client = training_client.save_weights_and_get_sampling_client()

    # Load dataset
    dataset = load_gsm8k_train()

    # Handle resume
    already_done = set()
    existing_traces = []
    if args.resume and os.path.exists(args.output):
        from utils.data import load_traces_jsonl
        existing_traces = load_traces_jsonl(args.output)
        already_done = {t["problem"] for t in existing_traces}
        logger.info(f"Resuming: {len(already_done)} problems already processed")

    # Generate traces
    new_traces = generate_traces(
        sampling_client=sampling_client,
        renderer=renderer,
        dataset=dataset,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        already_done=already_done,
    )

    # Combine with existing traces
    all_traces = existing_traces + new_traces
    save_traces_jsonl(all_traces, args.output)
    logger.info(f"Saved {len(all_traces)} total traces to {args.output}")

    # Print statistics
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
