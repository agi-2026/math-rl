"""
Phase 7: Local Full-Parameter GRPO Training on H100

Trains Qwen3-4B-Instruct with full-parameter GRPO (no LoRA) using TRL's
GRPOTrainer + vLLM for fast generation. Compares against Tinker LoRA results.

Usage:
    python scripts/7_grpo_local.py
    python scripts/7_grpo_local.py --num-epochs 1 --lr 1e-6
"""

import argparse
import logging
import os
import sys

import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils.data import load_math_train_hard

from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.recipes.math_rl.math_env import MathEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
QUESTION_SUFFIX = MathEnv.question_suffix()
FEWSHOT_PREFIX = MathEnv.standard_fewshot_prefix()


def build_dataset(tokenizer) -> datasets.Dataset:
    """Load MATH Level 4-5 and format for GRPOTrainer.

    GRPOTrainer expects each row to have:
      - "prompt": list of chat messages (conversational format)
      - any extra columns get passed to the reward function as kwargs
    """
    raw = load_math_train_hard()
    logger.info(f"Loaded {len(raw)} hard MATH problems (Level 4-5)")

    rows = []
    for example in raw:
        question = example["problem"]
        # Extract ground truth answer from solution's \boxed{}
        try:
            ground_truth = extract_boxed(example["solution"])
        except ValueError:
            ground_truth = example["solution"]

        # Build conversational prompt with few-shot prefix
        prompt = [
            *FEWSHOT_PREFIX,
            {"role": "user", "content": question + QUESTION_SUFFIX},
        ]

        rows.append({
            "prompt": prompt,
            "ground_truth": ground_truth,
        })

    ds = datasets.Dataset.from_list(rows)
    logger.info(f"Dataset prepared: {len(ds)} problems")
    return ds


def math_reward_fn(completions, ground_truth, **kwargs):
    """Binary reward: 1.0 if the completion's \\boxed{} answer matches ground truth.

    Args:
        completions: list of completion strings (one per sample)
        ground_truth: list of ground truth answer strings (repeated per group)
    Returns:
        list of float rewards
    """
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        # completion is a list of messages; extract the assistant's text
        if isinstance(completion, list):
            # Conversational format: [{"role": "assistant", "content": "..."}]
            text = completion[-1]["content"] if completion else ""
        else:
            text = str(completion)

        try:
            given_answer = extract_boxed(text)
            reward = 1.0 if grade_answer(given_answer, gt) else 0.0
        except (ValueError, Exception):
            reward = 0.0
        rewards.append(reward)
    return rewards


def main():
    parser = argparse.ArgumentParser(description="Local full-parameter GRPO on H100")
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--num-generations", type=int, default=16)
    parser.add_argument("--max-completion-length", type=int, default=4096)
    parser.add_argument("--output-dir", default="/tmp/tinker-math/grpo_local_fullparam")
    parser.add_argument("--save-steps", type=int, default=20)
    args = parser.parse_args()

    logger.info(f"Config: epochs={args.num_epochs}, lr={args.lr}, batch={args.batch_size}, "
                f"grad_accum={args.grad_accum}, generations={args.num_generations}, "
                f"max_completion={args.max_completion_length}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model (full-parameter, bf16)
    logger.info(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
    )
    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B parameters")

    # Build dataset
    train_dataset = build_dataset(tokenizer)

    # GRPOConfig
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,

        # vLLM generation (colocate mode — runs in same process)
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        vllm_max_model_length=8192,  # Prompt (~500 tokens) + completion (4096 tokens)
        vllm_enable_sleep_mode=True,

        # GRPO sampling
        num_generations=args.num_generations,
        temperature=0.7,
        max_completion_length=args.max_completion_length,

        # Logging & saving
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",

        # No KL penalty (matches our Tinker GRPO setup)
        beta=0.0,
        scale_rewards="group",
    )

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=math_reward_fn,
    )

    logger.info("Starting GRPO training...")
    trainer.train()

    # Save final model
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
    logger.info(f"Training complete. Model saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
