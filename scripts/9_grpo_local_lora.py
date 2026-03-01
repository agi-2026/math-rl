"""
Phase 9: Local LoRA GRPO Training on H100

Replicates the Tinker GRPO v2 setup (LoRA rank=64, group_size=16,
max_tokens=4096) but runs entirely locally using TRL's GRPOTrainer.
This validates that the Tinker LoRA results are reproducible with
standard open-source tooling.

Usage:
    python scripts/9_grpo_local_lora.py
    python scripts/9_grpo_local_lora.py --num-epochs 2 --lr 4e-5
"""

import argparse
import logging
import os
import sys

import datasets
import torch
from peft import LoraConfig
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
    """Load MATH Level 4-5 and format for GRPOTrainer."""
    raw = load_math_train_hard()
    logger.info(f"Loaded {len(raw)} hard MATH problems (Level 4-5)")

    rows = []
    for example in raw:
        question = example["problem"]
        try:
            ground_truth = extract_boxed(example["solution"])
        except ValueError:
            ground_truth = example["solution"]

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
    """Binary reward: 1.0 if the completion's \\boxed{} answer matches ground truth."""
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        if isinstance(completion, list):
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
    parser = argparse.ArgumentParser(description="Local LoRA GRPO — replicating Tinker v2")
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--num-generations", type=int, default=16)
    parser.add_argument("--max-completion-length", type=int, default=4096)
    parser.add_argument("--output-dir", default="/tmp/tinker-math/grpo_local_lora")
    parser.add_argument("--save-steps", type=int, default=20)
    args = parser.parse_args()

    logger.info(f"Config: epochs={args.num_epochs}, lr={args.lr}, lora_rank={args.lora_rank}, "
                f"batch={args.batch_size}, grad_accum={args.grad_accum}, "
                f"generations={args.num_generations}, max_completion={args.max_completion_length}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model (bf16, LoRA will be applied by GRPOTrainer)
    logger.info(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {total_params / 1e9:.1f}B parameters")

    # LoRA config — matching Tinker GRPO v2 setup
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,  # alpha=rank is standard
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    logger.info(f"LoRA config: rank={args.lora_rank}, targets={peft_config.target_modules}")

    # Build dataset
    train_dataset = build_dataset(tokenizer)

    # GRPOConfig — matching Tinker GRPO v2 hyperparameters
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

        # vLLM generation (colocate mode)
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.4,
        vllm_max_model_length=8192,
        vllm_enable_sleep_mode=True,

        # GRPO sampling — matching Tinker v2
        num_generations=args.num_generations,
        temperature=0.7,
        max_completion_length=args.max_completion_length,

        # Logging & saving
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",

        # No KL penalty (matches Tinker GRPO setup)
        beta=0.0,
        scale_rewards="group",
    )

    # Create trainer with LoRA
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=math_reward_fn,
        peft_config=peft_config,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params / 1e6:.1f}M "
                f"({trainable_params / total_params * 100:.2f}%)")

    logger.info("Starting LoRA GRPO training...")
    trainer.train()

    # Save final merged model (merge LoRA weights into base for easy inference)
    final_dir = os.path.join(args.output_dir, "final")
    logger.info(f"Saving merged model to {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Training complete. Model saved to {final_dir}")


if __name__ == "__main__":
    main()
