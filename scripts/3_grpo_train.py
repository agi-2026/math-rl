"""
Phase 3: Reinforcement Learning (GRPO-style) via Tinker

Applies Group Relative Policy Optimization to push beyond SFT ceiling.
Uses math correctness as verifiable reward — no learned reward model needed.

Follows the Tinker cookbook rl_loop.py pattern.

Usage:
    python scripts/3_grpo_train.py --config configs/grpo_config.yaml
    python scripts/3_grpo_train.py --sft-checkpoint checkpoints/sft/final
    python scripts/3_grpo_train.py --no-sft  # Direct RL ablation (skip SFT)
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import Future

import tinker
from tinker import types
from tinker.types.tensor_data import TensorData
import torch
import yaml
from tqdm import tqdm

from utils.answer_extraction import (
    extract_gsm8k_final_answer,
    extract_number_from_response,
    is_equivalent,
)
from utils.data import load_gsm8k_train
from utils.reward import compute_grpo_advantages, math_reward
from utils.tinker_helpers import (
    STUDENT_MODEL,
    build_rl_datum,
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

DEFAULT_CONFIG = {
    "model_name": STUDENT_MODEL,
    "lora_rank": 64,
    "learning_rate": 4e-5,
    "group_size": 8,
    "batch_size": 128,
    "max_tokens": 2048,
    "temperature": 0.7,
    "num_epochs": 2,
    "save_every": 20,
    "log_every": 1,
}


def load_config(config_path: str = None) -> dict:
    config = DEFAULT_CONFIG.copy()
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            overrides = yaml.safe_load(f)
        if overrides:
            config.update(overrides)
    return config


def main():
    parser = argparse.ArgumentParser(description="GRPO training via Tinker")
    parser.add_argument("--config", default="configs/grpo_config.yaml")
    parser.add_argument("--sft-checkpoint", default="checkpoints/sft/final")
    parser.add_argument("--no-sft", action="store_true", help="Skip SFT, train from base (ablation)")
    parser.add_argument("--output-dir", default="results/grpo")
    parser.add_argument("--checkpoint-dir", default="checkpoints/grpo")
    parser.add_argument("--tinker-url", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    # Setup Tinker
    service_client = get_service_client(args.tinker_url)
    tokenizer = get_tokenizer(config["model_name"])
    renderer = get_renderer(config["model_name"], tokenizer)

    # Create training client (load from SFT checkpoint or start fresh)
    if not args.no_sft and os.path.exists(args.sft_checkpoint):
        logger.info(f"Loading SFT checkpoint: {args.sft_checkpoint}")
        training_client = service_client.create_training_client_from_state_with_optimizer(
            args.sft_checkpoint
        )
    else:
        if args.no_sft:
            logger.info("Direct RL mode (no SFT cold-start)")
        else:
            logger.warning(
                f"SFT checkpoint not found at {args.sft_checkpoint}, starting from base model"
            )
        training_client = service_client.create_lora_training_client(
            base_model=config["model_name"],
            rank=config["lora_rank"],
        )

    # Load training data
    dataset = load_gsm8k_train()
    batch_size = config["batch_size"]
    n_batches = len(dataset) // batch_size

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config["max_tokens"],
        temperature=config["temperature"],
        stop=renderer.get_stop_sequences() if renderer else [],
    )

    adam_params = types.AdamParams(
        learning_rate=config["learning_rate"],
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    metrics_log = []

    logger.info(
        f"GRPO training: {config['num_epochs']} epochs, {n_batches} batches/epoch, "
        f"group_size={config['group_size']}, batch_size={batch_size}"
    )

    global_step = 0
    for epoch in range(config["num_epochs"]):
        logger.info(f"=== Epoch {epoch + 1}/{config['num_epochs']} ===")

        # Shuffle dataset each epoch
        shuffled_indices = list(range(len(dataset)))
        import random
        random.shuffle(shuffled_indices)

        for batch_idx in range(n_batches):
            t_start = time.time()

            # Save checkpoint periodically
            if config["save_every"] > 0 and global_step % config["save_every"] == 0 and global_step > 0:
                ckpt_path = os.path.join(args.checkpoint_dir, f"step_{global_step:06d}")
                training_client.save_state(ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")

            # Get batch
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(dataset))
            batch_indices = shuffled_indices[batch_start:batch_end]
            batch_rows = dataset.select(batch_indices)

            # Get sampling client with current weights
            sampling_client = training_client.save_weights_and_get_sampling_client()

            # Submit sampling requests for all problems in batch
            futures: list[Future] = []
            prompts: list[types.ModelInput] = []

            for question in batch_rows["question"]:
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
                    num_samples=config["group_size"],
                    sampling_params=sampling_params,
                )
                futures.append(future)
                prompts.append(model_input)

            # Collect results and compute advantages
            datums = []
            batch_rewards = []
            groups_skipped = 0

            for future, prompt, answer in zip(
                futures, prompts, batch_rows["answer"]
            ):
                ground_truth = extract_gsm8k_final_answer(answer)
                sample_result = future.result()

                # Collect rewards and tokens for the group
                group_rewards = []
                group_tokens = []
                group_logprobs = []

                for sequence in sample_result.sequences:
                    sampled_tokens = sequence.tokens
                    sampled_logprobs = sequence.logprobs
                    assert sampled_logprobs is not None

                    # Parse response text
                    if renderer:
                        parsed_message, _ = renderer.parse_response(sampled_tokens)
                        from tinker_cookbook import renderers as r
                        response_text = r.get_text_content(parsed_message)
                    else:
                        response_text = sequence.text

                    reward = math_reward(response_text, ground_truth)
                    group_rewards.append(reward)
                    group_tokens.append(sampled_tokens)
                    group_logprobs.append(sampled_logprobs)

                # Compute GRPO advantages
                advantages = compute_grpo_advantages(group_rewards)
                batch_rewards.append(sum(group_rewards) / len(group_rewards))

                # Skip if no gradient signal
                if all(a == 0.0 for a in advantages):
                    groups_skipped += 1
                    continue

                # Build datums for each sample in the group
                for tokens, logprobs, advantage in zip(
                    group_tokens, group_logprobs, advantages
                ):
                    datum = build_rl_datum(prompt, tokens, logprobs, advantage)
                    datums.append(datum)

            # Forward-backward + optimizer step
            if datums:
                fwd_bwd_future = training_client.forward_backward(
                    datums, loss_fn="importance_sampling"
                )
                optim_future = training_client.optim_step(adam_params)
                fwd_bwd_future.result()
                optim_result = optim_future.result()
            else:
                optim_result = None

            step_time = time.time() - t_start
            global_step += 1

            # Metrics
            mean_reward = sum(batch_rewards) / max(len(batch_rewards), 1)
            step_metrics = {
                "epoch": epoch + 1,
                "step": global_step,
                "batch_idx": batch_idx,
                "mean_reward": mean_reward,
                "groups_skipped": groups_skipped,
                "num_datums": len(datums),
                "time": step_time,
            }
            if optim_result and optim_result.metrics:
                step_metrics.update(optim_result.metrics)

            metrics_log.append(step_metrics)

            if global_step % config["log_every"] == 0:
                logger.info(
                    f"  Step {global_step} | reward={mean_reward:.3f} | "
                    f"datums={len(datums)} | skipped={groups_skipped} | "
                    f"time={step_time:.1f}s"
                )

    # Save final checkpoint
    final_path = os.path.join(args.checkpoint_dir, "final")
    training_client.save_state(final_path)
    logger.info(f"Saved final checkpoint: {final_path}")

    # Save metrics
    with open(os.path.join(args.output_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics_log, f, indent=2)

    with open(os.path.join(args.output_dir, "config_used.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print("GRPO TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total steps:     {global_step}")
    if metrics_log:
        final_reward = metrics_log[-1]["mean_reward"]
        print(f"  Final reward:    {final_reward:.3f}")
    print(f"  Checkpoint:      {final_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
