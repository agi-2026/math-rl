"""
Phase 3: Reinforcement Learning (GRPO-style) via Tinker

Follows tinker_cookbook/recipes/rl_loop.py pattern exactly.
Uses math correctness as verifiable reward.

Usage:
    python scripts/3_grpo_train.py --config configs/grpo_config.yaml
    python scripts/3_grpo_train.py --sft-checkpoint /tmp/tinker-math/sft
    python scripts/3_grpo_train.py --no-sft  # Direct RL ablation
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import Future

import datasets
import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
import yaml
from tqdm import tqdm

from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.recipes.math_rl.math_env import extract_gsm8k_final_answer, MathEnv
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils.tinker_helpers import STUDENT_MODEL, get_service_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)

DEFAULT_CONFIG = {
    "model_name": STUDENT_MODEL,
    "lora_rank": 64,
    "learning_rate": 4e-5,
    "group_size": 16,
    "batch_size": 128,
    "max_tokens": 2048,
    "save_every": 20,
}

QUESTION_SUFFIX = MathEnv.question_suffix()
CONVO_PREFIX = MathEnv.standard_fewshot_prefix()


def load_config(config_path: str | None = None) -> dict:
    config = DEFAULT_CONFIG.copy()
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            overrides = yaml.safe_load(f)
        if overrides:
            config.update(overrides)
    return config


def get_reward(response: str, answer: str) -> float:
    """Reward function matching rl_loop.py exactly."""
    try:
        given_answer = extract_boxed(response)
        ground_truth = extract_gsm8k_final_answer(answer)
        return 1.0 if grade_answer(given_answer, ground_truth) else 0.0
    except ValueError:
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="GRPO training via Tinker")
    parser.add_argument("--config", default="configs/grpo_config.yaml")
    parser.add_argument("--sft-checkpoint", default="/tmp/tinker-math/sft")
    parser.add_argument("--no-sft", action="store_true")
    parser.add_argument("--log-path", default="/tmp/tinker-math/grpo")
    parser.add_argument("--tinker-url", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    model_name = config["model_name"]
    service_client = get_service_client(args.tinker_url)
    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Model: {model_name}, Renderer: {renderer_name}")

    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=args.log_path,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    # Create or resume training client
    resume_info = checkpoint_utils.get_last_checkpoint(args.log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info["state_path"]
        )
        start_batch = resume_info.get("batch", 0)
        logger.info(f"Resuming from batch {start_batch}")
    elif not args.no_sft:
        # Try loading SFT checkpoint
        sft_resume = checkpoint_utils.get_last_checkpoint(args.sft_checkpoint)
        if sft_resume:
            training_client = service_client.create_training_client_from_state_with_optimizer(
                sft_resume["state_path"]
            )
            start_batch = 0
            logger.info(f"Loaded SFT checkpoint from {args.sft_checkpoint}")
        else:
            logger.warning(f"No SFT checkpoint at {args.sft_checkpoint}, starting from base")
            training_client = service_client.create_lora_training_client(
                base_model=model_name, rank=config["lora_rank"]
            )
            start_batch = 0
    else:
        logger.info("Direct RL mode (no SFT cold-start)")
        training_client = service_client.create_lora_training_client(
            base_model=model_name, rank=config["lora_rank"]
        )
        start_batch = 0

    # Load GSM8K training data
    dataset = datasets.load_dataset("openai/gsm8k", "main")
    assert isinstance(dataset, datasets.DatasetDict)
    train_dataset = dataset["train"]

    batch_size = config["batch_size"]
    group_size = config["group_size"]
    n_train_batches = len(train_dataset) // batch_size

    sampling_params = types.SamplingParams(
        max_tokens=config["max_tokens"],
        stop=renderer.get_stop_sequences(),
    )
    adam_params = types.AdamParams(
        learning_rate=config["learning_rate"], beta1=0.9, beta2=0.95, eps=1e-8
    )

    logger.info(f"Training for {n_train_batches} batches, group_size={group_size}")

    # Main training loop (follows rl_loop.py exactly)
    for batch_idx in range(start_batch, n_train_batches):
        t_start = time.time()
        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "optim/lr": config["learning_rate"],
            "progress/done_frac": (batch_idx + 1) / n_train_batches,
        }

        # Save checkpoint
        if config["save_every"] > 0 and batch_idx % config["save_every"] == 0 and batch_idx > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{batch_idx:06d}",
                log_path=args.log_path,
                kind="state",
                loop_state={"batch": batch_idx},
            )

        # Get batch
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(train_dataset))
        batch_rows = train_dataset.select(range(batch_start, batch_end))

        sampling_client = training_client.save_weights_and_get_sampling_client()

        # Sample responses for all problems in batch
        datums_D: list[types.Datum] = []
        rewards_P: list[float] = []
        futures_P: list[Future[types.SampleResponse]] = []
        prompts_P: list[types.ModelInput] = []

        for question in batch_rows["question"]:
            convo = [
                *CONVO_PREFIX,
                {"role": "user", "content": question + QUESTION_SUFFIX},
            ]
            model_input = renderer.build_generation_prompt(convo)
            future = sampling_client.sample(
                prompt=model_input,
                num_samples=group_size,
                sampling_params=sampling_params,
            )
            futures_P.append(future)
            prompts_P.append(model_input)

        # Collect results, compute advantages, build datums
        for future, prompt, answer in tqdm(
            zip(futures_P, prompts_P, batch_rows["answer"]),
            total=len(futures_P),
            desc=f"Batch {batch_idx}",
        ):
            sample_result = future.result()
            rewards_G: list[float] = []
            sampled_tokens_G_T: list[list[int]] = []
            logprobs_G_T: list[list[float]] = []

            for sequence in sample_result.sequences:
                sampled_tokens = sequence.tokens
                sampled_logprobs = sequence.logprobs
                assert sampled_logprobs is not None

                sampled_tokens_G_T.append(sampled_tokens)
                logprobs_G_T.append(sampled_logprobs)

                parsed_message, _ = renderer.parse_response(sampled_tokens)
                content = renderers.get_text_content(parsed_message)
                reward = get_reward(content, answer)
                rewards_G.append(reward)

            # GRPO advantage centering
            mean_reward = sum(rewards_G) / len(rewards_G)
            advantages_G = [reward - mean_reward for reward in rewards_G]
            rewards_P.append(mean_reward)

            if all(a == 0.0 for a in advantages_G):
                continue

            # Build datums (matches rl_loop.py datum construction)
            for sampled_tokens, logprobs, advantage in zip(
                sampled_tokens_G_T, logprobs_G_T, advantages_G
            ):
                ob_len = prompt.length - 1
                model_input = prompt.append(types.EncodedTextChunk(tokens=sampled_tokens[:-1]))
                target_tokens = [0] * ob_len + sampled_tokens
                padded_logprobs = [0.0] * ob_len + logprobs
                padded_advantages = [0.0] * ob_len + [advantage] * (model_input.length - ob_len)

                assert (
                    model_input.length
                    == len(target_tokens)
                    == len(padded_logprobs)
                    == len(padded_advantages)
                )

                datum = types.Datum(
                    model_input=model_input,
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                    },
                )
                datums_D.append(datum)

        # Training step
        fwd_bwd_future = training_client.forward_backward(datums_D, loss_fn="importance_sampling")
        optim_step_future = training_client.optim_step(adam_params)
        _fwd_bwd_result = fwd_bwd_future.result()
        optim_result = optim_step_future.result()

        if optim_result.metrics:
            metrics.update(optim_result.metrics)

        metrics["time/total"] = time.time() - t_start
        metrics["reward/total"] = sum(rewards_P) / len(rewards_P) if rewards_P else 0.0
        metrics["datums/count"] = len(datums_D)
        ml_logger.log_metrics(metrics, step=batch_idx)

        logger.info(
            f"  Batch {batch_idx}/{n_train_batches} | "
            f"reward={metrics['reward/total']:.3f} | "
            f"datums={len(datums_D)} | "
            f"time={metrics['time/total']:.1f}s"
        )

    # Save final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=args.log_path,
        kind="both",
        loop_state={"batch": n_train_batches},
    )
    ml_logger.close()
    logger.info("GRPO training completed")


if __name__ == "__main__":
    main()
