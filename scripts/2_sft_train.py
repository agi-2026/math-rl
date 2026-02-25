"""
Phase 2: Supervised Fine-Tuning (SFT) via Tinker

Fine-tunes Qwen3-4B-Instruct on teacher-distilled reasoning traces
using LoRA through Tinker's API.

Usage:
    python scripts/2_sft_train.py --config configs/sft_config.yaml
    python scripts/2_sft_train.py --data data/traces_custom.jsonl
    python scripts/2_sft_train.py --data data/traces_public.jsonl --output-dir results/sft_public
"""

import argparse
import json
import logging
import math
import os
import time

import tinker
from tinker import types
import yaml
from tqdm import tqdm

from utils.data import load_traces_jsonl
from utils.tinker_helpers import (
    STUDENT_MODEL,
    build_sft_datum,
    get_renderer,
    get_service_client,
    get_tokenizer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "model_name": STUDENT_MODEL,
    "lora_rank": 64,
    "learning_rate": 2e-5,
    "epochs": 3,
    "batch_size": 4,
    "gradient_accumulation_steps": 8,
    "max_seq_length": 2048,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "save_every_epoch": True,
    "log_every": 10,
}


def load_config(config_path: str = None) -> dict:
    """Load config from YAML, falling back to defaults."""
    config = DEFAULT_CONFIG.copy()
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            overrides = yaml.safe_load(f)
        if overrides:
            config.update(overrides)
    return config


def main():
    parser = argparse.ArgumentParser(description="SFT training via Tinker")
    parser.add_argument("--config", default="configs/sft_config.yaml")
    parser.add_argument("--data", default="data/traces_custom.jsonl")
    parser.add_argument("--output-dir", default="results/sft_custom")
    parser.add_argument("--checkpoint-dir", default="checkpoints/sft")
    parser.add_argument("--tinker-url", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    # Load data
    traces = load_traces_jsonl(args.data)
    logger.info(f"Loaded {len(traces)} training traces from {args.data}")

    # Setup Tinker
    service_client = get_service_client(args.tinker_url)
    tokenizer = get_tokenizer(config["model_name"])
    renderer = get_renderer(config["model_name"], tokenizer)

    training_client = service_client.create_lora_training_client(
        base_model=config["model_name"],
        rank=config["lora_rank"],
    )

    # Build datums
    logger.info("Building SFT datums...")
    datums = []
    skipped = 0
    for trace in tqdm(traces, desc="Tokenizing"):
        datum = build_sft_datum(
            messages=trace["messages"],
            tokenizer=tokenizer,
            renderer=renderer,
            max_seq_length=config["max_seq_length"],
        )
        if datum is not None:
            datums.append(datum)
        else:
            skipped += 1

    logger.info(f"Built {len(datums)} datums ({skipped} skipped for length)")

    # Training loop
    batch_size = config["batch_size"]
    grad_accum = config["gradient_accumulation_steps"]
    effective_batch = batch_size * grad_accum
    num_epochs = config["epochs"]
    steps_per_epoch = math.ceil(len(datums) / effective_batch)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * config["warmup_ratio"])

    adam_params = types.AdamParams(
        learning_rate=config["learning_rate"],
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=config["weight_decay"],
    )

    logger.info(
        f"Training: {num_epochs} epochs, {steps_per_epoch} steps/epoch, "
        f"effective batch={effective_batch}, total steps={total_steps}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    metrics_log = []

    global_step = 0
    for epoch in range(num_epochs):
        logger.info(f"=== Epoch {epoch + 1}/{num_epochs} ===")

        # Shuffle data each epoch
        import random
        random.shuffle(datums)

        epoch_loss = 0.0
        epoch_batches = 0

        for step_in_epoch in range(steps_per_epoch):
            t_start = time.time()

            # Get batch
            start_idx = step_in_epoch * effective_batch
            end_idx = min(start_idx + effective_batch, len(datums))
            batch_datums = datums[start_idx:end_idx]

            if not batch_datums:
                continue

            # Forward-backward (Tinker handles grad accumulation internally
            # when we pass the full effective batch)
            fwd_bwd_future = training_client.forward_backward(
                batch_datums, loss_fn="cross_entropy"
            )

            # Optimizer step
            optim_future = training_client.optim_step(adam_params)

            fwd_bwd_result = fwd_bwd_future.result()
            optim_result = optim_future.result()

            step_time = time.time() - t_start
            global_step += 1

            # Log metrics
            step_metrics = {
                "epoch": epoch + 1,
                "step": global_step,
                "step_in_epoch": step_in_epoch + 1,
                "batch_size": len(batch_datums),
                "time": step_time,
            }
            if optim_result.metrics:
                step_metrics.update(optim_result.metrics)

            metrics_log.append(step_metrics)

            if global_step % config["log_every"] == 0:
                loss_str = step_metrics.get("loss", "N/A")
                logger.info(
                    f"  Step {global_step}/{total_steps} | "
                    f"loss={loss_str} | "
                    f"time={step_time:.1f}s"
                )

        # Save checkpoint after each epoch
        if config["save_every_epoch"]:
            ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch + 1}")
            training_client.save_state(ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    # Save final checkpoint
    final_path = os.path.join(args.checkpoint_dir, "final")
    training_client.save_state(final_path)
    logger.info(f"Saved final checkpoint: {final_path}")

    # Save training metrics
    with open(os.path.join(args.output_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics_log, f, indent=2)

    # Save config for reproducibility
    with open(os.path.join(args.output_dir, "config_used.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print("SFT TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Data:            {args.data} ({len(datums)} samples)")
    print(f"  Epochs:          {num_epochs}")
    print(f"  Total steps:     {total_steps}")
    print(f"  Checkpoint:      {final_path}")
    print(f"  Metrics:         {args.output_dir}/training_metrics.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
