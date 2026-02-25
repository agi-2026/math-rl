"""
Phase 2: Supervised Fine-Tuning (SFT) via Tinker

Fine-tunes Qwen3-4B on teacher-distilled reasoning traces using LoRA.
Follows the tinker_cookbook/recipes/sl_loop.py pattern exactly.

Usage:
    python scripts/2_sft_train.py --config configs/sft_config.yaml
    python scripts/2_sft_train.py --data data/traces_custom.jsonl
"""

import argparse
import json
import logging
import os
import sys
import time

import tinker
from tinker import types
import yaml
from tqdm import tqdm

from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils.data import load_traces_jsonl
from scripts.utils.tinker_helpers import STUDENT_MODEL, get_service_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)

DEFAULT_CONFIG = {
    "model_name": STUDENT_MODEL,
    "lora_rank": 64,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "batch_size": 128,
    "max_length": 4096,
    "save_every": 20,
    "train_on_what": "all_assistant_messages",
}


def load_config(config_path: str | None = None) -> dict:
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
    parser.add_argument("--log-path", default="/tmp/tinker-math/sft")
    parser.add_argument("--tinker-url", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    # Load data
    traces = load_traces_jsonl(args.data)
    logger.info(f"Loaded {len(traces)} training traces from {args.data}")

    # Setup Tinker (follows sl_loop.py pattern)
    model_name = config["model_name"]
    service_client = get_service_client(args.tinker_url)
    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Model: {model_name}, Renderer: {renderer_name}")

    train_on_what = renderers.TrainOnWhat(config["train_on_what"])

    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=args.log_path,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    # Check for resume
    resume_info = checkpoint_utils.get_last_checkpoint(args.log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info["state_path"]
        )
        start_step = resume_info.get("batch", 0)
        logger.info(f"Resuming from step {start_step}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=model_name, rank=config["lora_rank"]
        )
        start_step = 0

    # Build datums from traces (using conversation_to_datum from tinker_cookbook)
    # Each trace has a "messages" field with [system, user, assistant] messages
    logger.info("Building datums...")
    all_messages = [t["messages"] for t in traces]

    batch_size = config["batch_size"]
    max_length = config["max_length"]
    n_batches_per_epoch = len(all_messages) // batch_size
    total_steps = n_batches_per_epoch * config["num_epochs"]

    logger.info(
        f"Training: {config['num_epochs']} epochs, {n_batches_per_epoch} batches/epoch, "
        f"batch_size={batch_size}, total_steps={total_steps}"
    )

    import random
    global_step = start_step

    for epoch in range(config["num_epochs"]):
        logger.info(f"=== Epoch {epoch + 1}/{config['num_epochs']} ===")
        random.shuffle(all_messages)

        for batch_idx in range(n_batches_per_epoch):
            if global_step < start_step:
                global_step += 1
                continue

            t_start = time.time()
            metrics: dict[str, float] = {}

            # Save checkpoint
            if config["save_every"] > 0 and global_step % config["save_every"] == 0 and global_step > 0:
                checkpoint_utils.save_checkpoint(
                    training_client=training_client,
                    name=f"{global_step:06d}",
                    log_path=args.log_path,
                    kind="state",
                    loop_state={"batch": global_step},
                )

            # Linear LR decay
            lr_mult = max(0.0, 1.0 - global_step / total_steps)
            current_lr = config["learning_rate"] * lr_mult
            adam_params = tinker.AdamParams(
                learning_rate=current_lr, beta1=0.9, beta2=0.95, eps=1e-8
            )

            # Get batch and convert to datums
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(all_messages))
            batch_messages = all_messages[batch_start:batch_end]

            batch = [
                conversation_to_datum(msgs, renderer, max_length, train_on_what)
                for msgs in batch_messages
            ]

            # Training step (forward_backward + optim_step pipelined)
            fwd_bwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
            optim_future = training_client.optim_step(adam_params)

            fwd_bwd_result = fwd_bwd_future.result()
            optim_result = optim_future.result()

            if optim_result.metrics:
                metrics.update(optim_result.metrics)

            # Compute NLL
            train_logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
            train_weights = [d.loss_fn_inputs["weights"] for d in batch]
            train_nll = compute_mean_nll(train_logprobs, train_weights)

            metrics.update(
                epoch=epoch + 1,
                step=global_step,
                num_sequences=len(batch),
                num_tokens=sum(d.model_input.length for d in batch),
                learning_rate=current_lr,
                train_mean_nll=train_nll,
                progress=global_step / total_steps,
                time_total=time.time() - t_start,
            )
            ml_logger.log_metrics(metrics=metrics, step=global_step)

            if global_step % 10 == 0:
                logger.info(
                    f"  Step {global_step}/{total_steps} | "
                    f"NLL={train_nll:.4f} | LR={current_lr:.2e} | "
                    f"time={time.time()-t_start:.1f}s"
                )

            global_step += 1

    # Save final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=args.log_path,
        kind="both",
        loop_state={"batch": global_step},
    )

    ml_logger.close()
    logger.info("SFT training completed")

    print(f"\n{'='*60}")
    print("SFT TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Data:            {args.data} ({len(traces)} traces)")
    print(f"  Epochs:          {config['num_epochs']}")
    print(f"  Total steps:     {global_step}")
    print(f"  Checkpoint:      {args.log_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
