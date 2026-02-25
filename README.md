# DeepGrove Post-Training Assessment: Math Reasoning with SFT + RL

## Overview

This project demonstrates a complete post-training pipeline — from data generation through supervised fine-tuning (SFT) to reinforcement learning (GRPO) — applied to mathematical reasoning. The target is to measurably improve Qwen3-4B's math performance through a principled, reproducible pipeline that generalizes to other post-training domains (tool use, search, code).

**Base Model:** Qwen3-4B-Instruct-2507 (via Tinker)
**Teacher Model:** Qwen3-235B-A22B-Instruct (via Tinker, for trace distillation)
**Training Framework:** Tinker API (LoRA fine-tuning)
**Training Data:** GSM8K (8.5k train) + teacher-distilled traces + public comparison data
**Evaluation:** GSM8K test (1.3k) + MATH-500
**Hardware:** 1x NVIDIA H100 80GB

---

## Motivation

### Why Math Reasoning?

Math reasoning is the ideal domain for demonstrating post-training competence:

1. **Verifiable reward signal.** The answer is unambiguously correct or incorrect — no need for a learned reward model. This makes RL clean and interpretable.
2. **Well-studied baselines.** Qwen3-4B has known scores on GSM8K and MATH, so improvement is easy to measure.
3. **The pipeline transfers directly.** The same SFT → RL pipeline applies to tool use and search — the only difference is the reward function and action space.
4. **No infrastructure overhead.** No search backend, retrieval index, or API integration required. All time goes to the actual post-training work.

### Connection to DeepGrove's Mission

For on-device ternary models, post-training is critical because:
- Base models need instruction tuning to be usable
- Tool use (DeepGrove's target) requires the same SFT → RL pipeline demonstrated here
- Math's verifiable reward mirrors tool use's verifiable reward (tool execution succeeds or fails)
- The techniques explored here (STE-compatible SFT, GRPO, data curation) translate directly to ternary model post-training, with adjustments for higher learning rates, loss summation, and more training epochs (per Microsoft's BitNet findings)

---

## Methodology

### Phase 0: Baseline Evaluation

Before any training, establish Qwen3-4B-Instruct's baseline accuracy on both benchmarks to measure improvement.

```
Eval: Qwen3-4B-Instruct → GSM8K test (1,319 samples)
Eval: Qwen3-4B-Instruct → MATH-500 (500 samples)
```

### Phase 1: Reasoning Trace Generation

Generate high-quality chain-of-thought reasoning traces for SFT cold start.

**Approach:** Use Qwen3-235B (via Tinker) as teacher to generate high-quality traces on GSM8K train, then filter.

```
For each GSM8K train problem:
  1. Generate K=4 completions from Qwen3-235B with temperature=0.7
  2. Extract final numerical answer from each completion
  3. Compare against ground truth
  4. Keep traces where:
     - Answer is correct
     - Reasoning is step-by-step (not just restating the answer)
     - Length is within reasonable bounds (not too short, not pathologically long)
  5. Select the shortest correct trace (prefer concise reasoning)
```

**Why teacher distillation (Qwen3-235B → 4B):**
- Much higher quality traces than self-generation — 235B has stronger reasoning
- Consistent with recent literature (DeepSeek-R1 distillation, OpenThoughts)
- Tinker makes this seamless — same API for sampling and training

**Expected yield:** ~6-7k high-quality traces from 8.5k problems (some problems will have zero correct completions across K samples).

**Trace format:**
```
<system>You are a helpful math assistant. Solve the problem step by step.</system>
<user>{problem}</user>
<assistant><think>
Let me break this down step by step.
Step 1: ...
Step 2: ...
...
</think>

The answer is **{answer}**.</assistant>
```

### Phase 2: Supervised Fine-Tuning (Cold Start)

Fine-tune Qwen3-4B-Instruct on the filtered reasoning traces.

**Training configuration:**
```yaml
model: Qwen/Qwen3-4B
training:
  learning_rate: 2e-5
  lr_scheduler: cosine
  warmup_ratio: 0.05
  epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 8   # effective batch = 32
  max_seq_length: 2048
  bf16: true

loss:
  mask: system + user tokens masked, train on assistant tokens only

optimizer:
  type: AdamW
  weight_decay: 0.01

saving:
  strategy: epoch
  eval_steps: 100
```

**Framework:** Tinker API (LoRA fine-tuning via `forward_backward()` + `optim_step()`)

**Expected outcome:** Measurable improvement on GSM8K over base model, establishing a strong starting point for RL.

### Phase 3: Reinforcement Learning (GRPO)

Apply Group Relative Policy Optimization to push beyond SFT ceiling.

**Why GRPO:**
- No separate value model needed (saves VRAM on 1x H100)
- Groups multiple completions and uses relative ranking
- Well-suited for verifiable rewards (math)
- Same algorithm used by DeepSeek-R1, II-Search, and other recent work

**Reward function:**
```python
def math_reward(response: str, ground_truth: str) -> float:
    """
    Simple verifiable reward for math.
    """
    extracted = extract_final_answer(response)  # parse number from response

    if extracted is None:
        return -1.0    # failed to produce a parseable answer (format penalty)

    if is_equivalent(extracted, ground_truth):
        return 1.0     # correct answer

    return 0.0         # wrong answer, valid format
```

**GRPO configuration:**
```yaml
grpo:
  group_size: 8                # K completions per prompt
  learning_rate: 5e-6
  kl_coef: 0.05               # KL penalty against reference model
  max_new_tokens: 2048
  temperature: 0.7             # for generation during rollouts
  num_train_epochs: 2
  batch_size: 2
  gradient_accumulation_steps: 8

  # Reference model: the SFT checkpoint (frozen)
  ref_model: sft_checkpoint

  # Training data: GSM8K train (fresh problems, not seen traces)
  dataset: gsm8k_train
```

**Framework:** Tinker API (GRPO-style RL via `sample()` + `forward_backward()` with importance sampling loss)

**Key considerations:**
- VRAM budget: policy model (~8GB) + reference model (~8GB) + generation overhead (~16GB) + optimizer states (~16GB) ≈ ~48-50GB. Fits on H100 80GB.
- If tight, use LoRA for the policy model and load reference model in 8-bit.
- Monitor reward curves and KL divergence — if KL spikes, reduce learning rate.

### Phase 4: Evaluation

**Benchmarks:**
| Benchmark | Size | Difficulty | Purpose |
|-----------|------|-----------|---------|
| GSM8K test | 1,319 | Grade school math | In-distribution performance |
| MATH-500 | 500 | Competition math | Out-of-distribution generalization |

**Evaluation protocol:**
- Greedy decoding (temperature=0) for deterministic results
- Extract numerical answer using regex
- Compare against ground truth with numerical equivalence (handle float precision, fraction notation)
- Report accuracy (% correct) with 95% confidence intervals

**Ablation study:**

| Experiment | Purpose |
|-----------|---------|
| Base Qwen3-4B | Baseline (no post-training) |
| SFT (custom traces) | Measures teacher-distilled data contribution |
| SFT (public data) | Public data baseline for comparison |
| SFT + GRPO (custom) | Full pipeline |
| Direct GRPO (no SFT) | Tests whether cold start SFT is necessary |

This ablation addresses multiple questions: (1) Is SFT cold-start necessary before RL? (2) Does custom teacher-distilled data outperform public data? (3) How much does each stage contribute?

---

## Project Structure

```
howard/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore
├── configs/
│   ├── sft_config.yaml          # SFT hyperparameters
│   └── grpo_config.yaml         # GRPO hyperparameters
├── scripts/
│   ├── 0_baseline_eval.py       # Evaluate base model on GSM8K + MATH-500
│   ├── 1_generate_traces.py     # Teacher distillation (Qwen3-235B → traces)
│   ├── 2_sft_train.py           # SFT training via Tinker
│   ├── 3_grpo_train.py          # GRPO training via Tinker
│   ├── 4_eval.py                # Unified evaluation + ablation table
│   └── utils/
│       ├── answer_extraction.py # Parse numerical answers from responses
│       ├── reward.py            # Reward function + GRPO advantage computation
│       ├── data.py              # Data loading, formatting, public data loading
│       └── tinker_helpers.py    # Tinker client setup, datum builders
├── data/                        # Generated traces (gitignored)
├── results/
│   ├── baseline/                # Base model eval results
│   ├── sft_custom/              # SFT on custom data results
│   ├── sft_public/              # SFT on public data results
│   ├── grpo/                    # GRPO model eval results
│   └── ablation/                # Full ablation comparison
└── report.md                    # Final writeup with analysis
```

---

## Timeline

| Day | Focus | Deliverable |
|-----|-------|-------------|
| **Day 1** | Baseline eval, trace generation, SFT training | SFT checkpoint, baseline numbers |
| **Day 2** | GRPO setup, training, initial eval | GRPO checkpoint, preliminary results |
| **Day 3** | Ablation (direct RL), full eval suite, analysis | All experiment results |
| **Day 4** | Writeup, polish, reruns if needed | Final report with analysis |

---

## Expected Results

Based on published numbers for Qwen3-4B and the effectiveness of SFT + GRPO on math:

| Model | GSM8K (acc) | MATH-500 (acc) |
|-------|------------|----------------|
| Qwen3-4B-Instruct (baseline) | ~85-90% | ~55-65% |
| + SFT on GSM8K traces | ~90-93% | ~58-68% |
| + GRPO | ~92-95% | ~62-72% |

The biggest expected lift is on MATH-500 (OOD generalization), where RL teaches the model to explore and self-correct rather than follow memorized patterns.

---

## Discussion: From Math to Tool Use

### What transfers directly

The pipeline demonstrated here maps 1:1 to tool-use post-training:

| Math Pipeline | Tool Use Equivalent |
|--------------|-------------------|
| GSM8K problems | User queries requiring tool calls |
| Self-generated reasoning traces | Teacher-generated tool-use traces |
| SFT on traces | SFT on tool-call conversations |
| Math correctness reward | Tool execution success reward |
| GRPO | GRPO (same algorithm) |

### What changes for tool use

1. **Action space:** Instead of free-form text, the model generates structured tool calls (JSON). Constrained decoding may be needed.
2. **Multi-turn:** Tool use is inherently multi-turn (call → result → reason → call again). Math is typically single-turn.
3. **Reward complexity:** Tool use may need composite rewards (format correctness + task completion + efficiency), while math is binary.
4. **Environment:** Tool use RL needs a live execution environment. Math needs only answer comparison.

### What changes for ternary models

When applying this pipeline to DeepGrove's ternary models:

1. **SFT with STE:** Replace standard gradient descent with straight-through estimator. Weights stay ternary throughout.
2. **Higher learning rate:** Microsoft's BitNet findings suggest 2-5x higher LR for ternary SFT.
3. **Loss summation:** Use `reduction='sum'` instead of `reduction='mean'` for better convergence.
4. **More epochs:** Expect 2-3x more epochs due to STE gradient noise.
5. **Scale & norm tuning:** After SFT+RL, a final phase tuning only the continuous parameters (per-channel scales, LayerNorm) can provide cheap additional gains.
6. **GRPO considerations:** RL gradients are already high-variance; STE adds more noise. Larger group sizes (K=16+) and gradient clipping are recommended. Alternative: rejection sampling + SFT as a more stable substitute for online RL.

---

## Dependencies

```
tinker-api>=0.1.0     # Tinker training API
datasets              # HuggingFace datasets
numpy>=1.24.0
pyyaml>=6.0
tqdm>=4.65.0
sympy>=1.12           # Math answer grading
scipy>=1.11.0         # Confidence intervals
```

## Hardware Requirements

- **Minimum:** 1x NVIDIA H100 80GB
- **Training:** All via Tinker (cloud API, LoRA) — local GPU used for data processing only
- Tinker handles distributed training, VRAM management, and model serving

## Reproducing Results

```bash
# Install dependencies
pip install -r requirements.txt

# Step 0: Baseline evaluation
python scripts/0_baseline_eval.py

# Step 1: Generate teacher-distilled traces (Qwen3-235B → GSM8K)
python scripts/1_generate_traces.py --num-samples 4 --output data/traces_custom.jsonl

# Step 2: SFT on custom traces
python scripts/2_sft_train.py --config configs/sft_config.yaml --data data/traces_custom.jsonl

# Step 3: GRPO (from SFT checkpoint)
python scripts/3_grpo_train.py --config configs/grpo_config.yaml

# Step 3b: Direct RL ablation (no SFT)
python scripts/3_grpo_train.py --no-sft --checkpoint-dir checkpoints/direct_rl

# Step 4: Unified evaluation & ablation table
python scripts/4_eval.py --checkpoints baseline,sft_custom,sft_public,grpo,direct_rl
```
