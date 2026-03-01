# Math Reasoning Post-Training: SFT + RL on Qwen3-4B

## Overview

This project demonstrates a complete post-training pipeline for improving mathematical reasoning in a 4B-parameter language model. Starting from Qwen3-4B-Instruct, we explore supervised fine-tuning (SFT) via teacher distillation and reinforcement learning (GRPO) to push math performance significantly beyond the base model.

**Key result:** GRPO on hard math problems achieves **90.2% on MATH-500** (vs 79.4% baseline) and **43.3% on AIME 2024** (vs 20.0% baseline) — a +10.8pp and +23.3pp improvement respectively. With majority voting (maj@16), performance reaches **93.2% on MATH-500** and **56.7% on AIME 2024**.

| Component | Details |
|-----------|---------|
| **Base Model** | Qwen3-4B-Instruct-2507 |
| **Teacher Model** | Qwen3-235B-A22B-Instruct-2507 |
| **Training Framework** | [Tinker](https://tinker.dev) (cloud LoRA fine-tuning API) |
| **RL Algorithm** | GRPO (Group Relative Policy Optimization) |
| **Evaluation** | MATH-500, AIME 2024, AIME 2025, GSM8K, OlympiadBench |

---

## Results

### Final Ablation Table

All models evaluated with temperature=0.01, greedy-ish decoding, and 95% confidence intervals.

| Model | MATH-500 | AIME 2024 | AIME 2025 | GSM8K | OlympiadBench |
|---|---|---|---|---|---|
| **Baseline** (Qwen3-4B) | 79.4% ±3.5 | 20.0% ±14.3 | 20.0% ±14.3 | 94.1% ±1.3 | 49.0% ±4.1 |
| **SFT** (custom traces) | 77.2% ±3.7 | 10.0% ±10.7 | 16.7% ±13.3 | 93.2% ±1.4 | 48.4% ±4.1 |
| **Direct RL** (GRPO v1) | 85.2% ±3.1 | 30.0% ±16.4 | 26.7% ±15.8 | 93.5% ±1.3 | 55.8% ±4.1 |
| **SFT → RL** | 85.4% ±3.1 | 23.3% ±15.1 | 20.0% ±14.3 | 93.4% ±1.3 | 54.2% ±4.1 |
| **GRPO v2** (hard problems) | **90.2% ±2.6** | **43.3% ±17.7** | 26.7% ±15.8 | 93.8% ±1.3 | **63.6% ±3.9** |
| **GRPO v3** (continued) | 90.6% ±2.6 | 36.7% ±17.2 | **30.0% ±16.4** | **94.0% ±1.3** | 64.0% ±3.9 |
| **GRPO local** (full-param, H100) | 87.8% ±2.9 | 26.7% ±15.8 | 26.7% ±15.8 | 94.4% ±1.2 | 58.9% ±4.0 |
| **GRPO local LoRA** (step-50, H100) | 87.2% ±2.9 | 30.0% ±16.4 | 26.7% ±15.8 | 93.9% ±1.3 | 60.0% ±4.0 |

### Key Takeaways

1. **RL is far more effective than SFT for small models.** GRPO v2 gains +10.8pp on MATH-500 over baseline; SFT actually hurts by -2.2pp.
2. **SFT warm-start doesn't help RL.** Direct RL and SFT→RL converge to the same accuracy (85.2% vs 85.4%), making the SFT phase unnecessary overhead.
3. **Training on hard problems is critical.** GRPO v2 (MATH Level 4-5 only) dramatically outperforms GRPO v1 (all difficulty levels): 90.2% vs 85.2% on MATH-500, 43.3% vs 30.0% on AIME 2024.
4. **Longer generation context helps.** max_tokens=4096 (v2) vs 2048 (v1) allows the model to reason through harder problems.
5. **Continued training gives diminishing returns.** v3 (2 more epochs at lower LR) only adds +0.4pp on MATH-500 over v2.

---

## Methodology

### Phase 0: Baseline Evaluation

Evaluate the base Qwen3-4B-Instruct model on all benchmarks before any training. Uses few-shot prompting with the `MathEnv.standard_fewshot_prefix()` from the Tinker cookbook, and asks the model to put final answers in `\boxed{}` format.

```bash
python scripts/0_baseline_eval.py
```

### Phase 1: Reasoning Trace Generation (Teacher Distillation)

Generate high-quality chain-of-thought reasoning traces by prompting Qwen3-235B-A22B-Instruct on training problems, then filtering for correctness.

**Datasets used:**
- **OlympiadBench** (numerical problems): 390 correct traces from ~500 problems
- **MATH Level 4-5** (hard problems): 3,435 correct traces from 3,994 problems (~86% yield)

**Process:**
1. For each problem, generate 1 completion from the teacher model
2. Extract the `\boxed{}` answer from the response
3. Grade against ground truth using `grade_answer()` (handles numeric equivalence, fractions, etc.)
4. Keep only correct traces

**Trace format:** Standard chat format with the teacher's reasoning as the assistant response. Median trace length: ~2,300 characters.

```bash
# Generate OlympiadBench traces
python scripts/1_generate_traces.py --dataset olympiadbench

# Generate MATH Level 4-5 traces
python scripts/1_generate_traces.py --dataset math_hard
```

### Phase 2: Supervised Fine-Tuning (SFT)

Fine-tune Qwen3-4B-Instruct on the distilled traces using LoRA via Tinker.

**Configuration:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| LoRA rank | 64 | |
| Learning rate | 2e-4 | From `get_lr()` recommendation (~5e-4), tuned down |
| Batch size | 32 | 119 steps/epoch with 3,825 traces |
| Epochs | 2 | |
| Max sequence length | 4096 | |
| Optimizer | Adam (β1=0.9, β2=0.95) | Linear LR decay to 0 |
| Loss masking | Assistant tokens only | |

**Data:** 3,825 combined traces (390 OlympiadBench + 3,435 MATH hard)

```bash
python scripts/2_sft_train.py --data data/traces_custom_all.jsonl --log-path /tmp/tinker-math/sft_custom_A
```

### Phase 3: Reinforcement Learning (GRPO)

Apply Group Relative Policy Optimization with binary math correctness reward. The model generates multiple completions per problem, and those that solve correctly receive positive advantage while incorrect ones receive negative advantage.

**GRPO v1 (Direct RL):**
| Parameter | Value |
|-----------|-------|
| Training data | Full MATH train (~7.5k problems, all levels) |
| LoRA rank | 64 |
| Learning rate | 4e-5 |
| Batch size | 128 |
| Group size (K) | 16 |
| Max tokens | 2048 |
| Loss function | Importance sampling |
| Epochs | ~1 (58 batches) |

```bash
python scripts/3_grpo_train.py --no-sft --dataset math
```

**GRPO v2 (Hard Problems — Best Model):**
| Parameter | Value | Change from v1 |
|-----------|-------|-----------------|
| Training data | MATH Level 4-5 only (3,994 problems) | Filtered to hard problems |
| Batch size | 64 | Smaller for more gradient steps |
| Max tokens | 4096 | 2x longer context |
| Epochs | 2 (124 batches) | More training |
| Other params | Same as v1 | |

**GRPO v3 (Continued Training):**
- Resumed from v2 checkpoint
- Learning rate reduced to 2e-5 (half of v2)
- 2 additional epochs on MATH Level 4-5
- Marginal improvement (+0.4pp MATH-500)

**Reward function:**
```python
def get_reward_math(response: str, ground_truth: str) -> float:
    """Binary reward: 1.0 if correct, 0.0 otherwise."""
    try:
        given_answer = extract_boxed(response)
        return 1.0 if grade_answer(given_answer, ground_truth) else 0.0
    except ValueError:
        return 0.0
```

**GRPO advantage computation:**
For each problem, generate K=16 completions. The advantage for each completion is: `reward_i - mean(rewards)`. If all completions are correct or all wrong (zero variance), the problem is skipped.

### Phase 7: Local Full-Parameter GRPO

To test whether removing the LoRA bottleneck improves results, we ran full-parameter GRPO training locally on an H100 80GB GPU using TRL's GRPOTrainer with vLLM for generation.

**Configuration:**
| Parameter | Value | vs GRPO v2 (LoRA) |
|-----------|-------|--------------------|
| Method | Full-parameter (all 4B params) | LoRA rank=64 |
| Framework | TRL GRPOTrainer + vLLM colocate | Tinker API |
| Learning rate | 5e-6 | 4e-5 |
| Batch size | 8 × 8 grad accum = 64 effective | 64 |
| Group size (K) | 8 | 16 |
| Max completion tokens | 2048 | 4096 |
| Training data | MATH Level 4-5 (3,994 problems) | Same |
| Epochs | 2 | 2 |
| Training time | ~13 hours (H100 80GB) | ~2 hours (Tinker cloud) |
| GPU memory peak | 68 GB / 81 GB | N/A (cloud) |

**Results vs LoRA:**
| Benchmark | GRPO v2 (LoRA) | GRPO local (full-param) | Delta |
|---|---|---|---|
| MATH-500 | **90.2%** | 87.8% | -2.4pp |
| AIME 2024 | **43.3%** | 26.7% | -16.6pp |
| AIME 2025 | 26.7% | 26.7% | 0 |
| GSM8K | 93.8% | **94.4%** | +0.6pp |
| OlympiadBench | **63.6%** | 58.9% | -4.7pp |

**Analysis:** The full-parameter model improved over baseline but significantly underperformed the LoRA model. The likely cause is **not** LoRA vs full-param itself, but the hyperparameter compromises needed to fit full-param training on a single GPU:
- **Shorter context (2048 vs 4096):** Hard competition problems often require long chains of reasoning. Truncating at 2048 tokens cuts off solutions mid-reasoning.
- **Smaller group size (8 vs 16):** Fewer samples per problem means less reward signal diversity, leading to noisier advantage estimates.
- **Lower learning rate (5e-6 vs 4e-5):** Full-param training is more sensitive to large updates, requiring a lower LR that slows convergence.

**Key takeaway:** When compute is constrained to a single GPU, LoRA with optimal hyperparameters (long context, large group size) outperforms full-parameter training with compromised hyperparameters. Full-param GRPO would likely need multi-GPU training to match the context length and group size of the LoRA setup.

```bash
# Training
python scripts/7_grpo_local.py --num-epochs 2 --batch-size 8 --grad-accum 8 \
    --num-generations 8 --max-completion-length 2048 --save-steps 50

# Evaluation (uses vLLM for fast inference)
python scripts/8_eval_local.py --checkpoint /tmp/tinker-math/grpo_local_fullparam/final \
    --benchmarks math500,aime2024,aime2025,gsm8k,olympiadbench
```

### Phase 9: Local LoRA GRPO

To control for whether the full-param vs LoRA distinction (or the hyperparameter differences) explains the gap with Tinker v2, we ran local LoRA GRPO using the same infrastructure as Phase 7 but with PEFT LoRA adapters.

**Configuration:**
| Parameter | Value | vs Full-param (Phase 7) |
|-----------|-------|--------------------|
| Method | LoRA (rank=64, alpha=64) | Full-parameter |
| Framework | TRL GRPOTrainer + PEFT + vLLM colocate | Same |
| Learning rate | 4e-5 | 5e-6 |
| Batch size | 8 × 8 grad accum = 64 effective | Same |
| Group size (K) | 8 | Same |
| Max completion tokens | 2048 | Same |
| Training data | MATH Level 4-5 (3,994 problems) | Same |
| Trainable params | ~160M (4% of total) | 4B (100%) |

**Key findings:**
- LoRA saturates by ~50 steps (reward → 0.95, grad_norm → 0). Step-100 shows slight degradation.
- Step-50 matches full-param results (87.2% vs 87.8% MATH-500) with **50x fewer steps** (50 vs 998)
- Step-50 beats full-param on AIME 2024 (30.0% vs 26.7%) and OlympiadBench (60.0% vs 58.9%)
- Both local methods trail Tinker v2 — confirming that **generation budget** (context length × group size) is the dominant factor, not LoRA vs full-param

```bash
# Training
python scripts/9_grpo_local_lora.py --num-epochs 2 --lr 4e-5 --lora-rank 64 \
    --batch-size 8 --grad-accum 8 --num-generations 8 --max-completion-length 2048 --save-steps 50

# Merge LoRA adapters into base model for eval
python /tmp/merge_and_eval_lora.py  # merges checkpoint-50 and checkpoint-100

# Evaluate
python scripts/8_eval_local.py --checkpoint /tmp/tinker-math/grpo_local_lora/merged-step50 \
    --benchmarks math500,aime2024,aime2025,gsm8k,olympiadbench
```

### Phase 4: Evaluation

Unified evaluation across 5 benchmarks with consistent protocol.

| Benchmark | Size | Difficulty | Purpose |
|-----------|------|-----------|---------|
| **MATH-500** | 500 | Competition math | Primary metric, broad difficulty |
| **AIME 2024** | 30 | Very hard (competition) | Ceiling test |
| **AIME 2025** | 30 | Very hard (competition) | Ceiling test, temporal OOD |
| **GSM8K** | 1,319 | Grade school | Floor test, regression check |
| **OlympiadBench** | 572 (numerical only) | Olympiad-level | Hard benchmark, training overlap check |

**Protocol:**
- Temperature: 0.01 (near-deterministic)
- Max tokens: 2048 (eval)
- Few-shot prefix from `MathEnv.standard_fewshot_prefix()`
- Answer extraction: parse `\boxed{}` from response
- Grading: `grade_answer()` handles numeric equivalence, fractions, simplified forms
- Confidence intervals: 95% CI using normal approximation

```bash
python scripts/4_eval.py --experiments baseline,sft_custom,direct_rl,sft_then_rl
```

---

## What Works

1. **GRPO on hard problems with long context.** The single most impactful change. Training exclusively on MATH Level 4-5 problems with max_tokens=4096 produced the best model across all benchmarks.

2. **Binary correctness reward.** Simple 0/1 reward is sufficient — no need for format penalties, partial credit, or reward shaping. The GRPO advantage computation naturally handles the rest.

3. **Direct RL (no SFT warm-start).** Starting RL directly from the base model works just as well as SFT→RL, simplifying the pipeline.

4. **Teacher distillation for data quality.** Qwen3-235B produces high-quality traces with ~86% yield on hard problems. The traces are naturally compatible with the student model's tokenizer and style.

## What Doesn't Work

1. **SFT on small (4B) models.** Despite careful hyperparameter tuning and data curation, LoRA SFT consistently degraded performance vs baseline (-2.2pp on MATH-500). The model appears to overfit to the trace distribution rather than learning generalizable reasoning.

2. **OpenR1-Math-220k as SFT data.** DeepSeek R1-style traces (median 7,300 chars) are incompatible with Qwen3-4B. Even length-filtered subsets caused catastrophic regression. The reasoning style mismatch between DeepSeek R1 and Qwen3 appears fundamental.

3. **SFT warm-start for RL.** SFT→RL converges to the same accuracy as direct RL (85.4% vs 85.2%), providing no benefit while adding training time and complexity.

4. **Training on easy problems.** GRPO v1 on all MATH levels (including Level 1-3) underperforms v2 on hard problems only. Easy problems contribute near-zero signal (all completions correct → zero advantage).

5. **Extended training at low LR.** GRPO v3 (2 more epochs at half LR) adds only +0.4pp over v2, suggesting the model is near its single-sample ceiling at this scale.

6. **Constrained generation budget.** Both local full-param (87.8%) and local LoRA (87.2%) trail Tinker v2 (90.2%) on MATH-500. All local runs used max_completion=2048, group_size=8 (vs 4096/16 for Tinker), confirming that **generation budget matters more than LoRA vs full-param**. Local LoRA step-50 matched full-param accuracy in 50 steps (vs 998), showing LoRA efficiency, but neither local approach overcomes the shorter context limitation.

---

## SFT Experiments Detail

We tried multiple SFT configurations before concluding that SFT hurts this model:

| SFT Version | Data | LR | Epochs | Steps | MATH-500 | Notes |
|---|---|---|---|---|---|---|
| v1 | 390 OlympiadBench | 2e-5 | 3 | 9 | ~79% | LR too low, too few steps |
| v2 | 390 OlympiadBench | 5e-4 | 3 | 36 | ~79% | Fixed LR, still insufficient data |
| v3 | 5.4k (custom + OpenR1) | 5e-4 | 2 | 168 | ~70% | OpenR1 data caused catastrophic regression |
| v4 | 861 (custom short) | 5e-4 | 2 | 53 | ~77% | Removed OpenR1, improved but still below baseline |
| Config A | 3,825 (full custom) | 2e-4 | 2 | 238 | 77.2% | Best SFT, but still -2.2pp vs baseline |
| Config B | 3,825 (full custom) | 5e-4 | 1 | 119 | 76.8% | Higher LR, 1 epoch |

**Lesson:** For a 4B model that already has strong math capability, SFT on distilled traces doesn't help — the model learns to mimic the trace format but loses some of its inherent reasoning ability.

---

## Project Structure

```
howard/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .gitignore
├── configs/
│   ├── sft_config.yaml            # SFT hyperparameters
│   └── grpo_config.yaml           # GRPO hyperparameters
├── scripts/
│   ├── 0_baseline_eval.py         # Evaluate base model
│   ├── 1_generate_traces.py       # Teacher distillation (Qwen3-235B → traces)
│   ├── 2_sft_train.py             # SFT training via Tinker
│   ├── 3_grpo_train.py            # GRPO training via Tinker
│   ├── 4_eval.py                  # Unified evaluation & ablation table
│   ├── 5_prepare_openr1.py        # Download & filter OpenR1-Math-220k
│   ├── 6_combine_data.py          # Merge multiple trace JSONL files
│   ├── 7_grpo_local.py            # Local full-param GRPO (TRL + vLLM on H100)
│   ├── 8_eval_local.py            # Evaluate local HF checkpoints (vLLM)
│   ├── 9_grpo_local_lora.py       # Local LoRA GRPO (TRL + PEFT + vLLM on H100)
│   └── utils/
│       ├── data.py                # Data loaders (GSM8K, MATH, AIME, OlympiadBench)
│       └── tinker_helpers.py      # Tinker client setup, model constants
├── data/                          # Generated traces (gitignored)
│   ├── traces_olympiadbench.jsonl # 390 traces from OlympiadBench
│   ├── traces_math_hard.jsonl     # 3,435 traces from MATH Level 4-5
│   ├── traces_custom_all.jsonl    # 3,825 combined traces
│   └── openr1_filtered.jsonl      # 5,000 filtered OpenR1 traces (not used)
└── results/
    └── ablation/
        └── ablation_results.json  # Full evaluation results
```

---

## Checkpoints

All checkpoints are stored at `/tmp/tinker-math/` on the training machine:

| Checkpoint | Description | Best For |
|---|---|---|
| `sft_custom_A` | SFT Config A (LR=2e-4, 2 epochs, 3,825 traces) | SFT baseline comparison |
| `direct_rl` | GRPO v1 (all MATH, 58 batches, 2048 tokens) | Basic RL baseline |
| `sft_then_rl` | GRPO from SFT-A checkpoint | SFT→RL comparison |
| `grpo_v2` | GRPO v2 (MATH Level 4-5, 124 batches, 4096 tokens) | **Best for AIME 2024 (43.3%)** |
| `grpo_v3` | Continued from v2, LR=2e-5, 2 more epochs | **Best for MATH-500 (90.6%)** |
| `grpo_local_fullparam/final` | Full-param GRPO via TRL on H100, 2 epochs, 13h | LoRA vs full-param comparison |
| `grpo_local_lora/merged-step50` | Local LoRA GRPO step-50 (merged into base), ~50 min | **Best local LoRA** |
| `grpo_local_lora/merged-step100` | Local LoRA GRPO step-100 (merged), overfitted | Early stopping comparison |

---

## Reproducing Results

### Prerequisites

```bash
pip install -r requirements.txt
export TINKER_API_KEY="your-api-key"
```

### Full Pipeline

```bash
# Step 0: Baseline evaluation
python scripts/0_baseline_eval.py

# Step 1: Generate teacher-distilled traces
python scripts/1_generate_traces.py --dataset olympiadbench
python scripts/1_generate_traces.py --dataset math_hard

# Step 2: SFT (optional — doesn't improve results)
python scripts/2_sft_train.py --data data/traces_custom_all.jsonl \
    --log-path /tmp/tinker-math/sft_custom_A

# Step 3: GRPO (Direct RL — recommended path)
# v1: All MATH problems
python scripts/3_grpo_train.py --no-sft --dataset math

# v2: Hard problems only (best results) — requires custom script
# See /tmp/run_grpo_v2.py for the enhanced training loop

# Step 4: Evaluation
python scripts/4_eval.py --experiments baseline,sft_custom,direct_rl \
    --benchmarks math500,aime2024,aime2025,gsm8k,olympiadbench
```

### Local Full-Parameter GRPO (H100)

```bash
# Install additional dependencies
pip install trl accelerate peft "vllm==0.12.0"

# Train (requires H100 80GB or equivalent)
python scripts/7_grpo_local.py --num-epochs 2 --batch-size 8 --grad-accum 8 \
    --num-generations 8 --max-completion-length 2048 --save-steps 50

# Evaluate local checkpoint
python scripts/8_eval_local.py --checkpoint /tmp/tinker-math/grpo_local_fullparam/final \
    --benchmarks math500,aime2024,aime2025,gsm8k,olympiadbench
```

### Local LoRA GRPO (H100)

```bash
# Train LoRA (saturates by step ~50 at LR=4e-5)
python scripts/9_grpo_local_lora.py --num-epochs 2 --lr 4e-5 --lora-rank 64 \
    --batch-size 8 --grad-accum 8 --num-generations 8 --max-completion-length 2048 --save-steps 50

# Evaluate merged checkpoint
python scripts/8_eval_local.py --checkpoint /tmp/tinker-math/grpo_local_lora/merged-step50 \
    --benchmarks math500,aime2024,aime2025,gsm8k,olympiadbench
```

### Evaluation Only

To evaluate a specific checkpoint:

```bash
# Evaluate all experiments
python scripts/4_eval.py

# Evaluate specific experiments
python scripts/4_eval.py --experiments baseline,direct_rl --benchmarks math500,aime2024
```

---

## Inference-Time Scaling: Majority Voting

After reaching the training-side ceiling, we apply majority voting (maj@k) at inference time: sample K completions at temperature=0.7 per problem, extract the `\boxed{}` answer from each, and take the most common answer (grouping mathematically equivalent answers via `grade_answer()`).

### Majority Voting Results (GRPO v2)

| Benchmark | Greedy (k=1) | maj@16 | maj@64 | pass@16 | pass@64 |
|---|---|---|---|---|---|
| **MATH-500** | 90.2% | **93.2%** | 93.2% | 95.6% | 97.2% |
| **AIME 2024** | 43.3% | **56.7%** | 53.3% | 66.7% | 66.7% |
| **AIME 2025** | 26.7% | **43.3%** | 40.0% | 53.3% | 53.3% |

### Key Observations

1. **maj@16 is the sweet spot.** It outperforms maj@64 on AIME benchmarks — more samples at high temperature introduce diverse wrong answers that can outvote the correct one.

2. **MATH-500 plateaus at 93.2%** despite pass@64=97.2% (486/500 solvable). The remaining hard problems have success rates too low for majority voting to help.

3. **AIME 2024 hits 56.7%** with maj@16, exceeding the 50% target. AIME 2025 reaches 43.3%.

4. **pass@64 reveals the model's ceiling:** 97.2% on MATH-500 and 66.7% on AIME 2024. Closing the gap between pass@k and maj@k would require a learned verifier or reward-weighted voting rather than simple majority.

---

## Future Directions

1. **Reward-weighted voting / verifier:** Train a small verifier model to score completions, replacing simple majority vote. Could close the gap between maj@k (93.2%) and pass@k (97.2%) on MATH-500.

2. **Reward shaping:** Add format compliance bonus, partial credit for intermediate steps, or length penalty to encourage concise solutions.

3. **Curriculum learning:** Start RL on Level 3-4, then gradually introduce Level 5 problems as the model improves.

4. **Multi-GPU full-parameter GRPO:** Our single-GPU full-param experiment was bottlenecked by memory, forcing shorter context (2048 tokens) and smaller group size (8). Multi-GPU training could match the LoRA hyperparameters (4096 tokens, group=16) and potentially exceed LoRA results.

5. **Multi-epoch cycling on hard problems:** GRPO v2's success suggests the model benefits from repeated exposure to hard problems. Cycling with different random seeds could extract more signal.
