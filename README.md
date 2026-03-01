# Math Reasoning Post-Training: SFT + RL on Qwen3-4B

## Overview

A systematic study of post-training strategies for improving mathematical reasoning in a 4B-parameter language model. We compare supervised fine-tuning (SFT) via teacher distillation against reinforcement learning (GRPO) with verifiable math rewards, and ablate across training data difficulty, generation context length, and parameter efficiency (LoRA vs. full-parameter).

**Key result:** GRPO on hard problems achieves **90.2% on MATH-500** (+10.8pp over baseline) and **43.3% on AIME 2024** (+23.3pp). With majority voting (maj@16): **93.2% MATH-500**, **56.7% AIME 2024**.

| Component | Details |
|-----------|---------|
| **Base Model** | Qwen3-4B-Instruct-2507 |
| **Teacher Model** | Qwen3-235B-A22B-Instruct-2507 |
| **Training** | [Tinker](https://tinker.dev) (cloud LoRA) + TRL/vLLM (local H100) |
| **RL Algorithm** | GRPO (Group Relative Policy Optimization) |
| **Paper** | [`paper/main.tex`](paper/main.tex) — full methodology, ablations, and analysis |

---

## Results

### Ablation Table

All models evaluated with temperature=0.01 and greedy decoding. Best in **bold**.

| Model | MATH-500 | AIME 2024 | AIME 2025 | GSM8K | OlympiadBench |
|---|---|---|---|---|---|
| Baseline (Qwen3-4B) | 79.4% | 20.0% | 20.0% | 94.1% | 49.0% |
| SFT (custom traces) | 77.2% | 10.0% | 16.7% | 93.2% | 48.4% |
| Direct RL (GRPO v1) | 85.2% | 30.0% | 26.7% | 93.5% | 55.8% |
| SFT → RL | 85.4% | 23.3% | 20.0% | 93.4% | 54.2% |
| **GRPO v2** (hard, 4096 tok) | **90.2%** | **43.3%** | 26.7% | 93.8% | **63.6%** |
| GRPO v3 (continued) | 90.6% | 36.7% | **30.0%** | **94.0%** | 64.0% |
| Local full-param (H100) | 87.8% | 26.7% | 26.7% | 94.4% | 58.9% |
| Local LoRA step-50 (H100) | 87.2% | 30.0% | 26.7% | 93.9% | 60.0% |

### Majority Voting (GRPO v2, temp=0.7)

| Benchmark | Greedy | maj@16 | maj@64 | pass@64 |
|---|---|---|---|---|
| MATH-500 | 90.2% | **93.2%** | 93.2% | 97.2% |
| AIME 2024 | 43.3% | **56.7%** | 53.3% | 66.7% |
| AIME 2025 | 26.7% | **43.3%** | 40.0% | 53.3% |

---

## Key Findings

1. **RL >> SFT for small models.** GRPO gains +10.8pp on MATH-500; SFT *hurts* by -2.2pp. SFT warm-start provides no benefit over direct RL.
2. **Hard problems + long context = breakthrough.** Training on MATH Level 4-5 only with 4096-token completions yields +5.0pp over all-levels with 2048 tokens.
3. **Generation budget > parameterization.** LoRA with K=16, 4096 tokens outperforms full-param with K=8, 2048 tokens. When matched, LoRA converges in 50 steps vs. 998.
4. **maj@16 is optimal.** More samples (maj@64) actually hurts on AIME — diverse wrong answers outvote the correct one.
5. **Diminishing returns from continued training.** v3 adds only +0.4pp over v2 on MATH-500.

See [`paper/main.tex`](paper/main.tex) for detailed ablation analysis, GPU memory breakdowns, and discussion.

---

## Reproducing Results

### Prerequisites

```bash
pip install -r requirements.txt
export TINKER_API_KEY="your-api-key"  # for cloud training/eval

# For local training (H100 80GB)
pip install trl accelerate peft "vllm==0.12.0"
```

### Pipeline

```bash
# 1. Baseline evaluation
python scripts/0_baseline_eval.py

# 2. Teacher distillation (generate reasoning traces)
python scripts/1_generate_traces.py --dataset olympiadbench
python scripts/1_generate_traces.py --dataset math_hard

# 3. SFT (optional — doesn't improve results)
python scripts/2_sft_train.py --data data/traces_custom_all.jsonl

# 4. GRPO via Tinker (cloud)
python scripts/3_grpo_train.py --no-sft --dataset math

# 5. GRPO via TRL (local H100)
python scripts/7_grpo_local.py --num-epochs 2 --batch-size 8 --grad-accum 8 \
    --num-generations 8 --max-completion-length 2048
python scripts/9_grpo_local_lora.py --num-epochs 2 --lr 4e-5 --lora-rank 64 \
    --batch-size 8 --grad-accum 8 --num-generations 8 --max-completion-length 2048 --save-steps 50

# 6. Evaluation
python scripts/4_eval.py --experiments baseline,direct_rl \
    --benchmarks math500,aime2024,aime2025,gsm8k,olympiadbench
python scripts/8_eval_local.py --checkpoint /tmp/tinker-math/grpo_local_lora/merged-step50 \
    --benchmarks math500,aime2024,aime2025,gsm8k,olympiadbench
```

---

## Project Structure

```
howard/
├── paper/
│   └── main.tex                   # Research paper (full details)
├── configs/
│   ├── sft_config.yaml            # SFT hyperparameters
│   └── grpo_config.yaml           # GRPO hyperparameters
├── scripts/
│   ├── 0_baseline_eval.py         # Baseline evaluation
│   ├── 1_generate_traces.py       # Teacher distillation (235B → traces)
│   ├── 2_sft_train.py             # SFT via Tinker
│   ├── 3_grpo_train.py            # GRPO via Tinker
│   ├── 4_eval.py                  # Unified evaluation & ablation
│   ├── 5_prepare_openr1.py        # OpenR1-Math-220k data prep
│   ├── 6_combine_data.py          # Merge trace JSONL files
│   ├── 7_grpo_local.py            # Local full-param GRPO (TRL + vLLM)
│   ├── 8_eval_local.py            # Local evaluation (vLLM)
│   ├── 9_grpo_local_lora.py       # Local LoRA GRPO (TRL + PEFT + vLLM)
│   └── utils/
│       ├── data.py                # Data loaders
│       └── tinker_helpers.py      # Tinker API helpers
├── data/                          # Generated traces (gitignored)
└── results/
    └── ablation/
        └── ablation_results.json  # Full evaluation results
```

### Checkpoints

All stored at `/tmp/tinker-math/`:

| Checkpoint | Description |
|---|---|
| `grpo_v2` | **Best model** — LoRA GRPO on MATH Level 4-5, 4096 tokens |
| `grpo_v3` | Continued from v2 at lower LR |
| `grpo_local_fullparam/final` | Full-param GRPO (TRL, H100, 13h) |
| `grpo_local_lora/merged-step50` | Local LoRA GRPO step-50 (best local) |
| `sft_custom_A` | SFT Config A (baseline comparison) |
| `direct_rl` | GRPO v1, all MATH levels |
| `sft_then_rl` | GRPO from SFT checkpoint |
