"""
Reward functions for GRPO math reasoning training.

Uses tinker_cookbook's grading utilities for robust answer checking.
"""

from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.recipes.math_rl.math_env import extract_gsm8k_final_answer


def get_reward_gsm8k(response: str, answer: str) -> float:
    """Compute reward for a GSM8K response.

    Matches the pattern from tinker_cookbook/recipes/rl_loop.py.

    Args:
        response: Model response text (should contain \\boxed{...}).
        answer: Raw GSM8K answer field (contains #### prefix).

    Returns:
        1.0 if correct, 0.0 otherwise.
    """
    try:
        given_answer = extract_boxed(response)
        ground_truth = extract_gsm8k_final_answer(answer)
        return 1.0 if grade_answer(given_answer, ground_truth) else 0.0
    except ValueError:
        return 0.0


def get_reward_math(response: str, ground_truth: str) -> float:
    """Compute reward for a MATH-500 response.

    Args:
        response: Model response text (should contain \\boxed{...}).
        ground_truth: Ground truth answer string.

    Returns:
        1.0 if correct, 0.0 otherwise.
    """
    try:
        given_answer = extract_boxed(response)
        return 1.0 if grade_answer(given_answer, ground_truth) else 0.0
    except ValueError:
        return 0.0
