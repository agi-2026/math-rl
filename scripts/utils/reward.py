"""
Reward functions for GRPO math reasoning training.
"""

from scripts.utils.answer_extraction import (
    extract_number_from_response,
    is_equivalent,
)


def math_reward(response: str, ground_truth: str) -> float:
    """Compute reward for a math response.

    Args:
        response: Full model response text.
        ground_truth: Ground truth answer string.

    Returns:
        1.0 if correct, 0.0 if wrong but parseable, -1.0 if unparseable.
    """
    extracted = extract_number_from_response(response)

    if extracted is None:
        return -1.0  # format penalty — failed to produce parseable answer

    if is_equivalent(extracted, ground_truth):
        return 1.0  # correct

    return 0.0  # wrong answer, valid format


def compute_grpo_advantages(rewards: list[float]) -> list[float]:
    """Compute GRPO-style advantages from a group of rewards.

    Advantage = reward - mean(rewards) for the group.
    Returns zero advantages if all rewards are identical (no gradient signal).
    """
    mean_reward = sum(rewards) / len(rewards)
    advantages = [r - mean_reward for r in rewards]

    # If all advantages are zero, no learning signal
    if all(a == 0.0 for a in advantages):
        return [0.0] * len(rewards)

    return advantages
