"""
Answer extraction utilities — thin wrappers around tinker_cookbook.

Re-exports the battle-tested grading/extraction from tinker_cookbook
and adds GSM8K-specific helpers.
"""

from tinker_cookbook.recipes.math_rl.math_grading import (
    extract_boxed,
    grade_answer,
    grade_answer_math_verify,
    normalize_answer,
    run_with_timeout_signal,
)
from tinker_cookbook.recipes.math_rl.math_env import (
    extract_gsm8k_final_answer,
    safe_grade,
)

__all__ = [
    "extract_boxed",
    "grade_answer",
    "grade_answer_math_verify",
    "normalize_answer",
    "extract_gsm8k_final_answer",
    "safe_grade",
    "run_with_timeout_signal",
]
