"""
Answer extraction utilities for math evaluation.

Handles parsing numerical answers from model responses and ground truth formats.
Supports GSM8K (#### format) and MATH (\\boxed{} format).
"""

import re
from typing import Optional


def extract_gsm8k_final_answer(solution: str) -> str:
    """Extract the final numerical answer from a GSM8K solution string.

    GSM8K ground truth format: reasoning text followed by #### <answer>
    """
    match = re.search(r"####\s*(.*?)$", solution, re.MULTILINE)
    if match:
        answer = match.group(1).strip()
        # Remove commas from numbers (e.g., "1,234" -> "1234")
        answer = answer.replace(",", "").replace("$", "")
        return answer
    raise ValueError(f"Could not extract GSM8K answer from: {solution[:200]}")


def extract_boxed(text: str) -> str:
    """Extract answer from \\boxed{...} in model output.

    Handles nested braces correctly.
    """
    # Find the last \boxed occurrence (models sometimes have intermediate boxed expressions)
    idx = text.rfind("\\boxed{")
    if idx == -1:
        raise ValueError(f"No \\boxed{{}} found in: {text[:200]}")

    # Parse with brace matching
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth != 0:
        raise ValueError(f"Unmatched braces in \\boxed expression: {text[idx:idx+100]}")

    return text[start : i - 1].strip()


def extract_number_from_response(text: str) -> Optional[str]:
    """Try multiple strategies to extract a numerical answer from model output.

    Priority:
    1. \\boxed{} expression
    2. "The answer is X" pattern
    3. "= X" at end of text
    4. Last standalone number in text
    """
    # Strategy 1: \boxed{}
    try:
        return extract_boxed(text)
    except ValueError:
        pass

    # Strategy 2: "The answer is X" / "the final answer is X"
    patterns = [
        r"[Tt]he\s+(?:final\s+)?answer\s+is[:\s]*\*?\*?([^\n\*]+)",
        r"[Aa]nswer[:\s]+\*?\*?([^\n\*]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            answer = match.group(1).strip().rstrip(".")
            answer = answer.replace(",", "").replace("$", "")
            if answer:
                return answer

    # Strategy 3: Last "= <number>" pattern
    match = re.findall(r"=\s*([+-]?\d[\d,]*\.?\d*)", text)
    if match:
        return match[-1].replace(",", "")

    # Strategy 4: Last standalone number
    match = re.findall(r"(?<!\w)([+-]?\d[\d,]*\.?\d*)(?!\w)", text)
    if match:
        return match[-1].replace(",", "")

    return None


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison.

    Handles: commas, dollar signs, percent signs, trailing periods,
    fraction notation, whitespace.
    """
    answer = answer.strip()
    # Remove common formatting
    answer = answer.replace(",", "")
    answer = answer.replace("$", "")
    answer = answer.replace("%", "")
    answer = answer.rstrip(".")
    answer = answer.strip()

    # Try to evaluate as a number
    try:
        val = float(answer)
        # Return as int if it's a whole number
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        pass

    return answer


def is_equivalent(answer: str, ground_truth: str) -> bool:
    """Check if two answers are mathematically equivalent.

    Handles numerical comparison with float tolerance, and string matching
    as fallback.
    """
    ans_norm = normalize_answer(answer)
    gt_norm = normalize_answer(ground_truth)

    # Direct string match
    if ans_norm == gt_norm:
        return True

    # Numerical comparison with tolerance
    try:
        ans_val = float(ans_norm)
        gt_val = float(gt_norm)
        return abs(ans_val - gt_val) < 1e-6
    except (ValueError, OverflowError):
        pass

    return False
