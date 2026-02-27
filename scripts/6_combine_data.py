"""
Phase 6: Combine Data Sources for SFT v3

Merges custom teacher-distilled traces (OlympiadBench + MATH hard) with
filtered OpenR1-Math-220k data into a single training file.

Usage:
    python scripts/6_combine_data.py
    python scripts/6_combine_data.py --output data/traces_combined.jsonl
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils.data import load_traces_jsonl, save_traces_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Combine data sources for SFT")
    parser.add_argument("--output", default="data/traces_combined.jsonl")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=[
            "data/traces_olympiadbench.jsonl",
            "data/traces_math_hard.jsonl",
            "data/openr1_filtered.jsonl",
        ],
    )
    args = parser.parse_args()

    all_traces = []
    source_counts = {}

    for source_path in args.sources:
        if not os.path.exists(source_path):
            logger.warning(f"Source file not found, skipping: {source_path}")
            continue

        traces = load_traces_jsonl(source_path)
        source_name = os.path.basename(source_path).replace(".jsonl", "")
        source_counts[source_name] = len(traces)
        all_traces.extend(traces)
        logger.info(f"Loaded {len(traces)} traces from {source_path}")

    save_traces_jsonl(all_traces, args.output)

    # Summary
    print(f"\n{'='*60}")
    print("DATA COMBINATION SUMMARY")
    print(f"{'='*60}")
    for source, count in source_counts.items():
        pct = count / len(all_traces) * 100 if all_traces else 0
        print(f"  {source:<30} {count:>6} ({pct:.1f}%)")
    print(f"  {'─'*45}")
    print(f"  {'Total':<30} {len(all_traces):>6}")
    avg_len = sum(t.get("trace_length", 0) for t in all_traces) / max(len(all_traces), 1)
    print(f"  Avg trace length:             {avg_len:.0f} chars")
    print(f"  Output: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
