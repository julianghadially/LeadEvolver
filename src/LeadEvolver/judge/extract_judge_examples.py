#!/usr/bin/env python3
"""
Script to extract judge examples from the CSV into a JSON file.

Usage:
    python -m src.LeadEvolver.judge.extract_judge_examples

This reads rows from data/github_users.csv where judge_example=True
and writes them to src/LeadEvolver/judge/judge_examples.json
"""

import csv
import json
from pathlib import Path


# Canonical conversion from CSV display format to internal snake_case
CSV_TO_INTERNAL = {
    "Strong fit": "strong_fit",
    "Weak fit": "weak_fit",
    "Not a fit": "not_a_fit",
}


def normalize_classification(value: str) -> str:
    """Convert CSV display format to internal snake_case format."""
    return CSV_TO_INTERNAL.get(value, value)


def extract_judge_examples(
    csv_path: Path,
    output_path: Path
) -> list[dict]:
    """
    Extract judge examples from CSV and save to JSON.

    Args:
        csv_path: Path to the github_users.csv file
        output_path: Path to write the JSON output

    Returns:
        List of extracted examples
    """
    examples = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Check if this row is marked as a judge example
            is_judge_example = row.get('judge_example', '').lower() in ['true', 'yes']

            if is_judge_example:
                example = {
                    "name": row.get('name') or row.get('username', ''),
                    "username": row.get('username', ''),
                    "context": row.get('context', ''),
                    "icp_match": normalize_classification(row.get('icp_match', '')),
                    "rationale": row.get('icp_match_rationale', '')
                }
                examples.append(example)

    # Write to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"Extracted {len(examples)} judge examples to {output_path}")
    for ex in examples:
        print(f"  - {ex['name']}: {ex['icp_match']}")

    return examples


def main():
    # Determine paths relative to project root
    # Script is at: src/LeadEvolver/judge/extract_judge_examples.py
    # Project root is 3 levels up from judge/
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent  # judge -> LeadEvolver -> src -> project root

    csv_path = project_root / "data" / "github_users.csv"
    output_path = script_dir / "judge_examples.json"

    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        print(f"Script dir: {script_dir}")
        print(f"Project root: {project_root}")
        return

    extract_judge_examples(csv_path, output_path)


if __name__ == "__main__":
    main()
