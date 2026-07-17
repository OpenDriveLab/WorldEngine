#!/usr/bin/env python3
"""Merge isolated official NAVSIM PDM-score shard CSV files."""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence


def _clean_fieldnames(fieldnames: Sequence[str]) -> List[str]:
    return [name for name in fieldnames if name and not name.startswith("Unnamed:")]


def merge_score_files(input_paths: Sequence[Path], output_path: Path) -> Path:
    """Merge token rows and recompute the official average row."""

    if not input_paths:
        raise ValueError("at least one shard CSV is required")

    rows: List[Dict[str, str]] = []
    fieldnames: List[str] = []
    seen_tokens = set()
    for input_path in input_paths:
        with Path(input_path).open(newline="") as file:
            reader = csv.DictReader(file)
            current_fields = _clean_fieldnames(reader.fieldnames or [])
            if not fieldnames:
                fieldnames = current_fields
            elif current_fields != fieldnames:
                raise ValueError(f"incompatible shard columns in {input_path}")

            for raw_row in reader:
                row = {name: raw_row.get(name, "") for name in fieldnames}
                token = row.get("token")
                if token == "average":
                    continue
                if not token:
                    raise ValueError(f"missing token in {input_path}")
                if token in seen_tokens:
                    raise ValueError(f"duplicate token across score shards: {token}")
                seen_tokens.add(token)
                rows.append(row)

    if not rows or "token" not in fieldnames:
        raise ValueError("score shards contain no token rows")

    average = {name: "" for name in fieldnames}
    average["token"] = "average"
    if "valid" in fieldnames:
        average["valid"] = str(all(row["valid"].lower() == "true" for row in rows))

    for name in fieldnames:
        if name in ("token", "valid"):
            continue
        values = []
        for row in rows:
            value = row[name].strip()
            if not value:
                continue
            try:
                number = float(value)
            except ValueError:
                values = []
                break
            if not math.isnan(number):
                values.append(number)
        if values:
            average[name] = str(sum(values) / len(values))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(average)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = merge_score_files(args.inputs, args.output)
    print(f"Merged {len(args.inputs)} NAVSIM score shards at {output}")


if __name__ == "__main__":
    main()
