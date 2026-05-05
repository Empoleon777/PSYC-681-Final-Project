#!/usr/bin/env python3
"""Task 20: MITweet loader and split sanity check."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def split_name(text: str) -> str:
    h = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16) % 100
    if h < 70:
        return "train"
    if h < 85:
        return "val"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mitweet-csv", type=Path, default=Path("Data/MITweet.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/mitweet_split_summary.json"))
    args = parser.parse_args()

    rows = read_csv(args.mitweet_csv)
    split_counts = {"train": 0, "val": 0, "test": 0}
    missing_tweet = 0
    missing_ideology = 0
    for r in rows:
        text = (r.get("tweet") or "").strip()
        if not text:
            missing_tweet += 1
            continue
        split_counts[split_name(text)] += 1
        if all((r.get(f"I{i}") or "").strip() in {"", "-1"} for i in range(1, 13)):
            missing_ideology += 1

    payload = {
        "rows_total": len(rows),
        "split_counts": split_counts,
        "missing_tweet_rows": missing_tweet,
        "rows_without_any_ideology_label": missing_ideology,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

