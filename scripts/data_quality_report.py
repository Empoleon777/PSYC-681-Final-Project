#!/usr/bin/env python3
"""Generate a before/after QA report for Reddit ingestion hygiene checks."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def quality_summary(rows: List[Dict[str, str]]) -> Dict[str, object]:
    seen_ids = set()
    seen_text = set()
    duplicate_post_ids = 0
    duplicate_texts = 0
    empty_text = 0
    missing_required = 0
    normalized_changed = 0
    retained_rows = 0

    for row in rows:
        post_id = (row.get("post_id") or "").strip()
        text_raw = row.get("text") or ""
        text = normalize_text(text_raw)
        if text != text_raw:
            normalized_changed += 1
        if not text or text in {"[deleted]", "[removed]"}:
            empty_text += 1
            continue
        if not post_id or not (row.get("subreddit") or "").strip() or not (row.get("user_id") or "").strip():
            missing_required += 1
            continue
        if post_id in seen_ids:
            duplicate_post_ids += 1
            continue
        text_key = text.lower()
        if text_key in seen_text:
            duplicate_texts += 1
            continue
        seen_ids.add(post_id)
        seen_text.add(text_key)
        retained_rows += 1

    return {
        "rows_total": len(rows),
        "rows_retained": retained_rows,
        "rows_removed": len(rows) - retained_rows,
        "duplicate_post_id_rows": duplicate_post_ids,
        "duplicate_text_rows": duplicate_texts,
        "empty_or_deleted_text_rows": empty_text,
        "missing_required_rows": missing_required,
        "rows_with_normalization_change": normalized_changed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/ingestion/data_quality_report.json"))
    args = parser.parse_args()

    rows = read_csv(args.input_csv)
    summary = quality_summary(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

