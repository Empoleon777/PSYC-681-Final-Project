#!/usr/bin/env python3
"""Upsert normalized raw_posts CSV rows into PostgreSQL."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List

try:
    import psycopg2
    from psycopg2.extras import Json, execute_values
except ImportError:  # pragma: no cover
    psycopg2 = None
    Json = None
    execute_values = None


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--database-url", type=str, default=os.environ.get("DATABASE_URL", ""))
    args = parser.parse_args()

    if psycopg2 is None or Json is None or execute_values is None:
        raise SystemExit("psycopg2-binary is required.")
    if not args.database_url:
        raise SystemExit("DATABASE_URL is required.")

    rows = read_csv(args.input_csv)
    values = []
    for r in rows:
        raw_json = r.get("raw_json") or "{}"
        try:
            parsed_raw = json.loads(raw_json)
        except json.JSONDecodeError:
            parsed_raw = {"raw_json_parse_error": True}
        values.append(
            (
                r.get("post_id", ""),
                r.get("source_dataset", "csv_import"),
                r.get("subreddit", ""),
                r.get("user_id", ""),
                r.get("flair", ""),
                r.get("created_at", None) or None,
                r.get("thread_context", ""),
                r.get("text", ""),
                r.get("topic", ""),
                Json(parsed_raw),
            )
        )

    with psycopg2.connect(args.database_url) as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO raw_posts
                    (post_id, source_dataset, subreddit, user_id, flair, created_at, thread_context, text, topic, raw_json)
                VALUES %s
                ON CONFLICT (post_id) DO UPDATE
                SET source_dataset = EXCLUDED.source_dataset,
                    subreddit = EXCLUDED.subreddit,
                    user_id = EXCLUDED.user_id,
                    flair = EXCLUDED.flair,
                    created_at = EXCLUDED.created_at,
                    thread_context = EXCLUDED.thread_context,
                    text = EXCLUDED.text,
                    topic = EXCLUDED.topic,
                    raw_json = EXCLUDED.raw_json
                """,
                values,
                page_size=1000,
            )
        conn.commit()
    print(f"Upserted {len(values)} rows into raw_posts from {args.input_csv}")


if __name__ == "__main__":
    main()

