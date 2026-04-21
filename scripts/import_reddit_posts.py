#!/usr/bin/env python3
"""Normalize Reddit-like records into a common post schema."""

from __future__ import annotations

import argparse
import bz2
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

try:
    import psycopg2
    from psycopg2.extras import Json, execute_values
except ImportError:  # pragma: no cover
    psycopg2 = None
    Json = None
    execute_values = None


TEXT_KEYS = ("text", "body", "selftext", "content", "comment", "tweet", "tokenized tweet")
SUBREDDIT_KEYS = ("subreddit", "community")
AUTHOR_KEYS = ("author", "user", "user_id", "username")
FLAIR_KEYS = ("author_flair_text", "flair")
TIME_KEYS = ("created_utc", "created_at", "timestamp", "datetime")
THREAD_KEYS = ("title", "parent_id", "link_id", "thread_context")
TOPIC_KEYS = ("topic", "label_topic")


@dataclass
class PostRow:
    post_id: str
    source_dataset: str
    subreddit: str
    user_id: str
    flair: str
    created_at: Optional[str]
    thread_context: str
    text: str
    topic: str
    raw_json: str


def pick_first(record: Dict[str, object], keys: Iterable[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def parse_time(record: Dict[str, object]) -> Optional[str]:
    raw = pick_first(record, TIME_KEYS)
    if not raw:
        return None
    if raw.isdigit():
        return datetime.fromtimestamp(int(raw), tz=timezone.utc).isoformat()
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
    except ValueError:
        return None


def choose_post_id(record: Dict[str, object], text: str) -> str:
    direct = pick_first(record, ("id", "post_id", "comment_id"))
    if direct:
        return direct
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]
    return f"auto_{digest}"


def normalize_row(record: Dict[str, object], source: str) -> Optional[PostRow]:
    text = pick_first(record, TEXT_KEYS)
    if text in {"[deleted]", "[removed]"} or not text:
        return None

    topic = pick_first(record, TOPIC_KEYS)
    subreddit = pick_first(record, SUBREDDIT_KEYS) or topic or "unknown_subreddit"
    user_id = pick_first(record, AUTHOR_KEYS)
    if not user_id:
        user_id = f"anon_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]}"

    context_parts = [pick_first(record, (k,)) for k in THREAD_KEYS]
    context = " | ".join(part for part in context_parts if part)

    return PostRow(
        post_id=choose_post_id(record, text),
        source_dataset=source,
        subreddit=subreddit,
        user_id=user_id,
        flair=pick_first(record, FLAIR_KEYS),
        created_at=parse_time(record),
        thread_context=context,
        text=text,
        topic=topic,
        raw_json=json.dumps(record, ensure_ascii=True),
    )


def iter_json_records(path: Path) -> Iterator[Dict[str, object]]:
    open_fn = bz2.open if path.suffix.lower() == ".bz2" else open
    with open_fn(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def iter_csv_records(path: Path) -> Iterator[Dict[str, object]]:
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield dict(row)


def iter_records(paths: List[Path]) -> Iterator[Dict[str, object]]:
    for path in paths:
        ext = path.suffix.lower()
        if ext == ".csv":
            yield from iter_csv_records(path)
        elif ext in {".jsonl", ".json", ".bz2"}:
            yield from iter_json_records(path)


def discover_input_files(input_dir: Path) -> List[Path]:
    allowed = {".csv", ".jsonl", ".json", ".bz2"}
    return sorted(
        path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in allowed
    )


def save_csv(rows: List[PostRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "post_id",
                "source_dataset",
                "subreddit",
                "user_id",
                "flair",
                "created_at",
                "thread_context",
                "text",
                "topic",
                "raw_json",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.post_id,
                    row.source_dataset,
                    row.subreddit,
                    row.user_id,
                    row.flair,
                    row.created_at or "",
                    row.thread_context,
                    row.text,
                    row.topic,
                    row.raw_json,
                ]
            )


def upsert_raw_posts(rows: List[PostRow], dsn: str) -> int:
    if psycopg2 is None or Json is None or execute_values is None:
        raise SystemExit("psycopg2-binary is required for --write-db.")
    if not rows:
        return 0

    values = [
        (
            row.post_id,
            row.source_dataset,
            row.subreddit,
            row.user_id,
            row.flair,
            row.created_at,
            row.thread_context,
            row.text,
            row.topic,
            Json(json.loads(row.raw_json)),
        )
        for row in rows
    ]

    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO raw_posts
                    (post_id, source_dataset, subreddit, user_id, flair, created_at, thread_context, text, topic, raw_json)
                VALUES %s
                ON CONFLICT (post_id) DO UPDATE
                SET subreddit = EXCLUDED.subreddit,
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
    return len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--source-dataset", type=str, required=True)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/ingestion/raw_posts.csv"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--write-db", action="store_true")
    parser.add_argument("--database-url", type=str, default=os.environ.get("DATABASE_URL", ""))
    args = parser.parse_args()

    files = discover_input_files(args.input_dir)
    if not files:
        raise SystemExit(f"No supported files found in {args.input_dir}")

    seen_ids = set()
    rows: List[PostRow] = []
    for record in iter_records(files):
        parsed = normalize_row(record, source=args.source_dataset)
        if not parsed:
            continue
        if parsed.post_id in seen_ids:
            continue
        seen_ids.add(parsed.post_id)
        rows.append(parsed)
        if args.limit and len(rows) >= args.limit:
            break

    save_csv(rows, args.output_csv)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")

    if args.write_db:
        if not args.database_url:
            raise SystemExit("DATABASE_URL is required when --write-db is set.")
        count = upsert_raw_posts(rows, args.database_url)
        print(f"Upserted {count} rows into raw_posts")

    missing_required = sum(
        1 for row in rows if not all([row.post_id, row.subreddit, row.user_id, row.text])
    )
    print(f"Rows missing required fields: {missing_required}")


if __name__ == "__main__":
    main()
