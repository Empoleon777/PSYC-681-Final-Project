#!/usr/bin/env python3
"""Fold per-annotator files into majority + soft annotation summaries."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List

try:
    import psycopg2
    from psycopg2.extras import Json, execute_values
except ImportError:  # pragma: no cover - runtime dependency
    psycopg2 = None
    Json = None
    execute_values = None


QUESTION_FIELDS = [
    "q01_relevance",
    "q02_target",
    "q03_stance",
    "q04_frame",
    "q05_moralization",
    "q06_identity_signaling",
    "q07_economic_direction",
    "q08_social_direction",
    "q09_intensity",
    "q10_ambiguity",
    "q11_confidence",
]

REQUIRED_QUESTION_FIELDS = list(QUESTION_FIELDS)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def entropy(values: List[str]) -> float:
    c = Counter(values)
    n = len(values)
    if n == 0:
        return 0.0
    probs = [v / n for v in c.values()]
    return -sum(p * math.log(p + 1e-12, 2) for p in probs)


def majority(values: List[str]) -> str:
    c = Counter(values)
    if not c:
        return ""
    top = c.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return "TIE"
    return top[0][0]


def soft_distribution(values: List[str]) -> Dict[str, float]:
    c = Counter(values)
    n = len(values)
    if n == 0:
        return {}
    return {k: v / n for k, v in sorted(c.items())}


def parse_evidence_spans(raw: str) -> List[Dict[str, object]]:
    text = (raw or "").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    out: List[Dict[str, object]] = []
    for item in payload:
        if isinstance(item, dict):
            out.append(item)
    return out


def upsert_db(
    dsn: str,
    annotations_rows: List[Dict[str, str]],
    aggregate_rows: List[Dict[str, str]],
) -> None:
    if psycopg2 is None or Json is None or execute_values is None:
        raise SystemExit("psycopg2-binary is required for --write-db.")
    ann_values = []
    for r in annotations_rows:
        ann_json = {k: r.get(k, "") for k in QUESTION_FIELDS + ["q12_notes"]}
        spans = r.get("evidence_spans_json") or "[]"
        ann_values.append(
            (
                r["annotation_id"],
                r["post_id"],
                r["annotator_id"],
                Json(ann_json),
                Json(json.loads(spans)),
            )
        )

    agg_values = [
        (
            r["post_id"],
            Json(json.loads(r["majority_json"])),
            Json(json.loads(r["soft_distribution_json"])),
            float(r["disagreement_entropy"]),
            int(r["n_annotations"]),
        )
        for r in aggregate_rows
    ]

    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO gold_annotations
                    (annotation_id, post_id, annotator_id, annotation_json, evidence_spans)
                VALUES %s
                ON CONFLICT (annotation_id, annotator_id) DO UPDATE
                SET annotation_json = EXCLUDED.annotation_json,
                    evidence_spans = EXCLUDED.evidence_spans,
                    created_at = NOW()
                """,
                ann_values,
                page_size=1000,
            )

            execute_values(
                cur,
                """
                INSERT INTO gold_aggregates
                    (post_id, majority_json, soft_distribution_json, disagreement_entropy, n_annotations)
                VALUES %s
                ON CONFLICT (post_id) DO UPDATE
                SET majority_json = EXCLUDED.majority_json,
                    soft_distribution_json = EXCLUDED.soft_distribution_json,
                    disagreement_entropy = EXCLUDED.disagreement_entropy,
                    n_annotations = EXCLUDED.n_annotations,
                    updated_at = NOW()
                """,
                agg_values,
                page_size=1000,
            )
        conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation-glob",
        type=str,
        default="Data/annotation/annotation_packet_annotator_*.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/annotation"))
    parser.add_argument("--write-db", action="store_true")
    parser.add_argument("--database-url", type=str, default=os.environ.get("DATABASE_URL", ""))
    parser.add_argument(
        "--require-complete-labels",
        action="store_true",
        help="Fail if any annotation row has blank required question fields.",
    )
    parser.add_argument(
        "--require-three-annotators",
        action="store_true",
        help="Fail if any post has fewer or more than 3 annotators.",
    )
    parser.add_argument(
        "--require-evidence-spans",
        action="store_true",
        help="Fail if any relevant row (q01_relevance=1) has empty/invalid evidence_spans_json.",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.annotation_glob))
    if not files:
        raise SystemExit(f"No annotation files matched: {args.annotation_glob}")

    all_rows: List[Dict[str, str]] = []
    for path in files:
        rows = read_csv(Path(path))
        fallback_annotator = Path(path).stem.split("_")[-1]
        for r in rows:
            if not r.get("annotator_id"):
                r["annotator_id"] = fallback_annotator
            all_rows.append(r)

    by_post: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in all_rows:
        by_post[r["post_id"]].append(r)

    aggregate_rows: List[Dict[str, str]] = []
    incomplete = 0
    rows_with_any_blank = 0
    field_blank_counts = {field: 0 for field in REQUIRED_QUESTION_FIELDS}
    posts_with_blank = 0
    relevant_rows = 0
    relevant_rows_with_valid_evidence = 0
    relevant_rows_missing_evidence = 0
    for post_id, rows in by_post.items():
        annotators = {r["annotator_id"] for r in rows if r.get("annotator_id")}
        if len(annotators) != 3:
            incomplete += 1

        post_has_blank = False
        for row in rows:
            row_blank = False
            for field in REQUIRED_QUESTION_FIELDS:
                if row.get(field, "").strip() == "":
                    field_blank_counts[field] += 1
                    row_blank = True
            if row_blank:
                rows_with_any_blank += 1
                post_has_blank = True

            if (row.get("q01_relevance", "").strip() == "1"):
                relevant_rows += 1
                spans = parse_evidence_spans(row.get("evidence_spans_json", ""))
                if spans:
                    relevant_rows_with_valid_evidence += 1
                else:
                    relevant_rows_missing_evidence += 1
        if post_has_blank:
            posts_with_blank += 1

        majority_json = {}
        soft_json = {}
        per_question_entropy = []

        for q in QUESTION_FIELDS:
            vals = [r.get(q, "").strip() for r in rows if r.get(q, "").strip() != ""]
            majority_json[q] = majority(vals)
            soft_json[q] = soft_distribution(vals)
            per_question_entropy.append(entropy(vals))

        aggregate_rows.append(
            {
                "post_id": post_id,
                "majority_json": json.dumps(majority_json, ensure_ascii=True),
                "soft_distribution_json": json.dumps(soft_json, ensure_ascii=True),
                "disagreement_entropy": str(round(mean(per_question_entropy), 6)),
                "n_annotations": str(len(annotators)),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "gold_annotations.csv", all_rows)
    write_csv(args.output_dir / "gold_aggregates.csv", aggregate_rows)

    summary = {
        "posts": len(by_post),
        "annotation_rows": len(all_rows),
        "posts_with_non_3_annotators": incomplete,
        "rows_with_any_blank_required_field": rows_with_any_blank,
        "posts_with_any_blank_required_field": posts_with_blank,
        "blank_counts_by_field": field_blank_counts,
        "relevant_rows": relevant_rows,
        "relevant_rows_with_valid_evidence": relevant_rows_with_valid_evidence,
        "relevant_rows_missing_evidence": relevant_rows_missing_evidence,
        "input_files": files,
    }
    (args.output_dir / "gold_validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    if args.write_db:
        if not args.database_url:
            raise SystemExit("DATABASE_URL is required when --write-db is set.")
        upsert_db(args.database_url, all_rows, aggregate_rows)

    if args.require_three_annotators and incomplete > 0:
        raise SystemExit(
            "Found posts without exactly 3 annotators. "
            "Task 14 requires 3 annotations per post."
        )

    if args.require_complete_labels and rows_with_any_blank > 0:
        raise SystemExit(
            "Found blank required labels. "
            "Fix annotation rows or regenerate packet before considering Task 14 complete."
        )

    if args.require_evidence_spans and relevant_rows_missing_evidence > 0:
        raise SystemExit(
            "Found relevant rows (q01_relevance=1) with empty or invalid evidence_spans_json. "
            "Fill evidence spans before considering Task 14 complete."
        )

    print(f"Posts aggregated: {len(by_post)}")
    print(f"Incomplete posts (not exactly 3 annotators): {incomplete}")
    print(f"Rows with blank required fields: {rows_with_any_blank}")
    print(
        f"Relevant rows missing evidence spans: "
        f"{relevant_rows_missing_evidence}/{relevant_rows}"
    )


if __name__ == "__main__":
    main()
