#!/usr/bin/env python3
"""Create a stratified annotation batch from raw posts + weak labels."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


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
    "q12_notes",
    "evidence_spans_json",
]


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


def quartiles(values: List[float]) -> Tuple[float, float, float]:
    vals = sorted(values)
    if not vals:
        return (0.25, 0.5, 0.75)
    n = len(vals)
    q1 = vals[int(0.25 * (n - 1))]
    q2 = vals[int(0.50 * (n - 1))]
    q3 = vals[int(0.75 * (n - 1))]
    return q1, q2, q3


def quartile_name(v: float, bounds: Tuple[float, float, float]) -> str:
    q1, q2, q3 = bounds
    if v <= q1:
        return "Q1"
    if v <= q2:
        return "Q2"
    if v <= q3:
        return "Q3"
    return "Q4"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-posts-csv", type=Path, default=Path("outputs/ingestion/raw_posts.csv"))
    parser.add_argument("--weak-labels-csv", type=Path, default=Path("outputs/weak_labels/post_weak_labels.csv"))
    parser.add_argument("--sample-size", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("Data/annotation"))
    args = parser.parse_args()

    random.seed(args.seed)

    raw_rows = read_csv(args.raw_posts_csv)
    weak_rows = read_csv(args.weak_labels_csv)
    weak_index = {r["post_id"]: r for r in weak_rows}

    merged: List[Dict[str, str]] = []
    q_values: List[float] = []

    for r in raw_rows:
        w = weak_index.get(r["post_id"])
        if not w:
            continue
        q = float(w.get("q_score", 0.0))
        q_values.append(q)
        merged.append(
            {
                "post_id": r["post_id"],
                "subreddit": r.get("subreddit", ""),
                "topic": r.get("topic") or r.get("subreddit", ""),
                "text": r.get("text", ""),
                "q_score": str(q),
            }
        )

    if not merged:
        raise SystemExit("No overlapping rows between raw posts and weak labels.")

    bounds = quartiles(q_values)
    for row in merged:
        row["weak_prior_quartile"] = quartile_name(float(row["q_score"]), bounds)

    strata: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in merged:
        key = (row["subreddit"], row["topic"], row["weak_prior_quartile"])
        strata[key].append(row)

    target_n = min(args.sample_size, len(merged))
    sampled: List[Dict[str, str]] = []
    total = len(merged)
    for key, rows in strata.items():
        proportion = len(rows) / total
        take = max(1, int(round(proportion * target_n)))
        random.shuffle(rows)
        sampled.extend(rows[: min(take, len(rows))])

    random.shuffle(sampled)
    sampled = sampled[:target_n]
    if len(sampled) < target_n:
        sampled_ids = {r["post_id"] for r in sampled}
        remaining = [r for r in merged if r["post_id"] not in sampled_ids]
        random.shuffle(remaining)
        sampled.extend(remaining[: target_n - len(sampled)])

    packet: List[Dict[str, str]] = []
    for i, row in enumerate(sampled, 1):
        out = {
            "annotation_id": f"ann_{i:06d}",
            "post_id": row["post_id"],
            "annotator_id": "",
            "subreddit": row["subreddit"],
            "topic": row["topic"],
            "weak_prior_quartile": row["weak_prior_quartile"],
            "text": row["text"],
        }
        for q in QUESTION_FIELDS:
            out[q] = "[]" if q == "evidence_spans_json" else ""
        packet.append(out)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    packet_path = args.output_dir / "annotation_packet.csv"
    write_csv(packet_path, packet)

    for idx in range(1, 4):
        ann_rows = [dict(r, annotator_id=f"annotator_{idx}") for r in packet]
        write_csv(args.output_dir / f"annotation_packet_annotator_{idx}.csv", ann_rows)

    summary = {
        "requested_sample_size": args.sample_size,
        "actual_sample_size": len(packet),
        "population_size": len(merged),
        "quartile_bounds": {"Q1": bounds[0], "Q2": bounds[1], "Q3": bounds[2]},
        "num_strata": len(strata),
    }
    (args.output_dir / "annotation_sampling_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print(f"Wrote annotation packet: {packet_path}")
    print(f"Sample size: {len(packet)}")


if __name__ == "__main__":
    main()
