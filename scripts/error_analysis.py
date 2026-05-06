#!/usr/bin/env python3
"""Stage-decomposed + content-taxonomy error analysis (Task 35).

Inputs are a per-row predictions CSV (post_id, pred_<task>, probs_<task>) and
the gold majority labels. Produces:
  * Per-task confusion matrices and per-class F1.
  * Stage decomposition: which head erred (relevance / target / stance / frame
    / projection failure for ideology heads conditioned on correct discourse).
  * Content taxonomy: sarcasm, hedged, identity-signaling, mixed-signal,
    off-topic, short.
  * Calibration of errors: mean predicted-class probability on errors vs.
    correct rows. High-confidence errors are the dangerous ones.
  * Top-K representative errors per category.
  * Per-stratum (subreddit / topic) error rate so we can spot systematic
    failures under shift.

Usage:
    python scripts/error_analysis.py \
        --predictions-csv outputs/model_runs/b5_val_predictions.csv \
        --gold-aggregates-csv outputs/annotation_60k/gold_aggregates.csv \
        --raw-posts-csv outputs/ingestion/raw_posts_60k.csv \
        --model-name b5 \
        --output-dir outputs/error_analysis/b5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


TASK_SPECS: Dict[str, Tuple[str, List[str]]] = {
    "relevance": ("q01_relevance", ["0", "1"]),
    "target": (
        "q02_target",
        [
            "government_institutions",
            "political_party_or_actor",
            "policy_or_legislation",
            "social_group_or_identity",
            "foreign_actor_or_geopolitics",
            "media_or_information",
            "other",
        ],
    ),
    "stance": ("q03_stance", ["support", "oppose", "mixed_or_unclear"]),
    "frame": (
        "q04_frame",
        [
            "economic_freedom",
            "economic_redistribution",
            "law_and_order",
            "civil_rights",
            "national_security",
            "environmental_protection",
            "public_health",
            "identity_and_values",
            "other",
        ],
    ),
    "moralization": ("q05_moralization", ["0", "1", "2"]),
    "identity_signaling": ("q06_identity_signaling", ["0", "1", "2"]),
    "economic_direction": ("q07_economic_direction", ["-1", "0", "1"]),
    "social_direction": ("q08_social_direction", ["-1", "0", "1"]),
    "intensity": ("q09_intensity", ["0", "1", "2"]),
    "ambiguity": ("q10_ambiguity", ["0", "1", "2"]),
}

# Tasks split by "what stage of the pipeline" they live in.
DISCOURSE_TASKS = ["relevance", "target", "stance", "frame"]
IDEOLOGY_TASKS = ["economic_direction", "social_direction", "intensity", "ambiguity"]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_macro_f1_per_class(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for label in labels:
        tp = fp = fn = 0
        for gt, pd in zip(y_true, y_pred):
            if gt == label and pd == label:
                tp += 1
            elif gt != label and pd == label:
                fp += 1
            elif gt == label and pd != label:
                fn += 1
        if tp == 0 and (fp > 0 or fn > 0):
            out[label] = 0.0
            continue
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        out[label] = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return out


def confusion(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Dict[str, int]]:
    mat: Dict[str, Dict[str, int]] = {l: {p: 0 for p in labels} for l in labels}
    for gt, pd in zip(y_true, y_pred):
        if gt in mat and pd in mat[gt]:
            mat[gt][pd] += 1
    return mat


def content_tag(text: str) -> str:
    t = text.lower()
    if any(x in t for x in ["yeah right", "sure buddy", "/s ", " /s", "lol", "lmao", " sarcasm"]):
        return "sarcasm_or_irony"
    if any(x in t for x in ["i think", "maybe", "perhaps", "not sure", "kind of", "sort of"]):
        return "vague_or_hedged"
    if any(x in t for x in ["as a ", "we conservatives", "we liberals", "our community", "we are the", "our side"]):
        return "identity_signaling"
    if any(x in t for x in [" both ", " however", "on the other hand", " but also", " yet "]):
        return "mixed_signals"
    if len(t.split()) < 8:
        return "too_short"
    return "other"


def parse_probs(raw: str) -> Dict[str, float]:
    if not raw:
        return {}
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return {str(k): float(v) for k, v in d.items()}


def predicted_confidence(probs: Dict[str, float], pred: str) -> Optional[float]:
    if not probs:
        return None
    return probs.get(pred)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-csv", type=Path, required=True)
    ap.add_argument("--gold-aggregates-csv", type=Path, default=Path("outputs/annotation_60k/gold_aggregates.csv"))
    ap.add_argument("--raw-posts-csv", type=Path, default=Path("outputs/ingestion/raw_posts_60k.csv"))
    ap.add_argument("--model-name", type=str, default="model")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--top-k-errors", type=int, default=5)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pred_rows = read_csv(args.predictions_csv)
    gold_rows = read_csv(args.gold_aggregates_csv)
    raw_rows = read_csv(args.raw_posts_csv)
    raw_by_post = {r["post_id"]: r for r in raw_rows}
    maj_by_post: Dict[str, Dict[str, str]] = {}
    for r in gold_rows:
        try:
            maj_by_post[r["post_id"]] = json.loads(r["majority_json"])
        except (json.JSONDecodeError, KeyError):
            continue

    # Build a row-per-post analysis record.
    records: List[Dict[str, object]] = []
    for pred in pred_rows:
        pid = pred.get("post_id", "")
        majority = maj_by_post.get(pid)
        raw = raw_by_post.get(pid, {})
        if not majority:
            continue
        text = (raw.get("text") or "").strip()
        record: Dict[str, object] = {
            "post_id": pid,
            "subreddit": raw.get("subreddit", ""),
            "topic": raw.get("topic", ""),
            "text": text,
            "content_tag": content_tag(text),
        }
        for task, (qkey, values) in TASK_SPECS.items():
            gt = str(majority.get(qkey, "")).strip()
            pd = str(pred.get(f"pred_{task}", "")).strip()
            probs = parse_probs(pred.get(f"probs_{task}", ""))
            conf = predicted_confidence(probs, pd) if pd in values else None
            record[f"gold_{task}"] = gt
            record[f"pred_{task}"] = pd
            record[f"correct_{task}"] = bool(gt and gt == pd)
            record[f"conf_{task}"] = conf
            record[f"available_{task}"] = bool(gt in values and pd in values)
        records.append(record)

    if not records:
        raise SystemExit(f"No overlap between {args.predictions_csv} and gold for model={args.model_name}")

    # 1) Per-task macro/per-class F1 + confusion matrix.
    per_task: Dict[str, Dict[str, object]] = {}
    for task, (_, values) in TASK_SPECS.items():
        ys = [(str(r[f"gold_{task}"]), str(r[f"pred_{task}"])) for r in records if r[f"available_{task}"]]
        if not ys:
            per_task[task] = {"available": False, "n": 0}
            continue
        y_true = [a for a, _ in ys]
        y_pred = [b for _, b in ys]
        f1_per = safe_macro_f1_per_class(y_true, y_pred, values)
        macro_f1 = float(statistics.mean(f1_per.values())) if f1_per else 0.0
        n_correct = sum(1 for a, b in ys if a == b)
        per_task[task] = {
            "available": True,
            "n": len(ys),
            "macro_f1": round(macro_f1, 4),
            "accuracy": round(n_correct / len(ys), 4),
            "per_class_f1": {k: round(v, 4) for k, v in f1_per.items()},
            "support_per_class": dict(Counter(y_true)),
            "confusion_matrix": confusion(y_true, y_pred, values),
        }

    # 2) Stage decomposition for ideology errors. For each ideology task, count
    # how often the discourse pipeline (relevance/target/stance/frame) was right
    # but the ideology head still erred = "projection_failure", vs. errors that
    # co-occurred with at least one upstream error.
    stage_decomp: Dict[str, Dict[str, int]] = {}
    for task in IDEOLOGY_TASKS:
        bucket: Counter = Counter()
        for r in records:
            if not r[f"available_{task}"]:
                continue
            if r[f"correct_{task}"]:
                continue
            upstream_wrong = []
            for u in DISCOURSE_TASKS:
                if r.get(f"available_{u}") and not r.get(f"correct_{u}"):
                    upstream_wrong.append(u)
            if not upstream_wrong:
                bucket["projection_failure"] += 1
            else:
                bucket["co_with_" + "_".join(sorted(upstream_wrong))] += 1
        stage_decomp[task] = dict(bucket)

    # 3) Content taxonomy: error rate by tag, per task.
    by_tag_task: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"n": 0, "errors": 0}))
    for r in records:
        tag = str(r["content_tag"])
        for task in TASK_SPECS:
            if not r[f"available_{task}"]:
                continue
            by_tag_task[tag][task]["n"] += 1
            if not r[f"correct_{task}"]:
                by_tag_task[tag][task]["errors"] += 1
    content_taxonomy: Dict[str, Dict[str, object]] = {}
    for tag, task_map in by_tag_task.items():
        content_taxonomy[tag] = {
            task: {
                "n": v["n"],
                "errors": v["errors"],
                "error_rate": round(v["errors"] / v["n"], 4) if v["n"] > 0 else None,
            }
            for task, v in task_map.items()
        }

    # 4) Calibration of errors: mean predicted-class probability on correct
    # vs. error rows. High-confidence errors are the worst kind.
    confidence_stats: Dict[str, Dict[str, float]] = {}
    for task in TASK_SPECS:
        confs_correct = [float(r[f"conf_{task}"]) for r in records if r[f"available_{task}"] and r[f"correct_{task}"] and r[f"conf_{task}"] is not None]
        confs_error = [float(r[f"conf_{task}"]) for r in records if r[f"available_{task}"] and not r[f"correct_{task}"] and r[f"conf_{task}"] is not None]
        if not (confs_correct or confs_error):
            continue
        confidence_stats[task] = {
            "n_correct": len(confs_correct),
            "n_error": len(confs_error),
            "mean_conf_correct": round(statistics.mean(confs_correct), 4) if confs_correct else None,
            "mean_conf_error": round(statistics.mean(confs_error), 4) if confs_error else None,
            "high_conf_error_rate_at_0.7": round(
                sum(1 for c in confs_error if c >= 0.7) / max(1, len(confs_correct) + len(confs_error)),
                4,
            ),
        }

    # 5) Stratified error rate by subreddit + topic.
    def stratify(field: str, task: str) -> Dict[str, Dict[str, object]]:
        bucket: Dict[str, Dict[str, int]] = defaultdict(lambda: {"n": 0, "errors": 0})
        for r in records:
            if not r[f"available_{task}"]:
                continue
            key = str(r.get(field, "") or "(unknown)")
            bucket[key]["n"] += 1
            if not r[f"correct_{task}"]:
                bucket[key]["errors"] += 1
        return {
            k: {
                "n": v["n"],
                "errors": v["errors"],
                "error_rate": round(v["errors"] / v["n"], 4) if v["n"] > 0 else None,
            }
            for k, v in sorted(bucket.items(), key=lambda kv: -kv[1]["errors"])
        }

    strata = {
        "by_subreddit": {task: stratify("subreddit", task) for task in IDEOLOGY_TASKS},
        "by_topic": {task: stratify("topic", task) for task in IDEOLOGY_TASKS},
    }

    # 6) Top-K representative errors per content tag, sorted by predicted
    # confidence (high-confidence errors surface first).
    top_errors_by_tag: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    error_rows_for_csv: List[Dict[str, object]] = []
    for r in records:
        for task in IDEOLOGY_TASKS:
            if not r[f"available_{task}"]:
                continue
            if r[f"correct_{task}"]:
                continue
            tag = str(r["content_tag"])
            entry = {
                "post_id": r["post_id"],
                "task": task,
                "subreddit": r.get("subreddit", ""),
                "topic": r.get("topic", ""),
                "content_tag": tag,
                "gold": r[f"gold_{task}"],
                "pred": r[f"pred_{task}"],
                "confidence": r[f"conf_{task}"],
                "text": str(r.get("text", ""))[:280],
            }
            top_errors_by_tag[tag].append(entry)
            error_rows_for_csv.append(entry)
    for tag in list(top_errors_by_tag.keys()):
        top_errors_by_tag[tag].sort(
            key=lambda e: -(e["confidence"] if e["confidence"] is not None else 0.0)
        )
        top_errors_by_tag[tag] = top_errors_by_tag[tag][: args.top_k_errors]

    # Write all errors CSV (for later inspection).
    errors_csv = args.output_dir / "errors.csv"
    if error_rows_for_csv:
        with open(errors_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(error_rows_for_csv[0].keys()))
            w.writeheader()
            w.writerows(error_rows_for_csv)

    # Stage_failure summary table (one row per IDEOLOGY task).
    stage_csv = args.output_dir / "stage_decomposition.csv"
    with open(stage_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "category", "count"])
        for task, buckets in stage_decomp.items():
            for cat, c in sorted(buckets.items(), key=lambda kv: -kv[1]):
                w.writerow([task, cat, c])

    # Per-task per-class F1 table.
    per_class_csv = args.output_dir / "per_class_f1.csv"
    with open(per_class_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "class", "support", "f1"])
        for task, info in per_task.items():
            if not info.get("available"):
                continue
            for cls, f1 in info["per_class_f1"].items():
                w.writerow([task, cls, info["support_per_class"].get(cls, 0), f1])

    summary = {
        "model_name": args.model_name,
        "predictions_csv": str(args.predictions_csv),
        "n_records": len(records),
        "per_task_metrics": per_task,
        "stage_decomposition_for_ideology_errors": stage_decomp,
        "content_taxonomy_error_rates": content_taxonomy,
        "confidence_stats_correct_vs_error": confidence_stats,
        "stratified_error_rates": strata,
        "top_errors_per_content_tag": top_errors_by_tag,
        "outputs": {
            "errors_csv": str(errors_csv),
            "stage_decomposition_csv": str(stage_csv),
            "per_class_f1_csv": str(per_class_csv),
        },
    }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    # Short stdout digest.
    print(f"=== Error analysis: {args.model_name} ===")
    print(f"records analyzed: {len(records)}")
    for task, info in per_task.items():
        if info.get("available"):
            print(f"  {task:22s}  n={info['n']:4d}  acc={info['accuracy']:.3f}  macro_f1={info['macro_f1']:.3f}")
    print("stage decomposition (ideology):")
    for task, b in stage_decomp.items():
        total = sum(b.values()) or 1
        proj_fail = b.get("projection_failure", 0)
        print(f"  {task:22s}  errors={total}  projection_failure={proj_fail}  ({proj_fail / total:.1%})")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
