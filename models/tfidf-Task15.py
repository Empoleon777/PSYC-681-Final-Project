#!/usr/bin/env python3
"""Task 15: B1 lexical baseline (TF-IDF + Logistic Regression)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


TASK_SPECS: Dict[str, Dict[str, object]] = {
    "relevance": {"qkey": "q01_relevance", "values": ["0", "1"]},
    "target": {
        "qkey": "q02_target",
        "values": [
            "government_institutions",
            "political_party_or_actor",
            "policy_or_legislation",
            "social_group_or_identity",
            "foreign_actor_or_geopolitics",
            "media_or_information",
            "other",
        ],
    },
    "stance": {"qkey": "q03_stance", "values": ["support", "oppose", "mixed_or_unclear"]},
    "frame": {
        "qkey": "q04_frame",
        "values": [
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
    },
    "moralization": {"qkey": "q05_moralization", "values": ["0", "1", "2"]},
    "identity_signaling": {"qkey": "q06_identity_signaling", "values": ["0", "1", "2"]},
    "economic_direction": {"qkey": "q07_economic_direction", "values": ["-1", "0", "1"]},
    "social_direction": {"qkey": "q08_social_direction", "values": ["-1", "0", "1"]},
    "intensity": {"qkey": "q09_intensity", "values": ["0", "1", "2"]},
    "ambiguity": {"qkey": "q10_ambiguity", "values": ["0", "1", "2"]},
}

CORE_TASKS = ["relevance", "economic_direction", "social_direction"]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_dataset(raw_posts_csv: Path, gold_aggregates_csv: Path) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    raw_rows = read_csv(raw_posts_csv)
    gold_rows = read_csv(gold_aggregates_csv)
    raw_by_post = {r["post_id"]: r for r in raw_rows}

    texts: List[str] = []
    post_ids: List[str] = []
    labels: Dict[str, List[str]] = {task: [] for task in TASK_SPECS}

    for g in gold_rows:
        raw = raw_by_post.get(g["post_id"])
        if raw is None:
            continue
        text = (raw.get("text") or "").strip()
        if not text:
            continue
        majority = json.loads(g["majority_json"])

        row_vals: Dict[str, str] = {}
        valid = True
        for task, spec in TASK_SPECS.items():
            qkey = str(spec["qkey"])
            values = list(spec["values"])
            val = str(majority.get(qkey, "")).strip()
            if val not in values:
                valid = False
                break
            row_vals[task] = val
        if not valid:
            continue

        texts.append(text)
        post_ids.append(g["post_id"])
        for task in TASK_SPECS:
            labels[task].append(row_vals[task])
    return texts, post_ids, labels


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return float(f1_score(list(y_true), list(y_pred), average="macro", zero_division=0))


def fit_or_constant_classifier(x_train_vec, y_train: np.ndarray):
    unique = np.unique(y_train)
    if len(unique) < 2:
        return {"kind": "constant", "label": str(unique[0])}
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(x_train_vec, y_train)
    return {"kind": "logreg", "clf": clf}


def predict_labels_and_probs(model_obj, x_test_vec, values: List[str]) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    if model_obj["kind"] == "constant":
        label = model_obj["label"]
        pred = np.array([label] * x_test_vec.shape[0], dtype=object)
        probs = []
        for _ in range(x_test_vec.shape[0]):
            probs.append({v: (1.0 if v == label else 0.0) for v in values})
        return pred, probs

    clf = model_obj["clf"]
    pred = clf.predict(x_test_vec)
    proba = clf.predict_proba(x_test_vec)
    classes = [str(c) for c in clf.classes_]
    probs: List[Dict[str, float]] = []
    for row in proba:
        d = {v: 0.0 for v in values}
        for cls, p in zip(classes, row):
            if cls in d:
                d[cls] = float(p)
        probs.append(d)
    return pred, probs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-posts-csv", type=Path, default=Path("outputs/ingestion/raw_posts_60k.csv"))
    parser.add_argument("--gold-aggregates-csv", type=Path, default=Path("outputs/annotation_60k/gold_aggregates.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/model_runs/b1_summary.json"))
    parser.add_argument(
        "--output-predictions-csv",
        type=Path,
        default=Path("outputs/model_runs/b1_val_predictions.csv"),
    )
    args = parser.parse_args()

    texts, post_ids, labels = load_dataset(args.raw_posts_csv, args.gold_aggregates_csv)
    if len(texts) < 50:
        raise SystemExit("Not enough labeled rows for B1 baseline.")

    idx = np.arange(len(texts))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=args.seed)
    x_train = [texts[i] for i in idx_train]
    x_test = [texts[i] for i in idx_test]
    id_test = [post_ids[i] for i in idx_test]

    tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2, max_df=0.98)
    x_train_vec = tfidf.fit_transform(x_train)
    x_test_vec = tfidf.transform(x_test)

    rng = np.random.default_rng(args.seed)
    task_models: Dict[str, object] = {}
    task_preds: Dict[str, np.ndarray] = {}
    task_probs: Dict[str, List[Dict[str, float]]] = {}
    task_metrics: Dict[str, Dict[str, float]] = {}
    train_class_distribution: Dict[str, Dict[str, int]] = {}

    for task, spec in TASK_SPECS.items():
        values = list(spec["values"])
        y_all = np.array(labels[task], dtype=object)
        y_train = y_all[idx_train]
        y_test = y_all[idx_test]

        model_obj = fit_or_constant_classifier(x_train_vec, y_train)
        task_models[task] = model_obj
        pred, probs = predict_labels_and_probs(model_obj, x_test_vec, values)
        task_preds[task] = pred
        task_probs[task] = probs

        unique, counts = np.unique(y_train, return_counts=True)
        train_class_distribution[task] = {str(u): int(c) for u, c in zip(unique, counts)}

        score = macro_f1(y_test, pred)
        random_pred = rng.choice(np.unique(y_train), size=len(y_test))
        random_score = macro_f1(y_test, random_pred)
        task_metrics[task] = {
            "macro_f1": float(score),
            "random_macro_f1": float(random_score),
            "delta": float(score - random_score),
        }

    core_scores = [task_metrics[t]["macro_f1"] for t in CORE_TASKS]
    core_random_scores = [task_metrics[t]["random_macro_f1"] for t in CORE_TASKS]
    mean_macro = float(np.mean(core_scores))
    mean_random_macro = float(np.mean(core_random_scores))
    beats_random = bool(mean_macro > mean_random_macro)

    for task in CORE_TASKS:
        print(
            f"{task}: macro_f1={task_metrics[task]['macro_f1']:.4f} "
            f"random_macro_f1={task_metrics[task]['random_macro_f1']:.4f}"
        )
        print(f"{task}: train_class_distribution={train_class_distribution[task]}")
    print(f"macro_f1_mean={mean_macro:.4f}")
    print(f"random_macro_f1_mean={mean_random_macro:.4f}")
    print(f"beats_random={beats_random}")

    payload = {
        "variant": "b1",
        "train_rows": int(len(idx_train)),
        "test_rows": int(len(idx_test)),
        "core_tasks": CORE_TASKS,
        "metrics": {
            "macro_f1_mean": mean_macro,
            "random_macro_f1_mean": mean_random_macro,
            "beats_random": beats_random,
            "per_task": task_metrics,
        },
        "train_class_distribution": train_class_distribution,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    pred_rows: List[Dict[str, str]] = []
    for i, post_id in enumerate(id_test):
        row: Dict[str, str] = {
            "post_id": post_id,
            "split_name": "val",
            "model_name": "b1_tfidf_lr",
        }
        for task in TASK_SPECS:
            row[f"pred_{task}"] = str(task_preds[task][i])
            row[f"probs_{task}"] = json.dumps(task_probs[task][i], ensure_ascii=True)
        pred_rows.append(row)

    args.output_predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_predictions_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pred_rows)


if __name__ == "__main__":
    main()
