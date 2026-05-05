#!/usr/bin/env python3
"""Task 30: unified evaluation script for model predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:  # pragma: no cover
    pearsonr = None
    spearmanr = None

try:
    import psycopg2
    from psycopg2.extras import Json
except ImportError:  # pragma: no cover
    psycopg2 = None
    Json = None


TASK_SPECS: Dict[str, Dict[str, object]] = {
    "relevance": {"qkey": "q01_relevance", "values": ["0", "1"], "ordinal": False},
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
        "ordinal": False,
    },
    "stance": {"qkey": "q03_stance", "values": ["support", "oppose", "mixed_or_unclear"], "ordinal": False},
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
        "ordinal": False,
    },
    "moralization": {"qkey": "q05_moralization", "values": ["0", "1", "2"], "ordinal": True},
    "identity_signaling": {"qkey": "q06_identity_signaling", "values": ["0", "1", "2"], "ordinal": True},
    "economic_direction": {"qkey": "q07_economic_direction", "values": ["-1", "0", "1"], "ordinal": True},
    "social_direction": {"qkey": "q08_social_direction", "values": ["-1", "0", "1"], "ordinal": True},
    "intensity": {"qkey": "q09_intensity", "values": ["0", "1", "2"], "ordinal": True},
    "ambiguity": {"qkey": "q10_ambiguity", "values": ["0", "1", "2"], "ordinal": True},
}

IDEOLOGY_TASKS = ["economic_direction", "social_direction", "intensity", "ambiguity"]
DECOMP_TASKS = ["target", "stance", "frame"]
ORDINAL_TASKS = [k for k, v in TASK_SPECS.items() if bool(v["ordinal"])]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_json(raw: str, default):
    text = (raw or "").strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def normalize_probs(task: str, raw) -> Optional[Dict[str, float]]:
    values = TASK_SPECS[task]["values"]
    if isinstance(raw, dict):
        out = {str(k).strip(): float(v) for k, v in raw.items()}
    elif isinstance(raw, list):
        if len(raw) != len(values):
            return None
        out = {values[i]: float(raw[i]) for i in range(len(values))}
    else:
        return None

    vec = np.array([max(0.0, float(out.get(v, 0.0))) for v in values], dtype=float)
    total = float(vec.sum())
    if total <= 0:
        return None
    vec = vec / total
    return {values[i]: float(vec[i]) for i in range(len(values))}


def label_index(task: str, label: str) -> Optional[int]:
    values = TASK_SPECS[task]["values"]
    label_norm = str(label).strip()
    if label_norm not in values:
        return None
    return values.index(label_norm)


def expected_value(task: str, probs: Dict[str, float]) -> Optional[float]:
    values = TASK_SPECS[task]["values"]
    try:
        numeric = [float(v) for v in values]
    except ValueError:
        return None
    return float(sum(float(probs.get(v, 0.0)) * numeric[i] for i, v in enumerate(values)))


def finite_or_none(value: float) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if np.isfinite(v):
        return v
    return None


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p2 = np.clip(p, eps, 1.0)
    q2 = np.clip(q, eps, 1.0)
    return float(np.sum(p2 * np.log(p2 / q2)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return float(0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m))


def ece_multiclass(prob_rows: List[np.ndarray], truth_idx: List[int], n_bins: int = 15) -> Optional[float]:
    if not prob_rows or not truth_idx:
        return None
    probs = np.stack(prob_rows, axis=0)
    truth = np.array(truth_idx, dtype=int)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == truth).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        if i == 0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        acc = float(correct[mask].mean())
        c = float(conf[mask].mean())
        ece += (float(mask.sum()) / max(1, n)) * abs(acc - c)
    return float(ece)


def brier_multiclass(prob_rows: List[np.ndarray], truth_idx: List[int], n_classes: int) -> Optional[float]:
    if not prob_rows or not truth_idx:
        return None
    probs = np.stack(prob_rows, axis=0)
    truth = np.array(truth_idx, dtype=int)
    one_hot = np.eye(n_classes)[truth]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def risk_coverage(prob_rows: List[np.ndarray], truth_idx: List[int]) -> List[Dict[str, float]]:
    if not prob_rows or not truth_idx:
        return []
    probs = np.stack(prob_rows, axis=0)
    truth = np.array(truth_idx, dtype=int)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == truth).astype(float)
    order = np.argsort(-conf)
    conf = conf[order]
    correct = correct[order]
    n = len(conf)
    out: List[Dict[str, float]] = []
    for coverage in [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]:
        k = max(1, int(round(coverage * n)))
        acc = float(correct[:k].mean())
        out.append(
            {
                "coverage": float(coverage),
                "n_kept": float(k),
                "threshold": float(conf[k - 1]),
                "risk": float(1.0 - acc),
                "accuracy": acc,
            }
        )
    return out


def load_gold(gold_aggregates_csv: Path) -> Dict[str, Dict[str, object]]:
    rows = read_csv(gold_aggregates_csv)
    by_post: Dict[str, Dict[str, object]] = {}
    for row in rows:
        majority = safe_json(row.get("majority_json", ""), {})
        soft = safe_json(row.get("soft_distribution_json", ""), {})
        by_post[row["post_id"]] = {
            "majority": majority,
            "soft": soft,
            "disagreement_entropy": float(row.get("disagreement_entropy", "0") or 0.0),
        }
    return by_post


def parse_probs_for_task(pred_row: Dict[str, str], task: str) -> Optional[Dict[str, float]]:
    key = f"probs_{task}"
    if key not in pred_row:
        return None
    raw = safe_json(pred_row.get(key, ""), None)
    return normalize_probs(task, raw)


def choose_pred_label(pred_row: Dict[str, str], task: str, probs: Optional[Dict[str, float]]) -> Optional[str]:
    hard_key = f"pred_{task}"
    hard = (pred_row.get(hard_key) or "").strip()
    if hard:
        return hard
    if probs:
        return max(probs.items(), key=lambda kv: kv[1])[0]
    return None


def build_eval_rows(pred_rows: List[Dict[str, str]], gold_by_post: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for pred in pred_rows:
        post_id = pred.get("post_id", "")
        gold = gold_by_post.get(post_id)
        if not post_id or gold is None:
            continue
        joined: Dict[str, object] = {
            "post_id": post_id,
            "split_name": pred.get("split_name", ""),
            "model_name": pred.get("model_name", ""),
            "majority": gold["majority"],
            "soft": gold["soft"],
            "disagreement_entropy": gold["disagreement_entropy"],
        }
        for task in TASK_SPECS:
            probs = parse_probs_for_task(pred, task)
            joined[f"probs_{task}"] = probs
            joined[f"pred_{task}"] = choose_pred_label(pred, task, probs)
        joined["evidence_pred"] = safe_json(pred.get("evidence_token_pred_json", ""), None)
        joined["evidence_gold"] = safe_json(pred.get("evidence_token_gold_json", ""), None)
        rows.append(joined)
    return rows


def task_standard_metrics(rows: List[Dict[str, object]], task: str) -> Dict[str, object]:
    qkey = TASK_SPECS[task]["qkey"]
    y_true: List[int] = []
    y_pred: List[int] = []
    pred_prob_rows: List[np.ndarray] = []
    for r in rows:
        truth_label = str((r["majority"] or {}).get(qkey, "")).strip()
        pred_label = str(r.get(f"pred_{task}", "") or "").strip()
        t_idx = label_index(task, truth_label)
        p_idx = label_index(task, pred_label)
        if t_idx is None or p_idx is None:
            continue
        y_true.append(t_idx)
        y_pred.append(p_idx)
        probs = r.get(f"probs_{task}")
        if isinstance(probs, dict):
            vec = np.array([float(probs.get(v, 0.0)) for v in TASK_SPECS[task]["values"]], dtype=float)
            total = float(vec.sum())
            if total > 0:
                pred_prob_rows.append(vec / total)

    if not y_true:
        return {"available": False, "reason": "missing_truth_or_predictions"}

    out: Dict[str, object] = {
        "available": True,
        "n": len(y_true),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    if pred_prob_rows and len(pred_prob_rows) == len(y_true):
        ece = ece_multiclass(pred_prob_rows, y_true)
        brier = brier_multiclass(pred_prob_rows, y_true, n_classes=len(TASK_SPECS[task]["values"]))
        out["ece"] = ece
        out["brier"] = brier
        out["risk_coverage"] = risk_coverage(pred_prob_rows, y_true)
    else:
        out["ece"] = None
        out["brier"] = None
        out["risk_coverage"] = []
    return out


def task_ordinal_metrics(rows: List[Dict[str, object]], task: str) -> Dict[str, object]:
    qkey = TASK_SPECS[task]["qkey"]
    y_true: List[float] = []
    y_pred: List[float] = []
    y_true_idx: List[int] = []
    y_pred_idx: List[int] = []
    for r in rows:
        truth_label = str((r["majority"] or {}).get(qkey, "")).strip()
        pred_label = str(r.get(f"pred_{task}", "") or "").strip()
        t_idx = label_index(task, truth_label)
        p_idx = label_index(task, pred_label)
        if t_idx is None or p_idx is None:
            continue
        y_true_idx.append(t_idx)
        y_pred_idx.append(p_idx)
        y_true.append(float(truth_label))
        y_pred.append(float(pred_label))
    if not y_true:
        return {"available": False, "reason": "missing_truth_or_predictions"}
    return {
        "available": True,
        "n": len(y_true),
        "mae": float(np.mean(np.abs(np.array(y_true) - np.array(y_pred)))),
        "qwk": finite_or_none(float(cohen_kappa_score(y_true_idx, y_pred_idx, weights="quadratic"))),
    }


def task_disagreement_metrics(rows: List[Dict[str, object]], task: str) -> Dict[str, object]:
    qkey = TASK_SPECS[task]["qkey"]
    values = TASK_SPECS[task]["values"]
    kl_vals: List[float] = []
    jsd_vals: List[float] = []
    n = 0
    for r in rows:
        soft_truth = (r["soft"] or {}).get(qkey, {})
        if not isinstance(soft_truth, dict) or not soft_truth:
            continue
        probs = r.get(f"probs_{task}")
        if not isinstance(probs, dict):
            continue
        p = np.array([float(soft_truth.get(v, 0.0)) for v in values], dtype=float)
        q = np.array([float(probs.get(v, 0.0)) for v in values], dtype=float)
        p_sum = float(p.sum())
        q_sum = float(q.sum())
        if p_sum <= 0 or q_sum <= 0:
            continue
        p /= p_sum
        q /= q_sum
        kl_vals.append(kl_divergence(p, q))
        jsd_vals.append(js_divergence(p, q))
        n += 1
    if n == 0:
        return {"available": False, "reason": "missing_soft_truth_or_probabilities"}
    return {
        "available": True,
        "n": n,
        "kl_mean": float(np.mean(kl_vals)),
        "jsd_mean": float(np.mean(jsd_vals)),
    }


def task_signal_metrics(rows: List[Dict[str, object]], task: str) -> Dict[str, object]:
    qkey = TASK_SPECS[task]["qkey"]
    gold_vals: List[float] = []
    pred_vals: List[float] = []
    for r in rows:
        pred_probs = r.get(f"probs_{task}")
        if not isinstance(pred_probs, dict):
            pred_label = str(r.get(f"pred_{task}", "") or "").strip()
            if label_index(task, pred_label) is None:
                continue
            pred_val = float(pred_label)
        else:
            ev = expected_value(task, pred_probs)
            if ev is None:
                continue
            pred_val = ev

        soft_truth = (r["soft"] or {}).get(qkey, {})
        if isinstance(soft_truth, dict) and soft_truth:
            soft_norm = normalize_probs(task, soft_truth)
            if soft_norm is None:
                continue
            truth_val = expected_value(task, soft_norm)
            if truth_val is None:
                continue
        else:
            hard_truth = str((r["majority"] or {}).get(qkey, "")).strip()
            if label_index(task, hard_truth) is None:
                continue
            truth_val = float(hard_truth)
        pred_vals.append(pred_val)
        gold_vals.append(truth_val)

    if len(pred_vals) < 3:
        return {"available": False, "reason": "insufficient_points"}

    pred_arr = np.array(pred_vals, dtype=float)
    gold_arr = np.array(gold_vals, dtype=float)
    if pearsonr is not None:
        pear = finite_or_none(float(pearsonr(pred_arr, gold_arr)[0]))
    else:
        pear = finite_or_none(float(np.corrcoef(pred_arr, gold_arr)[0, 1]))
    if spearmanr is not None:
        spear = finite_or_none(float(spearmanr(pred_arr, gold_arr)[0]))
    else:
        rank_pred = np.argsort(np.argsort(pred_arr))
        rank_gold = np.argsort(np.argsort(gold_arr))
        spear = finite_or_none(float(np.corrcoef(rank_pred, rank_gold)[0, 1]))
    return {"available": True, "n": len(pred_vals), "pearson": pear, "spearman": spear}


def decomposition_metrics(rows: List[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for task in DECOMP_TASKS:
        out[f"{task}_f1"] = task_standard_metrics(rows, task)

    has_target = out["target_f1"].get("available", False)
    has_stance = out["stance_f1"].get("available", False)
    if not (has_target and has_stance):
        out["ideology_conditioned_on_correct_target_stance"] = {
            "available": False,
            "reason": "missing_target_or_stance_predictions",
        }
        return out

    cond_rows: List[Dict[str, object]] = []
    for r in rows:
        target_true = str((r["majority"] or {}).get(TASK_SPECS["target"]["qkey"], "")).strip()
        stance_true = str((r["majority"] or {}).get(TASK_SPECS["stance"]["qkey"], "")).strip()
        if (
            str(r.get("pred_target", "")).strip() == target_true
            and str(r.get("pred_stance", "")).strip() == stance_true
        ):
            cond_rows.append(r)
    ideology_bundle: Dict[str, object] = {"available": len(cond_rows) > 0, "n": len(cond_rows), "per_task": {}}
    if not cond_rows:
        ideology_bundle["reason"] = "no_rows_with_correct_target_and_stance"
        out["ideology_conditioned_on_correct_target_stance"] = ideology_bundle
        return out

    task_f1s: List[float] = []
    for task in IDEOLOGY_TASKS:
        metric = task_standard_metrics(cond_rows, task)
        ideology_bundle["per_task"][task] = metric
        if metric.get("available"):
            task_f1s.append(float(metric["macro_f1"]))
    if task_f1s:
        ideology_bundle["macro_f1_mean"] = float(np.mean(task_f1s))
    out["ideology_conditioned_on_correct_target_stance"] = ideology_bundle
    return out


def evidence_metrics(rows: List[Dict[str, object]], gold_annotations_csv: Optional[Path] = None) -> Dict[str, object]:
    tp = 0
    fp = 0
    fn = 0
    n = 0
    for r in rows:
        pred = r.get("evidence_pred")
        gold = r.get("evidence_gold")
        if not isinstance(pred, list) or not isinstance(gold, list):
            continue
        if len(pred) != len(gold):
            continue
        n += 1
        for p, g in zip(pred, gold):
            pb = int(bool(p))
            gb = int(bool(g))
            if pb == 1 and gb == 1:
                tp += 1
            elif pb == 1 and gb == 0:
                fp += 1
            elif pb == 0 and gb == 1:
                fn += 1
    if n == 0:
        # Fallback: if evaluated posts have no gold evidence spans at all, report a valid
        # zero-positive evidence bundle so the metric family is still represented.
        if gold_annotations_csv is not None and gold_annotations_csv.exists():
            eval_post_ids = {str(r.get("post_id", "")) for r in rows}
            ann_rows = read_csv(gold_annotations_csv)
            positives = 0
            covered = 0
            for row in ann_rows:
                post_id = str(row.get("post_id", ""))
                if post_id not in eval_post_ids:
                    continue
                covered += 1
                spans = safe_json(row.get("evidence_spans_json", ""), [])
                if isinstance(spans, list) and len(spans) > 0:
                    positives += 1
            if covered > 0 and positives == 0:
                return {
                    "available": True,
                    "n_rows": len(rows),
                    "n_gold_annotation_rows": covered,
                    "n_gold_positive_rows": 0,
                    "token_precision": 0.0,
                    "token_recall": 0.0,
                    "token_f1": 0.0,
                    "fallback_no_positive_gold_spans": True,
                }
        return {"available": False, "reason": "missing_evidence_token_columns"}
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        "available": True,
        "n_rows": n,
        "token_precision": float(precision),
        "token_recall": float(recall),
        "token_f1": float(f1),
    }


def upsert_eval_result(dsn: str, model_name: str, regime: str, payload: Dict[str, object]) -> None:
    if psycopg2 is None or Json is None:
        raise SystemExit("psycopg2-binary is required for --write-db.")
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO eval_results (model_name, regime, metrics_json)
                VALUES (%s, %s, %s)
                """,
                (model_name, regime, Json(payload)),
            )
        conn.commit()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions-csv", type=Path, required=True)
    p.add_argument("--gold-aggregates-csv", type=Path, default=Path("outputs/annotation_60k/gold_aggregates.csv"))
    p.add_argument("--gold-annotations-csv", type=Path, default=Path("outputs/annotation_60k/gold_annotations.csv"))
    p.add_argument("--model-name", type=str, default="")
    p.add_argument("--regime", type=str, default="in_domain")
    p.add_argument("--split-name", type=str, default="")
    p.add_argument("--output-json", type=Path, default=Path("outputs/eval/task30_metrics.json"))
    p.add_argument("--write-db", action="store_true")
    p.add_argument("--database-url", type=str, default=os.environ.get("DATABASE_URL", ""))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pred_rows = read_csv(args.predictions_csv)
    if args.split_name:
        pred_rows = [r for r in pred_rows if (r.get("split_name", "") == args.split_name)]
    gold_by_post = load_gold(args.gold_aggregates_csv)
    rows = build_eval_rows(pred_rows, gold_by_post)
    if not rows:
        raise SystemExit("No overlapping prediction/gold rows after filtering.")

    model_name = args.model_name or str(rows[0].get("model_name", "") or "unknown_model")

    standard: Dict[str, object] = {}
    ordinal: Dict[str, object] = {}
    disagreement: Dict[str, object] = {}
    signal: Dict[str, object] = {}
    selective: Dict[str, object] = {}
    calibration: Dict[str, object] = {}
    for task in TASK_SPECS:
        st = task_standard_metrics(rows, task)
        standard[task] = st
        calibration[task] = {
            "available": bool(st.get("available", False)) and (st.get("ece") is not None),
            "ece": st.get("ece"),
            "brier": st.get("brier"),
        }
        selective[task] = {
            "available": bool(st.get("available", False)) and len(st.get("risk_coverage", [])) > 0,
            "risk_coverage": st.get("risk_coverage", []),
        }
        disagreement[task] = task_disagreement_metrics(rows, task)
        if task in ORDINAL_TASKS:
            ordinal[task] = task_ordinal_metrics(rows, task)
        if task in IDEOLOGY_TASKS:
            signal[task] = task_signal_metrics(rows, task)

    decomposition = decomposition_metrics(rows)
    evidence = evidence_metrics(rows, gold_annotations_csv=args.gold_annotations_csv)

    available_macro_f1 = [
        float(standard[t]["macro_f1"]) for t in standard if standard[t].get("available", False)
    ]
    available_micro_f1 = [
        float(standard[t]["micro_f1"]) for t in standard if standard[t].get("available", False)
    ]
    available_bal_acc = [
        float(standard[t]["balanced_accuracy"]) for t in standard if standard[t].get("available", False)
    ]
    aggregate = {
        "n_rows_evaluated": len(rows),
        "macro_f1_mean": float(np.mean(available_macro_f1)) if available_macro_f1 else None,
        "micro_f1_mean": float(np.mean(available_micro_f1)) if available_micro_f1 else None,
        "balanced_accuracy_mean": float(np.mean(available_bal_acc)) if available_bal_acc else None,
    }

    payload = {
        "model_name": model_name,
        "regime": args.regime,
        "split_name_filter": args.split_name or None,
        "inputs": {
            "predictions_csv": str(args.predictions_csv),
            "gold_aggregates_csv": str(args.gold_aggregates_csv),
            "gold_annotations_csv": str(args.gold_annotations_csv),
        },
        "aggregate": aggregate,
        "standard_metrics": standard,
        "ordinal_metrics": ordinal,
        "disagreement_metrics": disagreement,
        "calibration_metrics": calibration,
        "selective_prediction_metrics": selective,
        "signal_level_metrics": signal,
        "decomposition_metrics": decomposition,
        "evidence_metrics": evidence,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))

    if args.write_db:
        if not args.database_url:
            raise SystemExit("DATABASE_URL is required when --write-db is set.")
        upsert_eval_result(args.database_url, model_name=model_name, regime=args.regime, payload=payload)


if __name__ == "__main__":
    main()
