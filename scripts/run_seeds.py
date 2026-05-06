#!/usr/bin/env python3
"""Multi-seed runner with mean/std aggregation and paired bootstrap tests.

Two modes:

  1. `run`  — execute a model training command with N seeds, collect each
              summary JSON, and report mean / std / 95% CI per metric.
  2. `compare` — paired-bootstrap test between two predictions CSVs that
              share post_ids (typically: same val split, different model).

Examples:

  # 1) run B5 with three seeds and aggregate
  python scripts/run_seeds.py run \\
      --command "python models/flat_transformer_b3_b5.py --variant b5 \\
                 --offline-random-init --weak-epochs 1 --gold-epochs 2 \\
                 --weak-max-samples 200 --gold-max-samples 200 \\
                 --batch-size 8 --max-length 64" \\
      --seeds 42,43,44 \\
      --summary-template outputs/model_runs_b5_seed{seed}/b5_summary.json \\
      --output-dir-template outputs/model_runs_b5_seed{seed} \\
      --predictions-template outputs/model_runs_b5_seed{seed}/b5_val_predictions.csv \\
      --metrics metrics.macro_f1_mean,metrics.relevance_macro_f1 \\
      --output-json outputs/seed_runs/b5.json

  # 2) compare B5 vs B6 with paired bootstrap on shared val rows
  python scripts/run_seeds.py compare \\
      --pred-a outputs/model_runs/b5_val_predictions.csv \\
      --pred-b outputs/model_runs/b6_seed42/val_predictions.csv \\
      --gold-aggregates-csv outputs/annotation_60k/gold_aggregates.csv \\
      --label-a b5 --label-b b6_seed42 \\
      --output-json outputs/seed_runs/b5_vs_b6.json

The bootstrap test resamples post_ids with replacement, recomputes the
per-task macro-F1 difference (B - A), and reports a two-sided p-value
(twice the smaller tail mass) plus a 95% CI.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shlex
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._run_manifest import build_manifest, write_manifest


def _walk(d: Any, path: str) -> Optional[Any]:
    """Walk a dotted path through a nested dict/list. Integer pieces index
    into lists (negative integers index from the end, so `-1` works). Returns
    None if any piece is missing or the type is wrong.
    """
    cur: Any = d
    for piece in path.split("."):
        if isinstance(cur, list):
            try:
                idx = int(piece)
            except ValueError:
                return None
            if idx < -len(cur) or idx >= len(cur):
                return None
            cur = cur[idx]
        elif isinstance(cur, dict):
            if piece not in cur:
                return None
            cur = cur[piece]
        else:
            return None
    return cur


def _mean_std_ci(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "ci95_lo": None, "ci95_hi": None}
    n = len(values)
    mean = float(sum(values) / n)
    if n < 2:
        return {"n": n, "mean": mean, "std": None, "ci95_lo": None, "ci95_hi": None}
    std = float(statistics.stdev(values))
    se = std / math.sqrt(n)
    return {
        "n": n,
        "mean": round(mean, 6),
        "std": round(std, 6),
        "ci95_lo": round(mean - 1.96 * se, 6),
        "ci95_hi": round(mean + 1.96 * se, 6),
    }


# ---------- mode 1: run ----------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("--seeds must be a comma-separated list of integers.")

    metric_paths = [m.strip() for m in args.metrics.split(",") if m.strip()]
    per_seed: List[Dict[str, Any]] = []

    for seed in seeds:
        # The base command can already contain {seed} placeholders; we
        # substitute, then append --seed and (optionally) --output-predictions-csv.
        # We do NOT auto-append --output-json: scripts vary in how they
        # determine the summary path (e.g. B6 derives it from --output-dir),
        # so the user must put any required --output-json or --output-dir
        # in their command template themselves.
        cmd_str = args.command.format(seed=seed)
        cmd_str = cmd_str + f" --seed {seed}"
        if args.predictions_template and "{seed}" in args.predictions_template and "--output-predictions-csv" not in cmd_str:
            pred_path = args.predictions_template.format(seed=seed)
            cmd_str = cmd_str + f" --output-predictions-csv {pred_path}"
        cmd = shlex.split(cmd_str)

        print(f"[seed {seed}] $ {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, check=False)
        if result.returncode != 0:
            print(f"[seed {seed}] FAILED with exit code {result.returncode}", file=sys.stderr)

        summary_path = Path(args.summary_template.format(seed=seed))
        if not summary_path.exists():
            per_seed.append({"seed": seed, "summary_path": str(summary_path), "ok": False})
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            per_seed.append({"seed": seed, "summary_path": str(summary_path), "ok": False, "error": str(exc)})
            continue
        metrics = {m: _walk(payload, m) for m in metric_paths}
        per_seed.append({"seed": seed, "summary_path": str(summary_path), "ok": True, "metrics": metrics})

    aggregates: Dict[str, Dict[str, Optional[float]]] = {}
    for m in metric_paths:
        vals: List[float] = []
        for entry in per_seed:
            if not entry.get("ok"):
                continue
            v = (entry.get("metrics") or {}).get(m)
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                vals.append(float(v))
        aggregates[m] = _mean_std_ci(vals)

    out = {
        "mode": "run",
        "command_template": args.command,
        "seeds": seeds,
        "metric_paths": metric_paths,
        "per_seed": per_seed,
        "aggregates": aggregates,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")

    manifest = build_manifest(
        run_name=f"run_seeds_{'_'.join(str(s) for s in seeds)}",
        inputs={},
        outputs={"aggregate_json": str(out_path)},
        config={
            "command": args.command,
            "seeds": seeds,
            "metric_paths": metric_paths,
            "summary_template": args.summary_template,
            "output_dir_template": args.output_dir_template,
            "predictions_template": args.predictions_template,
        },
        extra={"per_seed_ok": [e.get("ok") for e in per_seed]},
    )
    write_manifest(out_path.with_name(out_path.stem + "_manifest.json"), manifest)

    print()
    print(f"=== seed-aggregated metrics (n={len(seeds)}) ===")
    for m, agg in aggregates.items():
        if agg["mean"] is None:
            print(f"  {m}: insufficient data (n={agg['n']})")
        elif agg["std"] is None:
            print(f"  {m}: mean={agg['mean']:.4f}  (only n={agg['n']} run)")
        else:
            print(f"  {m}: mean={agg['mean']:.4f}  std={agg['std']:.4f}  95%CI=[{agg['ci95_lo']:.4f}, {agg['ci95_hi']:.4f}]  n={agg['n']}")
    print(f"-> {out_path}")


# ---------- mode 2: compare ------------------------------------------------------

TASK_QKEYS: Dict[str, Tuple[str, List[str]]] = {
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
    "economic_direction": ("q07_economic_direction", ["-1", "0", "1"]),
    "social_direction": ("q08_social_direction", ["-1", "0", "1"]),
    "intensity": ("q09_intensity", ["0", "1", "2"]),
    "ambiguity": ("q10_ambiguity", ["0", "1", "2"]),
}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _macro_f1_supported(y_true: List[str], y_pred: List[str], labels: List[str]) -> Optional[float]:
    f1s: List[float] = []
    seen = False
    for label in labels:
        tp = fp = fn = 0
        n_true = 0
        for gt, pd in zip(y_true, y_pred):
            if gt == label:
                n_true += 1
                if pd == label:
                    tp += 1
                else:
                    fn += 1
            elif pd == label:
                fp += 1
        if n_true == 0:
            continue
        seen = True
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1s.append(float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0)
    if not seen or not f1s:
        return None
    return float(sum(f1s) / len(f1s))


def cmd_compare(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)

    pred_a = {r["post_id"]: r for r in _read_csv(Path(args.pred_a))}
    pred_b = {r["post_id"]: r for r in _read_csv(Path(args.pred_b))}
    gold_rows = _read_csv(Path(args.gold_aggregates_csv))
    maj_by_post: Dict[str, Dict[str, str]] = {}
    for r in gold_rows:
        try:
            maj_by_post[r["post_id"]] = json.loads(r["majority_json"])
        except (json.JSONDecodeError, KeyError):
            continue

    shared_ids = sorted(set(pred_a) & set(pred_b) & set(maj_by_post))
    if not shared_ids:
        raise SystemExit("No post_id overlap between the two predictions CSVs and gold aggregates.")

    # Build aligned (gold, pred_a, pred_b) tuples per task. Tasks where one
    # model wasn't trained (column missing) are skipped.
    aligned: Dict[str, List[Tuple[str, str, str]]] = {}
    for task, (qkey, values) in TASK_QKEYS.items():
        col = f"pred_{task}"
        a_has = any(col in pred_a[pid] for pid in shared_ids)
        b_has = any(col in pred_b[pid] for pid in shared_ids)
        if not (a_has and b_has):
            continue
        triples: List[Tuple[str, str, str]] = []
        for pid in shared_ids:
            gt = str(maj_by_post[pid].get(qkey, "")).strip()
            a = str(pred_a[pid].get(col, "")).strip()
            b = str(pred_b[pid].get(col, "")).strip()
            if gt in values and a in values and b in values:
                triples.append((gt, a, b))
        if triples:
            aligned[task] = triples

    if not aligned:
        raise SystemExit("No tasks had usable triples after alignment.")

    per_task: Dict[str, Dict[str, Any]] = {}
    n_boot = args.bootstrap
    for task, triples in aligned.items():
        labels = TASK_QKEYS[task][1]
        y_gt = [t[0] for t in triples]
        y_a = [t[1] for t in triples]
        y_b = [t[2] for t in triples]
        f1_a = _macro_f1_supported(y_gt, y_a, labels)
        f1_b = _macro_f1_supported(y_gt, y_b, labels)
        if f1_a is None or f1_b is None:
            continue
        observed_diff = f1_b - f1_a

        # Paired bootstrap: resample post indices with replacement, recompute
        # the difference, and take the two-sided p-value as 2 * min(tail).
        n = len(triples)
        diffs: List[float] = []
        for _ in range(n_boot):
            idxs = [rng.randrange(n) for _ in range(n)]
            gt_b = [y_gt[i] for i in idxs]
            ya_b = [y_a[i] for i in idxs]
            yb_b = [y_b[i] for i in idxs]
            fa = _macro_f1_supported(gt_b, ya_b, labels)
            fb = _macro_f1_supported(gt_b, yb_b, labels)
            if fa is None or fb is None:
                continue
            diffs.append(fb - fa)
        if not diffs:
            continue
        diffs.sort()
        ci_lo = diffs[int(0.025 * len(diffs))]
        ci_hi = diffs[int(0.975 * len(diffs)) - 1]
        # Two-sided p-value from the bootstrap distribution under the
        # hypothesis that the true difference is zero.
        n_le_zero = sum(1 for d in diffs if d <= 0)
        n_ge_zero = sum(1 for d in diffs if d >= 0)
        p_two_sided = 2.0 * min(n_le_zero, n_ge_zero) / len(diffs)
        per_task[task] = {
            "n_post": n,
            f"{args.label_a}_macro_f1_supported": round(f1_a, 4),
            f"{args.label_b}_macro_f1_supported": round(f1_b, 4),
            "diff_b_minus_a": round(observed_diff, 4),
            "bootstrap_n": len(diffs),
            "diff_ci95_lo": round(ci_lo, 4),
            "diff_ci95_hi": round(ci_hi, 4),
            "p_value_two_sided": round(min(1.0, p_two_sided), 4),
            "supports_b_better": bool(observed_diff > 0 and ci_lo > 0),
            "supports_a_better": bool(observed_diff < 0 and ci_hi < 0),
        }

    out = {
        "mode": "compare",
        "label_a": args.label_a,
        "label_b": args.label_b,
        "n_shared_posts": len(shared_ids),
        "bootstrap_iterations": n_boot,
        "per_task": per_task,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")

    manifest = build_manifest(
        run_name=f"compare_{args.label_a}_vs_{args.label_b}",
        seed=args.seed,
        inputs={
            "pred_a": Path(args.pred_a),
            "pred_b": Path(args.pred_b),
            "gold_aggregates_csv": Path(args.gold_aggregates_csv),
        },
        outputs={"comparison_json": str(out_path)},
        config={
            "label_a": args.label_a,
            "label_b": args.label_b,
            "bootstrap": n_boot,
        },
        extra={"n_shared_posts": len(shared_ids), "tasks_compared": list(per_task.keys())},
    )
    write_manifest(out_path.with_name(out_path.stem + "_manifest.json"), manifest)

    print(f"=== {args.label_a} vs {args.label_b} on {len(shared_ids)} shared posts ===")
    print(f"{'task':22s}  {'A':>6s}  {'B':>6s}  {'B-A':>7s}  {'CI95':>20s}  {'p':>6s}  verdict")
    for task, r in per_task.items():
        verdict = "B>A" if r["supports_b_better"] else ("A>B" if r["supports_a_better"] else "tie")
        ci = f"[{r['diff_ci95_lo']:+.3f},{r['diff_ci95_hi']:+.3f}]"
        print(
            f"  {task:20s}  {r[f'{args.label_a}_macro_f1_supported']:.3f}  "
            f"{r[f'{args.label_b}_macro_f1_supported']:.3f}  "
            f"{r['diff_b_minus_a']:+.3f}  {ci:>20s}  p={r['p_value_two_sided']:.3f}  {verdict}"
        )
    print(f"-> {out_path}")


# ---------- main -----------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    p_run = sub.add_parser("run", help="Run a model command with N seeds and aggregate.")
    p_run.add_argument("--command", type=str, required=True,
                       help="Base command (no --seed; that gets appended). Wrap in quotes.")
    p_run.add_argument("--seeds", type=str, required=True,
                       help="Comma-separated list of integer seeds (e.g. 42,43,44).")
    p_run.add_argument("--summary-template", type=str, required=True,
                       help="Path template for each seed's summary JSON; must contain {seed}.")
    p_run.add_argument("--output-dir-template", type=str, default="",
                       help="Deprecated: include any {seed}-templated --output-dir directly in --command instead. Kept for backwards compatibility; ignored.")
    p_run.add_argument("--predictions-template", type=str, default="",
                       help="Optional template for each seed's val predictions CSV.")
    p_run.add_argument("--metrics", type=str, required=True,
                       help="Dot-separated paths into the summary JSON, comma-separated.")
    p_run.add_argument("--output-json", type=str, required=True)
    p_run.set_defaults(func=cmd_run)

    p_cmp = sub.add_parser("compare", help="Paired bootstrap on two predictions CSVs.")
    p_cmp.add_argument("--pred-a", type=str, required=True)
    p_cmp.add_argument("--pred-b", type=str, required=True)
    p_cmp.add_argument("--gold-aggregates-csv", type=str, default="outputs/annotation_60k/gold_aggregates.csv")
    p_cmp.add_argument("--label-a", type=str, default="A")
    p_cmp.add_argument("--label-b", type=str, default="B")
    p_cmp.add_argument("--bootstrap", type=int, default=2000)
    p_cmp.add_argument("--seed", type=int, default=42)
    p_cmp.add_argument("--output-json", type=str, required=True)
    p_cmp.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
