#!/usr/bin/env python3
"""Compare Task 22 ambiguity-heavy behavior between hard and soft runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def load_summary(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_ambiguity_heavy(summary: Dict) -> Dict[str, float]:
    epochs = summary.get("stage2", {}).get("epochs", [])
    if not epochs:
        return {}
    return epochs[-1].get("ambiguity_heavy_val", {})


def extract_ambiguity_heavy_soft_alignment(summary: Dict) -> Dict[str, float]:
    epochs = summary.get("stage2", {}).get("epochs", [])
    if not epochs:
        return {}
    return epochs[-1].get("ambiguity_heavy_soft_alignment_val", {})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard-summary", type=Path, required=True)
    parser.add_argument("--soft-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/model_runs/task22_soft_vs_hard.json"))
    args = parser.parse_args()

    hard = extract_ambiguity_heavy(load_summary(args.hard_summary))
    soft = extract_ambiguity_heavy(load_summary(args.soft_summary))
    keys = sorted(set(hard.keys()) & set(soft.keys()))
    deltas = {k: float(soft[k]) - float(hard[k]) for k in keys if k.endswith("_ambiguity_heavy")}

    hard_soft_align = extract_ambiguity_heavy_soft_alignment(load_summary(args.hard_summary))
    soft_soft_align = extract_ambiguity_heavy_soft_alignment(load_summary(args.soft_summary))
    align_keys = sorted(set(hard_soft_align.keys()) & set(soft_soft_align.keys()))
    align_deltas = {
        k: float(soft_soft_align[k]) - float(hard_soft_align[k])
        for k in align_keys
        if k.endswith("_ambiguity_heavy")
    }

    target_key = "soft_kl_mean_ambiguity_heavy"
    if target_key in hard_soft_align and target_key in soft_soft_align:
        target_delta = float(soft_soft_align[target_key]) - float(hard_soft_align[target_key])
        improved = target_delta < 0.0
        improved_or_equal = target_delta <= 0.0
    else:
        target_key = "macro_f1_mean_ambiguity_heavy"
        target_delta = deltas.get(target_key, 0.0)
        improved = target_delta > 0.0
        improved_or_equal = target_delta >= 0.0

    payload = {
        "hard_summary": str(args.hard_summary),
        "soft_summary": str(args.soft_summary),
        "hard_metrics": hard,
        "soft_metrics": soft,
        "deltas_soft_minus_hard": deltas,
        "hard_soft_alignment_metrics": hard_soft_align,
        "soft_soft_alignment_metrics": soft_soft_align,
        "alignment_deltas_soft_minus_hard": align_deltas,
        "target_metric": target_key,
        "target_delta": target_delta,
        "improved": improved,
        "improved_or_equal": improved_or_equal,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
