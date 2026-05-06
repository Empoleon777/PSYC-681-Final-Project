#!/usr/bin/env python3
"""Summarize B2/B3/B4/B5 metrics and pairwise comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b2", type=Path, default=Path("outputs/model_runs/b2_summary.json"))
    parser.add_argument("--b3", type=Path, default=Path("outputs/model_runs/b3_summary.json"))
    parser.add_argument("--b4", type=Path, default=Path("outputs/model_runs/b4_summary.json"))
    parser.add_argument("--b5", type=Path, default=Path("outputs/model_runs/b5_summary.json"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/model_runs/b2_b5_comparison.json"))
    args = parser.parse_args()

    b2 = load_json(args.b2)
    b3 = load_json(args.b3)
    b4 = load_json(args.b4)
    b5 = load_json(args.b5)

    b2_m = float(b2["metrics"]["macro_f1_mean"])
    b3_m = float(b3["metrics"]["macro_f1_mean"])
    b4_m = float(b4["metrics"]["macro_f1_mean"])
    b5_m = float(b5["metrics"]["macro_f1_mean"])
    b3_soft_kl = float(
        b3.get("ambiguity_heavy_soft_alignment", {}).get("soft_kl_mean_ambiguity_heavy", float("nan"))
    )
    b4_soft_kl = float(
        b4.get("ambiguity_heavy_soft_alignment", {}).get("soft_kl_mean_ambiguity_heavy", float("nan"))
    )

    payload = {
        "macro_f1_mean": {
            "b2": b2_m,
            "b3": b3_m,
            "b4": b4_m,
            "b5": b5_m,
        },
        "ambiguity_heavy_soft_kl_mean": {
            "b3": b3_soft_kl,
            "b4": b4_soft_kl,
        },
        "comparisons": {
            "b3_minus_b2": b3_m - b2_m,
            "b4_minus_b3": b4_m - b3_m,
            "b5_minus_b4": b5_m - b4_m,
            "b4_soft_kl_minus_b3_soft_kl": b4_soft_kl - b3_soft_kl,
        },
        "task17_checkpoint_d_satisfied": (b3_m - b2_m) > 0.0,
        "task18_soft_label_benefit_on_ambiguity_heavy": (b4_soft_kl - b3_soft_kl) < 0.0,
        "task19_has_calibration_outputs": bool(b5.get("calibration")),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
