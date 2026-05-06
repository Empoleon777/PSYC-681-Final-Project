#!/bin/bash
# Post-process B5/B6 completed seed runs:
# - paired bootstrap compares per seed
# - error analysis per seed/model
# - aggregate markdown report

#SBATCH --job-name=psyc681_b5b6_post
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
OUT_ROOT="${OUT_ROOT:-outputs/real_fair_compare}"
SEEDS_STR="${SEEDS:-42 43 44}"
read -r -a SEEDS_ARR <<< "$SEEDS_STR"

cd "$PROJECT_DIR"
mkdir -p logs/slurm

if [[ -f ".venv-py312/bin/activate" ]]; then
  source .venv-py312/bin/activate
elif [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
else
  echo "No venv found (.venv-py312 or .venv)."
  exit 2
fi

RAW_POSTS_CSV="${RAW_POSTS_CSV:-outputs/ingestion/raw_posts_60k.csv}"
GOLD_AGGREGATES_CSV="${GOLD_AGGREGATES_CSV:-outputs/annotation_60k/gold_aggregates.csv}"
BOOTSTRAP="${BOOTSTRAP:-5000}"
MIN_SUPPORT="${MIN_SUPPORT:-30}"

mkdir -p "${OUT_ROOT}/compare" "${OUT_ROOT}/error_analysis"

for SEED in "${SEEDS_ARR[@]}"; do
  echo "Comparing seed ${SEED}..."
  python3 scripts/run_seeds.py compare \
    --pred-a "${OUT_ROOT}/b5_seed${SEED}/b5_val_predictions.csv" \
    --pred-b "${OUT_ROOT}/b6_seed${SEED}/val_predictions.csv" \
    --gold-aggregates-csv "$GOLD_AGGREGATES_CSV" \
    --label-a b5 \
    --label-b b6 \
    --bootstrap "$BOOTSTRAP" \
    --seed "$SEED" \
    --output-json "${OUT_ROOT}/compare/seed_${SEED}.json"

  python3 scripts/error_analysis.py \
    --predictions-csv "${OUT_ROOT}/b5_seed${SEED}/b5_val_predictions.csv" \
    --gold-aggregates-csv "$GOLD_AGGREGATES_CSV" \
    --raw-posts-csv "$RAW_POSTS_CSV" \
    --model-name "b5_seed${SEED}" \
    --min-support "$MIN_SUPPORT" \
    --output-dir "${OUT_ROOT}/error_analysis/b5_seed${SEED}"

  python3 scripts/error_analysis.py \
    --predictions-csv "${OUT_ROOT}/b6_seed${SEED}/val_predictions.csv" \
    --gold-aggregates-csv "$GOLD_AGGREGATES_CSV" \
    --raw-posts-csv "$RAW_POSTS_CSV" \
    --model-name "b6_seed${SEED}" \
    --min-support "$MIN_SUPPORT" \
    --output-dir "${OUT_ROOT}/error_analysis/b6_seed${SEED}"
done

# Build simple aggregate report.
python3 - "$OUT_ROOT" "$SEEDS_STR" <<'PY'
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
seeds = [int(x) for x in sys.argv[2].split()]

def mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")

b5_macro = []
b6_macro = []
rows = []
for seed in seeds:
    b5 = out_root / f"b5_seed{seed}" / "b5_summary.json"
    b6 = out_root / f"b6_seed{seed}" / "training_summary.json"
    cmpf = out_root / "compare" / f"seed_{seed}.json"
    if b5.exists():
        payload = json.loads(b5.read_text(encoding="utf-8"))
        b5_macro.append(float(payload.get("metrics", {}).get("macro_f1_mean", 0.0)))
    if b6.exists():
        payload = json.loads(b6.read_text(encoding="utf-8"))
        epochs = payload.get("stage2", {}).get("epochs", [])
        if epochs:
            b6_macro.append(float(epochs[-1].get("val", {}).get("macro_f1_mean", 0.0)))
    b6_better = b5_better = ties = 0
    if cmpf.exists():
        c = json.loads(cmpf.read_text(encoding="utf-8"))
        for task_row in c.get("per_task", {}).values():
            if task_row.get("supports_b_better"):
                b6_better += 1
            elif task_row.get("supports_a_better"):
                b5_better += 1
            else:
                ties += 1
    rows.append((seed, b6_better, b5_better, ties))

report = out_root / "fair_compare_report.md"
lines = [
    "# Fair B5 vs B6 Comparison (SLURM)",
    "",
    f"- Seeds: {', '.join(str(s) for s in seeds)}",
    f"- B5 mean macro_f1_mean: {mean(b5_macro):.4f}" if b5_macro else "- B5 mean macro_f1_mean: n/a",
    f"- B6 mean stage2 macro_f1_mean: {mean(b6_macro):.4f}" if b6_macro else "- B6 mean stage2 macro_f1_mean: n/a",
    "",
    "## Paired Bootstrap Verdict Counts",
    "",
    "| Seed | Tasks where B6>B5 | Tasks where B5>B6 | Ties |",
    "|---|---:|---:|---:|",
]
for seed, b6b, b5b, tie in rows:
    lines.append(f"| {seed} | {b6b} | {b5b} | {tie} |")
lines.append("")
report.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {report}")
PY

echo "Post-processing complete under ${OUT_ROOT}"
