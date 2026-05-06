#!/usr/bin/env bash
# Submit the full fair-comparison pipeline to SLURM.
#
# Usage:
#   bash scripts/slurm/submit_fair_compare.sh
#   SEEDS="42 43 44" OUT_ROOT="outputs/real_fair_compare" bash scripts/slurm/submit_fair_compare.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
SEEDS="${SEEDS:-42 43 44}"
OUT_ROOT="${OUT_ROOT:-outputs/real_fair_compare}"
read -r -a SEEDS_ARR <<< "$SEEDS"
N="${#SEEDS_ARR[@]}"
if (( N < 1 )); then
  echo "No seeds found in SEEDS."
  exit 2
fi

ARRAY_END=$((2 * N - 1))

mkdir -p "${PROJECT_DIR}/logs/slurm"

TRAIN_JOB_ID=$(
  sbatch --parsable \
    --array="0-${ARRAY_END}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",SEEDS="${SEEDS}",OUT_ROOT="${OUT_ROOT}" \
    "${PROJECT_DIR}/scripts/slurm/train_b5_b6_array.sbatch"
)

POST_JOB_ID=$(
  sbatch --parsable \
    --dependency=afterok:${TRAIN_JOB_ID} \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",SEEDS="${SEEDS}",OUT_ROOT="${OUT_ROOT}" \
    "${PROJECT_DIR}/scripts/slurm/postprocess_b5_b6.sbatch"
)

echo "Submitted train array job: ${TRAIN_JOB_ID} (array 0-${ARRAY_END})"
echo "Submitted postprocess job: ${POST_JOB_ID} (afterok:${TRAIN_JOB_ID})"
echo "Track with: squeue -u \$USER"
echo "Logs: ${PROJECT_DIR}/logs/slurm"
