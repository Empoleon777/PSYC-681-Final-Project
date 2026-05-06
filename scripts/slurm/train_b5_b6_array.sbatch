#!/bin/bash
# Parallel B5/B6 training across seeds (one model-seed per array task).
#
# Submit via:
#   sbatch --array=0-5 --export=ALL,SEEDS="42 43 44",PROJECT_DIR="$PWD" scripts/slurm/train_b5_b6_array.sbatch
#
# Array layout:
#   task_id [0..N-1]     -> B5, seed[i]
#   task_id [N..2N-1]    -> B6, seed[i-N]

#SBATCH --job-name=psyc681_b5b6_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
OUT_ROOT="${OUT_ROOT:-outputs/real_fair_compare}"
SEEDS_STR="${SEEDS:-42 43 44}"
read -r -a SEEDS_ARR <<< "$SEEDS_STR"
N_SEEDS="${#SEEDS_ARR[@]}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"

if (( N_SEEDS < 1 )); then
  echo "No seeds provided. Set SEEDS=\"42 43 44\"."
  exit 2
fi

if (( TASK_ID < N_SEEDS )); then
  MODEL="b5"
  IDX="$TASK_ID"
else
  MODEL="b6"
  IDX="$((TASK_ID - N_SEEDS))"
fi

if (( IDX < 0 || IDX >= N_SEEDS )); then
  echo "Task index out of range. TASK_ID=${TASK_ID}, N_SEEDS=${N_SEEDS}"
  exit 2
fi

SEED="${SEEDS_ARR[$IDX]}"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "OUT_ROOT=$OUT_ROOT"
echo "SEEDS=$SEEDS_STR"
echo "TASK_ID=$TASK_ID => MODEL=$MODEL SEED=$SEED"

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

# Optional module load hooks (safe if module command exists on cluster).
if command -v module >/dev/null 2>&1; then
  module purge || true
  module load cuda || true
fi

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE="${PROJECT_DIR}/.cache/huggingface"
export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
mkdir -p "$TRANSFORMERS_CACHE"

# Tunable knobs (overridable at submit time via --export=ALL,...).
RAW_POSTS_CSV="${RAW_POSTS_CSV:-outputs/ingestion/raw_posts_60k.csv}"
WEAK_LABELS_CSV="${WEAK_LABELS_CSV:-outputs/weak_labels_60k/post_weak_labels.csv}"
GOLD_AGGREGATES_CSV="${GOLD_AGGREGATES_CSV:-outputs/annotation_60k/gold_aggregates.csv}"
ENCODER_NAME="${ENCODER_NAME:-roberta-base}"
DEVICE="${DEVICE:-cuda}"
MAX_LENGTH="${MAX_LENGTH:-128}"
BATCH_SIZE="${BATCH_SIZE:-16}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
WEAK_EPOCHS="${WEAK_EPOCHS:-2}"
GOLD_EPOCHS="${GOLD_EPOCHS:-5}"
WEAK_MAX_SAMPLES="${WEAK_MAX_SAMPLES:-0}"
GOLD_MAX_SAMPLES="${GOLD_MAX_SAMPLES:-0}"
B6_GOLD_LOSS_MODE="${B6_GOLD_LOSS_MODE:-hybrid}"
B6_HYBRID_ALPHA="${B6_HYBRID_ALPHA:-0.5}"
B6_CONSISTENCY_WEIGHT="${B6_CONSISTENCY_WEIGHT:-0.05}"
B6_CONSISTENCY_TIME_SCALE_DAYS="${B6_CONSISTENCY_TIME_SCALE_DAYS:-30.0}"
OFFLINE_RANDOM_INIT="${OFFLINE_RANDOM_INIT:-0}"

if [[ "$OFFLINE_RANDOM_INIT" == "1" ]]; then
  OFFLINE_FLAG="--offline-random-init"
else
  OFFLINE_FLAG=""
fi

if [[ "$MODEL" == "b5" ]]; then
  OUT_DIR="${OUT_ROOT}/b5_seed${SEED}"
  mkdir -p "$OUT_DIR"
  python3 models/flat_transformer_b3_b5.py \
    --variant b5 \
    --raw-posts-csv "$RAW_POSTS_CSV" \
    --weak-labels-csv "$WEAK_LABELS_CSV" \
    --gold-aggregates-csv "$GOLD_AGGREGATES_CSV" \
    --encoder-name "$ENCODER_NAME" \
    --device "$DEVICE" \
    --max-length "$MAX_LENGTH" \
    --batch-size "$BATCH_SIZE" \
    --weak-epochs "$WEAK_EPOCHS" \
    --gold-epochs "$GOLD_EPOCHS" \
    --weak-max-samples "$WEAK_MAX_SAMPLES" \
    --gold-max-samples "$GOLD_MAX_SAMPLES" \
    --seed "$SEED" \
    $OFFLINE_FLAG \
    --output-json "${OUT_DIR}/b5_summary.json" \
    --output-predictions-csv "${OUT_DIR}/b5_val_predictions.csv"
else
  OUT_DIR="${OUT_ROOT}/b6_seed${SEED}"
  mkdir -p "$OUT_DIR"
  python3 scripts/train_b6_representation.py \
    --raw-posts-csv "$RAW_POSTS_CSV" \
    --weak-labels-csv "$WEAK_LABELS_CSV" \
    --gold-aggregates-csv "$GOLD_AGGREGATES_CSV" \
    --encoder-name "$ENCODER_NAME" \
    --device "$DEVICE" \
    --max-length "$MAX_LENGTH" \
    --batch-size "$BATCH_SIZE" \
    --val-fraction "$VAL_FRACTION" \
    --weak-epochs "$WEAK_EPOCHS" \
    --gold-epochs "$GOLD_EPOCHS" \
    --weak-max-samples "$WEAK_MAX_SAMPLES" \
    --gold-max-samples "$GOLD_MAX_SAMPLES" \
    --gold-loss-mode "$B6_GOLD_LOSS_MODE" \
    --hybrid-alpha "$B6_HYBRID_ALPHA" \
    --consistency-weight "$B6_CONSISTENCY_WEIGHT" \
    --consistency-time-scale-days "$B6_CONSISTENCY_TIME_SCALE_DAYS" \
    --enable-curriculum \
    --seed "$SEED" \
    $OFFLINE_FLAG \
    --output-dir "$OUT_DIR" \
    --output-predictions-csv "${OUT_DIR}/val_predictions.csv"
fi

echo "Done: MODEL=$MODEL SEED=$SEED"
