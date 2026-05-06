# PSYC-681 Final Project

This repo contains the project pipeline for:

- Reddit ingestion into a normalized post table
- weak-label construction from lightweight lexical/rule sources
- annotation packet creation and aggregation
- the B6 hierarchical ideology model

## Data Sources

Reference list:
- [data_sources.md](/Users/jacobkim/PSYC-681-Final-Project/docs/data_sources.md)

Quick links:
- MITweet: https://github.com/LST1836/MITweet
- Reddit Politosphere: https://github.com/valentinhofmann/politosphere
- Reddit Political Compass: https://github.com/arnestc/political-compass
- POLITISKY24: https://zenodo.org/records/15616911

## Setup

```bash
./scripts/bootstrap_env.sh
```

If you want to run database writes:

```bash
export DATABASE_URL=postgresql://USERNAME_GOES_HERE:PASSWORD_GOES_HERE@localhost:5432/psyc681_ideology
python scripts/apply_schema.py
python scripts/db_roundtrip_check.py
```

Schema file:
- [001_core_tables.sql](/Users/jacobkim/PSYC-681-Final-Project/sql/001_core_tables.sql)

Task spec:
- [task_spec.md](/Users/jacobkim/PSYC-681-Final-Project/docs/task_spec.md)

## Pull Source Repos

```bash
./scripts/sync_sources.sh
```

## Ingest Posts

General ingest:

```bash
python scripts/import_reddit_posts.py \
  --input-dir Data/sources \
  --source-dataset reddit_mixed_sources \
  --output-csv outputs/ingestion/raw_posts.csv
```

Politosphere 60k-line stream sample:

```bash
mkdir -p Data/sources/politosphere_raw
curl -L "https://zenodo.org/records/5851729/files/comments_2019-12.bz2?download=1" \
  | bzip2 -dc \
  | head -n 60000 > Data/sources/politosphere_raw/comments_2019-12_sample60k.jsonl

python scripts/import_reddit_posts.py \
  --input-dir Data/sources/politosphere_raw \
  --source-dataset politosphere_2019_12_sample \
  --output-csv outputs/ingestion/raw_posts_60k.csv \
  --limit 60000

python scripts/data_quality_report.py \
  --input-csv outputs/ingestion/raw_posts_60k.csv \
  --output-json outputs/ingestion/data_quality_report_60k.json
```

Subreddit scope notes:
- [subreddit_scope.md](/Users/jacobkim/PSYC-681-Final-Project/docs/subreddit_scope.md)

## Build Weak Labels

```bash
python scripts/build_weak_labels.py \
  --raw-posts-csv outputs/ingestion/raw_posts_60k.csv \
  --output-dir outputs/weak_labels_60k
```

Key outputs:

- `outputs/weak_labels_60k/source_events.csv`
- `outputs/weak_labels_60k/post_weak_labels.csv`
- `outputs/weak_labels_60k/user_priors.csv`
- `outputs/weak_labels_60k/false_positive_report.json`

Rule files:

- [flair_priors.json](/Users/jacobkim/PSYC-681-Final-Project/pipeline/lexicons/flair_priors.json)
- [community_axis_map.json](/Users/jacobkim/PSYC-681-Final-Project/pipeline/lexicons/community_axis_map.json)
- [self_id_rules.json](/Users/jacobkim/PSYC-681-Final-Project/pipeline/lexicons/self_id_rules.json)
- [target_seeds.json](/Users/jacobkim/PSYC-681-Final-Project/pipeline/lexicons/target_seeds.json)
- [frame_seeds.json](/Users/jacobkim/PSYC-681-Final-Project/pipeline/lexicons/frame_seeds.json)

## Annotation Workflow

Guidelines:
- [annotation_manual.md](/Users/jacobkim/PSYC-681-Final-Project/docs/annotation_manual.md)

Create a batch:

```bash
python scripts/make_annotation_batch.py \
  --raw-posts-csv outputs/ingestion/raw_posts_60k.csv \
  --weak-labels-csv outputs/weak_labels_60k/post_weak_labels.csv \
  --sample-size 1200 \
  --output-dir Data/annotation_60k
```

By default, the annotation packet carries weak-label starting values (`q01..q12`) so reviewers can correct them instead of starting cold.
Use `--blank-fields` if you want empty annotation fields instead.

Fold three annotator files into aggregate outputs:

```bash
python scripts/fold_annotations.py \
  --annotation-glob "Data/annotation_60k/annotation_packet_annotator_*.csv" \
  --output-dir outputs/annotation_60k \
  --require-complete-labels \
  --require-three-annotators
```

`--require-complete-labels` enforces non-blank `q01..q11` before writing final gold outputs.
`--require-three-annotators` enforces exactly three annotators per post.

To persist Task-14 outputs to PostgreSQL:

```bash
export DATABASE_URL=postgresql://USERNAME_GOES_HERE:PASSWORD_GOES_HERE@localhost:5432/psyc681_ideology
python scripts/fold_annotations.py \
  --annotation-glob "Data/annotation_60k/annotation_packet_annotator_*.csv" \
  --output-dir outputs/annotation_60k \
  --require-complete-labels \
  --require-three-annotators \
  --write-db
```

## Label Definitions

Canonical label schema:
- [label_schema.md](/Users/jacobkim/PSYC-681-Final-Project/docs/label_schema.md)
- [mitweet_mapping.md](/Users/jacobkim/PSYC-681-Final-Project/docs/mitweet_mapping.md)

## B1-B5 Baselines

B1 lexical baseline:

```bash
python models/tfidf-Task15.py \
  --raw-posts-csv outputs/ingestion/raw_posts_60k.csv \
  --gold-aggregates-csv outputs/annotation_60k/gold_aggregates.csv \
  --output-json outputs/model_runs/b1_summary.json \
  --output-predictions-csv outputs/model_runs/b1_val_predictions.csv
```

B2 flat transformer baseline:

```bash
python models/RoBERTa-base-gold.py \
  --raw-posts-csv outputs/ingestion/raw_posts_60k.csv \
  --gold-aggregates-csv outputs/annotation_60k/gold_aggregates.csv \
  --epochs 2
```

B3/B4/B5 flat weak-pretrained runs:

```bash
python models/flat_transformer_b3_b5.py --variant b3 --output-json outputs/model_runs/b3_summary.json
python models/flat_transformer_b3_b5.py --variant b4 --output-json outputs/model_runs/b4_summary.json
python models/flat_transformer_b3_b5.py --variant b5 --output-json outputs/model_runs/b5_summary.json
```

MITweet lexical baseline:

```bash
python models/tfidf-Task20.py
```

## Task 30 Unified Evaluation

Run the complete metric bundle (standard, ordinal, disagreement-aware, calibration,
selective prediction, signal-level, decomposition, evidence):

```bash
python scripts/eval.py \
  --predictions-csv outputs/model_runs/b1_val_predictions.csv \
  --gold-aggregates-csv outputs/annotation_60k/gold_aggregates.csv \
  --model-name b1_tfidf_lr \
  --regime in_domain \
  --split-name val \
  --output-json outputs/eval/task30_b1_in_domain_metrics.json
```

Optional DB write:

```bash
export DATABASE_URL=postgresql://localhost:5432/psyc681_ideology
python scripts/eval.py \
  --predictions-csv outputs/model_runs/b1_val_predictions.csv \
  --gold-aggregates-csv outputs/annotation_60k/gold_aggregates.csv \
  --model-name b1_tfidf_lr \
  --regime in_domain \
  --split-name val \
  --output-json outputs/eval/task30_b1_in_domain_metrics.json \
  --write-db
```

Note: metrics that require unavailable heads/outputs for a given model (for example
target/stance/frame or evidence-token predictions for B1) are explicitly marked
as unavailable in the output JSON.

## B6 Model

Architecture file:
- [b6_hierarchy.py](/Users/jacobkim/PSYC-681-Final-Project/models/b6_hierarchy.py)

Train full representation-learning pipeline (weak pretraining + gold fine-tuning):

```bash
python scripts/train_b6_representation.py \
  --raw-posts-csv outputs/ingestion/raw_posts_60k.csv \
  --weak-labels-csv outputs/weak_labels_60k/post_weak_labels.csv \
  --gold-aggregates-csv outputs/annotation_60k/gold_aggregates.csv \
  --output-dir outputs/model_runs/b6_representation \
  --weak-epochs 1 \
  --gold-epochs 3 \
  --gold-loss-mode hybrid
```

Gold supervision options:
- `--gold-loss-mode hard`: majority labels only
- `--gold-loss-mode soft`: annotator distributions only (KL objective for ideology heads)
- `--gold-loss-mode hybrid`: weighted mix (use `--hybrid-alpha`)

Stage-2 validation also logs an ambiguity-heavy subset slice to support disagreement-aware analysis.
Task 23/24 options:
- `--consistency-weight 0.05 --consistency-time-scale-days 30` for temporal consistency regularization.
- `--enable-curriculum` for confidence-based curriculum during weak pretraining.

Task 25 outputs:
- post-hoc temperature scaling, ECE/Brier, and risk-coverage tables in `calibration_summary.json`
  (disable with `--skip-calibration`).

Offline/restricted-network quick-start mode:
- add `--offline-random-init` to avoid downloading pretrained weights/tokenizer.

Task 26 analysis (continuous signal + vector field):

```bash
python scripts/analyze_b6_signal_field.py \
  --checkpoint outputs/model_runs/b6_representation/stage2_gold_finetuned.pt \
  --raw-posts-csv outputs/ingestion/raw_posts_60k.csv \
  --weak-labels-csv outputs/weak_labels_60k/post_weak_labels.csv \
  --output-dir outputs/task26_signal_analysis
```

Task-26 analysis notebook:
- [task26_signal_analysis.ipynb](/Users/jacobkim/PSYC-681-Final-Project/notebooks/task26_signal_analysis.ipynb)

Notes:
- [b6_architecture_notes.md](/Users/jacobkim/PSYC-681-Final-Project/docs/b6_architecture_notes.md)

## Task 27/28 Heads

Evidence + psychology extension (with ablation head checks):

```bash
python models/b6_hierarchy_task_27.py \
  --gold-annotations-csv outputs/annotation_60k/gold_annotations.csv \
  --output-json outputs/model_runs/task27_28_head_checks.json
```

## Tasks 32-36 Bundle

Run the end-to-end experiments/report bundle for Tasks 32-36:

```bash
python scripts/run_tasks_32_36.py
```

Primary outputs:

- `outputs/final_32_36/task32_model_ladder_table.csv`
- `outputs/final_32_36/task33_ablation_table.csv`
- `outputs/final_32_36/task34_robustness_summary.json`
- `outputs/final_32_36/task34_counterfactual_edit_set.csv`
- `outputs/final_32_36/task35_error_analysis_examples.csv`
- `outputs/final_32_36/task36_final_summary.json`
- `docs/final_deliverables_32_36.md`
