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
export DATABASE_URL=postgresql://localhost:5432/psyc681_ideology
python scripts/apply_schema.py
python scripts/db_roundtrip_check.py
```

Schema file:
- [001_core_tables.sql](/Users/jacobkim/PSYC-681-Final-Project/sql/001_core_tables.sql)

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
  --require-complete-labels
```

`--require-complete-labels` enforces non-blank `q01..q11` before writing final gold outputs.

## Label Definitions

Canonical label schema:
- [label_schema.md](PSYC-681-Final-Project/docs/label_schema.md)

## B6 Model

<<<<<<< HEAD
Architecture file:
- [b6_ideology_model.py](/Users/jacobkim/PSYC-681-Final-Project/models/b6_ideology_model.py)

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

Offline/restricted-network smoke mode:
- add `--offline-random-init` to avoid downloading pretrained weights/tokenizer.
=======
Starter architecture file:
- [b6_hierarchy_draft.py](PSYC-681-Final-Project/Models/b6_hierarchy_draft.py)
>>>>>>> origin/main

Notes:
- [b6_architecture_notes.md](PSYC-681-Final-Project/docs/b6_architecture_notes.md)