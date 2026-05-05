# PSYC-681 Final Project

This repo contains our current working pipeline for:

- Reddit ingestion into a normalized post table
- weak-label construction from lightweight lexical/rule sources
- annotation packet creation and aggregation
- an initial draft of the B6 hierarchical architecture

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
export DATABASE_URL=postgresql://USERNAME_HERE:PASSWORD_HERE@localhost:5432/psyc681_ideology
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

By default, the annotation packet is seeded with weak-label suggestions (`q01..q12`) so batches are not blank.
Use `--blank-fields` if you want empty annotation fields instead.

Fold three annotator files into aggregate outputs:

```bash
python scripts/fold_annotations.py \
  --annotation-glob "Data/annotation_60k/annotation_packet_annotator_*.csv" \
  --output-dir outputs/annotation_60k
```

## Label Definitions

Canonical label schema:
- [label_schema.md](PSYC-681-Final-Project/docs/label_schema.md)

## B6 Draft

Starter architecture file:
- [b6_hierarchy.py](PSYC-681-Final-Project/Models/b6_hierarchy.py)

Notes:
- [b6_architecture_notes.md](PSYC-681-Final-Project/docs/b6_architecture_notes.md)