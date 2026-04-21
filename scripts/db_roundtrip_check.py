#!/usr/bin/env python3
"""Quick DB round-trip check across the core project tables."""

import os
from datetime import datetime, timezone

try:
    import psycopg2
    from psycopg2.extras import Json
except ImportError:  # pragma: no cover - runtime dependency
    psycopg2 = None
    Json = None


def main() -> None:
    if psycopg2 is None or Json is None:
        raise SystemExit("psycopg2-binary is required for DB round-trip check.")
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("DATABASE_URL is required.")

    now = datetime.now(timezone.utc)

    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO raw_posts
                    (post_id, source_dataset, subreddit, user_id, flair, created_at, thread_context, text, topic, raw_json)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (post_id) DO NOTHING
                """,
                (
                    "probe_post_1",
                    "probe",
                    "politics",
                    "probe_user",
                    "libertarian",
                    now,
                    "thread probe context",
                    "This row validates DB read/write wiring.",
                    "probe_topic",
                    Json({"source": "db_probe"}),
                ),
            )

            cur.execute(
                """
                INSERT INTO user_priors
                    (user_id, zecon, zsoc, confidence, method, details)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE
                SET zecon = EXCLUDED.zecon,
                    zsoc = EXCLUDED.zsoc,
                    confidence = EXCLUDED.confidence,
                    method = EXCLUDED.method,
                    details = EXCLUDED.details,
                    updated_at = NOW()
                """,
                (
                    "probe_user",
                    0.2,
                    -0.1,
                    0.8,
                    "probe",
                    Json({"rule": "manual"}),
                ),
            )

            cur.execute(
                """
                INSERT INTO gold_annotations
                    (annotation_id, post_id, annotator_id, annotation_json, evidence_spans)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (annotation_id, annotator_id) DO NOTHING
                """,
                (
                    "probe_ann_1",
                    "probe_post_1",
                    "annotator_a",
                    Json({"q01_relevance": 1, "q07_economic_direction": 0}),
                    Json([{"start_char": 0, "end_char": 4, "text": "This", "type": "claim"}]),
                ),
            )

            cur.execute(
                """
                INSERT INTO gold_aggregates
                    (post_id, majority_json, soft_distribution_json, disagreement_entropy, n_annotations)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (post_id) DO UPDATE
                SET majority_json = EXCLUDED.majority_json,
                    soft_distribution_json = EXCLUDED.soft_distribution_json,
                    disagreement_entropy = EXCLUDED.disagreement_entropy,
                    n_annotations = EXCLUDED.n_annotations,
                    updated_at = NOW()
                """,
                (
                    "probe_post_1",
                    Json({"q01_relevance": 1}),
                    Json({"q01_relevance": {"0": 0.0, "1": 1.0}}),
                    0.0,
                    1,
                ),
            )

            cur.execute(
                """
                INSERT INTO model_outputs
                    (model_name, post_id, split_name, output_json)
                VALUES (%s, %s, %s, %s)
                """,
                ("probe_model", "probe_post_1", "dev", Json({"econ": 0.1, "soc": -0.2})),
            )

            cur.execute(
                """
                INSERT INTO eval_results
                    (model_name, regime, metrics_json)
                VALUES (%s, %s, %s)
                """,
                ("probe_model", "in_domain", Json({"macro_f1": 0.5})),
            )

            conn.commit()

            required_tables = [
                "raw_posts",
                "user_priors",
                "gold_annotations",
                "gold_aggregates",
                "model_outputs",
                "eval_results",
            ]
            for table in required_tables:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                n = cur.fetchone()[0]
                print(f"{table}: {n}")

    print("DB round-trip check passed.")


if __name__ == "__main__":
    main()
