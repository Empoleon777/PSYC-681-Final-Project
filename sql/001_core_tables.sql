CREATE TABLE IF NOT EXISTS raw_posts (
    post_id TEXT PRIMARY KEY,
    source_dataset TEXT NOT NULL,
    subreddit TEXT NOT NULL,
    user_id TEXT NOT NULL,
    flair TEXT,
    created_at TIMESTAMPTZ,
    thread_context TEXT,
    text TEXT NOT NULL,
    topic TEXT,
    raw_json JSONB,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_raw_posts_subreddit ON raw_posts (subreddit);
CREATE INDEX IF NOT EXISTS idx_raw_posts_user_id ON raw_posts (user_id);
CREATE INDEX IF NOT EXISTS idx_raw_posts_created_at ON raw_posts (created_at);

CREATE TABLE IF NOT EXISTS user_priors (
    user_id TEXT PRIMARY KEY,
    zecon DOUBLE PRECISION NOT NULL,
    zsoc DOUBLE PRECISION NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    method TEXT NOT NULL,
    details JSONB,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS gold_annotations (
    annotation_id TEXT NOT NULL,
    post_id TEXT NOT NULL REFERENCES raw_posts(post_id) ON DELETE CASCADE,
    annotator_id TEXT NOT NULL,
    annotation_json JSONB NOT NULL,
    evidence_spans JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (annotation_id, annotator_id)
);

CREATE INDEX IF NOT EXISTS idx_gold_annotations_post ON gold_annotations (post_id);

CREATE TABLE IF NOT EXISTS gold_aggregates (
    post_id TEXT PRIMARY KEY REFERENCES raw_posts(post_id) ON DELETE CASCADE,
    majority_json JSONB NOT NULL,
    soft_distribution_json JSONB NOT NULL,
    disagreement_entropy DOUBLE PRECISION NOT NULL,
    n_annotations INT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_outputs (
    output_id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    post_id TEXT NOT NULL REFERENCES raw_posts(post_id) ON DELETE CASCADE,
    split_name TEXT,
    output_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_model_outputs_model ON model_outputs (model_name);
CREATE INDEX IF NOT EXISTS idx_model_outputs_post ON model_outputs (post_id);

CREATE TABLE IF NOT EXISTS eval_results (
    result_id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    regime TEXT NOT NULL,
    metrics_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS schema_migrations (
    migration_name TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
