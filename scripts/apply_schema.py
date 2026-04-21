#!/usr/bin/env python3
"""Apply SQL files in the schema directory to Postgres in filename order."""

import os
from pathlib import Path

try:
    import psycopg2
except ImportError:  # pragma: no cover - runtime dependency
    psycopg2 = None


def main() -> None:
    if psycopg2 is None:
        raise SystemExit("psycopg2-binary is required to run migrations.")
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("DATABASE_URL is required.")

    migrations_dir = Path(os.environ.get("MIGRATIONS_DIR", "sql"))
    sql_files = sorted(migrations_dir.glob("*.sql"))
    if not sql_files:
        raise SystemExit(f"No migrations found in {migrations_dir}")

    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    migration_name TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            conn.commit()

            for migration in sql_files:
                cur.execute(
                    "SELECT 1 FROM schema_migrations WHERE migration_name = %s",
                    (migration.name,),
                )
                if cur.fetchone():
                    print(f"Skipping {migration.name} (already applied)")
                    continue

                sql = migration.read_text(encoding="utf-8")
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO schema_migrations (migration_name) VALUES (%s)",
                    (migration.name,),
                )
                conn.commit()
                print(f"Applied {migration.name}")


if __name__ == "__main__":
    main()
