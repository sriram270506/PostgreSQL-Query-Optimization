# PostgreSQL 16 + pg_hint_plan (lets scripts.e2e_runtime inject cardinalities via Rows() hints)
FROM postgres:16.4-bookworm
RUN apt-get update \
 && apt-get install -y --no-install-recommends postgresql-16-pg-hint-plan \
 && rm -rf /var/lib/apt/lists/*
