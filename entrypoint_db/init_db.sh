#!/bin/bash
# =========================================
# PostgreSQL Initialisierungsskript
# Wird beim ersten Start automatisch ausgeführt.
# Legt ggf. Tabellen oder Beispieldaten an.
# =========================================

set -e

echo "===> Initialisiere PostgreSQL-Datenbank ..."

# Beispiel: einfache Tabelle für Semantische SQL Tests
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE TABLE IF NOT EXISTS embeddings (
        id SERIAL PRIMARY KEY,
        term VARCHAR(255) UNIQUE NOT NULL,
        vector FLOAT8[]
    );

    CREATE TABLE IF NOT EXISTS queries (
        id SERIAL PRIMARY KEY,
        sql_text TEXT NOT NULL,
        embedding FLOAT8[],
        created_at TIMESTAMP DEFAULT NOW()
    );

    INSERT INTO embeddings (term, vector)
    VALUES
        ('bank', ARRAY[0.12, 0.33, 0.51]),
        ('customer', ARRAY[0.22, 0.44, 0.31])
    ON CONFLICT (term) DO NOTHING;

    CREATE INDEX IF NOT EXISTS idx_embeddings_term ON embeddings(term);
EOSQL

echo "===> PostgreSQL-Initialisierung abgeschlossen."
