#!/bin/bash
set -e

DB_HOST=${DB_HOST:-"db"}
DB_PORT=${DB_PORT:-5432}
DB_USER=${DB_USER:-"testuser"}

echo "Warte auf PostgreSQL unter ${DB_HOST}:${DB_PORT}..."
until pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER"; do
    sleep 2
done



echo "PostgreSQL ist erreichbar."

# Jupyter Lab starten
exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/workspace
