#!/bin/bash
set -e

echo "Запуск контейнера с БД..."
docker compose up -d db

echo "Запуск приложения (ожидание БД встроено в Python)..."
python main.py

cleanup() {
    echo "Остановка контейнера с БД..."
    docker compose down db
}
trap cleanup EXIT