from __future__ import annotations

import os
import sqlite3
from os.path import join
from pathlib import Path
from typing import Any

from .types import EmbeddingConfig

_connection: sqlite3.Connection | None = None


def _conn() -> sqlite3.Connection:
    if _connection is None:
        raise RuntimeError("Benchmark database is not initialized. Use DbContex first.")
    return _connection


def close_db():
    global _connection
    if _connection is not None:
        _connection.close()
        _connection = None


def _create_tables(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS classificationreport (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset TEXT NOT NULL,
            task TEXT NOT NULL,
            embedder TEXT NOT NULL,
            model TEXT NOT NULL,
            hyperparams TEXT NOT NULL,
            library_hash TEXT NOT NULL,
            cv_metric_name TEXT NOT NULL,
            cv_metric REAL NOT NULL,
            test_metric_name TEXT NOT NULL,
            test_metric REAL NOT NULL
        )
        """
    )
    connection.commit()


def count_classification_reports(
    *, dataset: str, embedder: str, cv_metric_name: str, model: str
) -> int:
    cursor = _conn().execute(
        """
        SELECT COUNT(*)
        FROM classificationreport
        WHERE dataset = ? AND embedder = ? AND cv_metric_name = ? AND model = ?
        """,
        (dataset, embedder, cv_metric_name, model),
    )
    return int(cursor.fetchone()[0])


def delete_classification_reports(
    *, dataset: str, embedder: str, cv_metric_name: str, model: str
) -> None:
    _conn().execute(
        """
        DELETE FROM classificationreport
        WHERE dataset = ? AND embedder = ? AND cv_metric_name = ? AND model = ?
        """,
        (dataset, embedder, cv_metric_name, model),
    )
    _conn().commit()


def create_classification_report(**row: Any) -> None:
    columns = [
        "dataset",
        "task",
        "embedder",
        "model",
        "hyperparams",
        "library_hash",
        "cv_metric_name",
        "cv_metric",
        "test_metric_name",
        "test_metric",
    ]
    values = [row[column] for column in columns]
    placeholders = ", ".join("?" for _ in columns)
    _conn().execute(
        f"INSERT INTO classificationreport ({', '.join(columns)}) VALUES ({placeholders})",
        values,
    )
    _conn().commit()


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class DbContex(metaclass=Singleton):
    def __init__(self, config: EmbeddingConfig):
        self._config = config

    def __enter__(self):
        global _connection
        database_path = Path(join(os.getcwd(), self._config.database))
        database_path.parent.mkdir(parents=True, exist_ok=True)
        _connection = sqlite3.connect(database_path)
        _create_tables(_connection)
        return None

    def __exit__(self, *args, **kwargs):
        close_db()
