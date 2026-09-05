"""Small JSON/CSV writers for standalone CPU evaluation and archived predictions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def clean_json(value):
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(file_path: Path, value) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(clean_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def append_json(file_path: Path, value) -> None:
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(clean_json(value), sort_keys=True, allow_nan=False) + "\n"
        )


def write_csv(file_path: Path, rows: list[dict]) -> None:
    columns = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if not isinstance(value, (dict, list))
        }
    )
    with file_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(clean_json(row) for row in rows)
