from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def resolve_path(path_like: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    if base_dir is None:
        return path
    return (base_dir / path).resolve()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)
