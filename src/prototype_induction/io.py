# === FILE META START ===
# File: /src/prototype_induction/io.py
# Category: Shared utility
# Author: project team
# Last Updated: 2026-08-19
# Purpose: Provide crash-safe JSONL reads, atomic writes, and append helpers.
# Input: JSON-compatible records and local file paths.
# Output: Valid JSONL files with fsynced append operations.
# === FILE META END ===
"""Crash-safe JSONL input/output helpers shared by evaluation workflows."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    fd = os.open(target, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, encoded.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def write_jsonl_atomic(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)


def completed_ids(path: str | Path, key: str = "run_id") -> set[str]:
    target = Path(path)
    if not target.exists():
        return set()
    return {str(record[key]) for record in read_jsonl(target) if key in record}
