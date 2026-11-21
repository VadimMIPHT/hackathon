"""Загрузка корпуса, запросов и qrels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    mailbox: str
    text: str
    date: date | None


@dataclass(frozen=True, slots=True)
class Query:
    query_id: str
    mailbox: str
    text: str
    date: date


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def load_documents(path: str | Path) -> list[Document]:
    rows = _read_jsonl(Path(path))
    return [
        Document(
            doc_id=row["doc_id"],
            mailbox=row["synthetic_user"],
            text=row["doc"],
            date=_parse_date(row.get("doc_dt")),
        )
        for row in rows
    ]


def load_queries(path: str | Path) -> list[Query]:
    rows = _read_jsonl(Path(path))
    return [
        Query(
            query_id=row["query_id"],
            mailbox=row["synthetic_user"],
            text=row["query"],
            date=date.fromisoformat(row["query_date"]),
        )
        for row in rows
    ]


def load_qrels(path: str | Path) -> dict[str, set[str]]:
    """query_id -> множество релевантных doc_id."""
    result: dict[str, set[str]] = {}
    for row in _read_jsonl(Path(path)):
        result.setdefault(row["query_id"], set()).add(row["doc_id"])
    return result
