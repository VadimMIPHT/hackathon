#!/usr/bin/env python3
"""Ручная проверка поиска на одном запросе.

Пример:
    python3 demo.py --query "Покажи уведомления об отклонённых платежах, полученные вчера." \
        --user box-... --date 2028-01-22 --k 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from smartmail.engine import SearchEngine
from smartmail.io import load_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Демо поиска по одному запросу")
    parser.add_argument("--data", default="data")
    parser.add_argument("--query", required=True)
    parser.add_argument("--user", required=True, help="synthetic_user (ящик)")
    parser.add_argument("--date", required=True, help="дата запроса, YYYY-MM-DD")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    documents = load_documents(Path(args.data) / "documents.jsonl")
    engine = SearchEngine(documents)
    result = engine.search(args.query, args.user, args.date, top_k=args.k)

    print(f"Запрос:   {result.query}")
    print(f"Ящик:     {result.mailbox}")
    print(f"Дата:     {result.query_date}  →  целевая дата: {result.target_date}")
    print(f"Термы:    {', '.join(result.terms) if result.terms else '—'}")
    print(f"Кандидатов: {result.candidates}, время: {result.elapsed_ms:.2f} мс\n")

    for rank, hit in enumerate(result.hits, start=1):
        print(f"{rank}. {hit.doc_id}  score={hit.score:.3f}  {hit.doc_dt}")
        print(f"   {hit.snippet}\n")


if __name__ == "__main__":
    main()
