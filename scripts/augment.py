#!/usr/bin/env python3
"""Генерация аугментированного набора запросов.

Примеры:
    # без сети: шаблонные запросы, быстро проверить пайплайн
    python3 scripts/augment.py --data data --out data_aug --backend template --limit 200

    # через OpenAI-совместимый API
    export OPENAI_API_KEY=...
    python3 scripts/augment.py --data data --out data_aug \
        --backend openai --model gpt-4o-mini --limit 2000 --per-email 5 --negatives 10

    # через локальную Ollama
    python3 scripts/augment.py --data data --out data_aug \
        --backend ollama --model qwen2.5:7b --limit 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartmail.augment import QueryGenerator, generate_dataset  # noqa: E402
from smartmail.engine import SearchEngine  # noqa: E402
from smartmail.io import load_documents  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Аугментация запросов для поиска писем")
    parser.add_argument("--data", default="data", help="каталог с documents.jsonl")
    parser.add_argument("--out", default="data_aug", help="каталог для результата")
    parser.add_argument(
        "--backend", choices=("template", "openai", "ollama"), default="template"
    )
    parser.add_argument("--model", default=None, help="модель для openai/ollama")
    parser.add_argument("--base-url", default=None, help="OpenAI-совместимый endpoint")
    parser.add_argument("--per-email", type=int, default=4, help="запросов на письмо")
    parser.add_argument("--limit", type=int, default=500, help="сколько писем обработать")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--negatives",
        type=int,
        default=0,
        help="сколько hard negatives подобрать BM25 (0 — не подбирать)",
    )
    parser.add_argument("--dry-run", action="store_true", help="показать примеры, не писать файлы")
    args = parser.parse_args()

    data_dir = Path(args.data)
    documents = load_documents(data_dir / "documents.jsonl")
    generator = QueryGenerator(
        args.backend, model=args.model, base_url=args.base_url
    )

    engine = None
    if args.negatives:
        print("Собираю BM25-индекс для hard negatives...")
        engine = SearchEngine(documents)

    def progress(done: int, total: int, records: int) -> None:
        print(f"  {done}/{total} писем, запросов: {records}", flush=True)

    print(
        f"Корпус: {len(documents)} писем; backend={args.backend}, "
        f"model={generator.model if args.backend != 'template' else '—'}, "
        f"limit={args.limit}, per_email={args.per_email}"
    )
    records = generate_dataset(
        documents,
        generator=generator,
        per_email=args.per_email,
        limit=args.limit,
        seed=args.seed,
        negatives=args.negatives,
        engine=engine,
        progress=progress,
    )

    if args.dry_run:
        for record in records[:8]:
            print(f"\n[{record['style']}] {record['query']}")
            print(f"  → {record['doc_id']} ({record['query_date']}), negatives={len(record['negatives'])}")
        print(f"\nВсего сгенерировано: {len(records)}")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = [
        {
            "query_id": row["query_id"],
            "synthetic_user": row["synthetic_user"],
            "query": row["query"],
            "query_date": row["query_date"],
        }
        for row in records
    ]
    qrels = [
        {"query_id": row["query_id"], "doc_id": row["doc_id"], "relevance": 1}
        for row in records
    ]

    write_jsonl(out_dir / "queries.jsonl", queries)
    write_jsonl(out_dir / "qrels.jsonl", qrels)
    write_jsonl(out_dir / "pairs.jsonl", records)

    styles = Counter(row["style"] for row in records)
    summary = {
        "documents_used": len({row["doc_id"] for row in records}),
        "queries": len(records),
        "backend": args.backend,
        "model": generator.model if args.backend != "template" else None,
        "per_email": args.per_email,
        "with_negatives": args.negatives,
        "styles": dict(styles),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nГотово: {len(records)} запросов от {summary['documents_used']} писем")
    print("Стили:", dict(styles))
    print(f"Файлы: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
