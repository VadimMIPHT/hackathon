#!/usr/bin/env python3
"""Замер бейзлайна на dev-наборе.

Примеры:
    python3 evaluate.py --data data
    python3 evaluate.py --data data --ablation
    python3 evaluate.py --data data --out reports --split dev
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from time import perf_counter

from smartmail import metrics
from smartmail.engine import SearchEngine
from smartmail.io import load_documents, load_qrels, load_queries

KS = (1, 5, 10, 20, 50, 100)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(pct / 100.0 * len(ordered)) - 1)
    return ordered[index]


def run_queries(
    engine: SearchEngine,
    queries,
    qrels: dict[str, set[str]],
    *,
    use_date_filter: bool,
    use_text: bool,
    top_k: int = max(KS),
) -> list[dict]:
    rows: list[dict] = []
    for query in queries:
        result = engine.search(
            query.text,
            query.mailbox,
            query.date,
            top_k=top_k,
            use_date_filter=use_date_filter,
            use_text=use_text,
        )
        ranked = [hit.doc_id for hit in result.hits]
        relevant = qrels.get(query.query_id, set())
        row = {
            "query_id": query.query_id,
            "query": query.text,
            "mailbox": query.mailbox,
            "query_date": query.date.isoformat(),
            "target_date": result.target_date.isoformat() if result.target_date else None,
            "terms": list(result.terms),
            "n_relevant": len(relevant),
            "n_candidates": result.candidates,
            "n_returned": len(ranked),
            "latency_ms": result.elapsed_ms,
            "first_relevant_rank": metrics.first_relevant_rank(ranked, relevant, top_k),
            "covers_all_relevant": relevant.issubset(
                {hit.doc_id for hit in result.hits}
            ),
        }
        for k in KS:
            row[f"recall@{k}"] = metrics.recall_at_k(ranked, relevant, k)
            row[f"ndcg@{k}"] = metrics.ndcg_at_k(ranked, relevant, k)
            row[f"mrr@{k}"] = metrics.mrr_at_k(ranked, relevant, k)
        rows.append(row)
    return rows


def aggregate(rows: list[dict]) -> dict[str, float]:
    total = len(rows)
    summary: dict[str, float] = {"queries": float(total)}
    for k in KS:
        summary[f"recall@{k}"] = sum(row[f"recall@{k}"] for row in rows) / total
        summary[f"ndcg@{k}"] = sum(row[f"ndcg@{k}"] for row in rows) / total
        summary[f"mrr@{k}"] = sum(row[f"mrr@{k}"] for row in rows) / total
    single = [row for row in rows if row["n_relevant"] == 1]
    summary["queries_single_relevant"] = float(len(single))
    if single:
        summary["mrr@10_single_relevant"] = sum(row["mrr@10"] for row in single) / len(single)
    done = [row for row in rows if row["target_date"]]
    summary["queries_with_date"] = float(len(done))
    summary["mean_candidates"] = sum(row["n_candidates"] for row in rows) / total
    summary["max_candidates"] = float(max(row["n_candidates"] for row in rows))
    return summary


def latency_stats(values: list[float]) -> dict[str, float]:
    return {
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "max_ms": max(values),
        "mean_ms": statistics.fmean(values),
    }


def measure_latency(engine: SearchEngine, queries, *, repeats: int = 3) -> dict[str, float]:
    """Латентность продового вызова top_k=10, без холодного старта."""
    for query in queries[:5]:
        engine.search(query.text, query.mailbox, query.date, top_k=10)

    samples: list[float] = []
    for _ in range(repeats):
        for query in queries:
            started = perf_counter()
            engine.search(query.text, query.mailbox, query.date, top_k=10)
            samples.append((perf_counter() - started) * 1000.0)
    return latency_stats(samples)


def variant_rows(
    engine: SearchEngine,
    queries,
    qrels,
    *,
    use_date_filter: bool,
    use_text: bool,
) -> dict:
    rows = run_queries(
        engine,
        queries,
        qrels,
        use_date_filter=use_date_filter,
        use_text=use_text,
    )
    summary = aggregate(rows)
    summary.update(measure_latency(engine, queries))
    return {"rows": rows, "summary": summary}


def print_summary(name: str, summary: dict[str, float]) -> None:
    print(
        f"{name:<14} "
        f"R@1={summary['recall@1']:.3f}  R@5={summary['recall@5']:.3f}  "
        f"R@10={summary['recall@10']:.3f}  nDCG@10={summary['ndcg@10']:.3f}  "
        f"MRR@10={summary['mrr@10']:.3f}  "
        f"p99={summary['p99_ms']:.2f}ms"
    )


def write_reports(
    out_dir: Path,
    *,
    split: str,
    dataset: dict,
    results: dict,
    ablation: dict | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "split": split,
        "dataset": dataset,
        "config": results["config"],
        "summary": results["summary"],
        "per_query": results["rows"],
    }
    (out_dir / f"{split}_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    with (out_dir / f"per_query_{split}.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "query_id", "query", "query_date", "target_date", "n_relevant",
            "n_candidates", "recall@10", "ndcg@10", "mrr@10",
            "first_relevant_rank", "latency_ms", "covers_all_relevant",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results["rows"]:
            writer.writerow({key: row[key] for key in fieldnames})

    summary = results["summary"]
    lines = [
        f"# Бейзлайн поиска писем · {split}",
        "",
        f"- Писем в корпусе: {dataset['documents']}",
        f"- Запросов: {int(summary['queries'])}",
        f"- Запросов с найденной датой: {int(summary['queries_with_date'])}",
        f"- Кандидатов на запрос: в среднем {summary['mean_candidates']:.1f}, "
        f"максимум {int(summary['max_candidates'])}",
        "",
        "## Качество",
        "",
        "| Метрика | Значение |",
        "| --- | ---: |",
        f"| Recall@1 | {summary['recall@1']:.3f} |",
        f"| Recall@5 | {summary['recall@5']:.3f} |",
        f"| Recall@10 | {summary['recall@10']:.3f} |",
        f"| nDCG@10 | {summary['ndcg@10']:.3f} |",
        f"| MRR@10 | {summary['mrr@10']:.3f} |",
        f"| MRR@10 (запросы с 1 релевантным) | {summary['mrr@10_single_relevant']:.3f} |",
        "",
        "## Латентность (top_k=10, 3 прогона)",
        "",
        "| p50 | p95 | p99 | max |",
        "| ---: | ---: | ---: | ---: |",
        f"| {summary['p50_ms']:.2f} мс | {summary['p95_ms']:.2f} мс | "
        f"{summary['p99_ms']:.2f} мс | {summary['max_ms']:.2f} мс |",
        "",
    ]

    if ablation:
        lines += [
            "## Абляция",
            "",
            "| Вариант | Recall@10 | nDCG@10 | MRR@10 | p99 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name, item in ablation.items():
            item_summary = item["summary"]
            lines.append(
                f"| {name} | {item_summary['recall@10']:.3f} | "
                f"{item_summary['ndcg@10']:.3f} | {item_summary['mrr@10']:.3f} | "
                f"{item_summary['p99_ms']:.2f} мс |"
            )
        lines.append("")

    lines += [
        "## Что важно знать",
        "",
        "Все 99 публичных запросов устроены как «<тип письма>, полученные <дата>»,",
        "поэтому основной вклад даёт фильтр по дате, а BM25 лишь упорядочивает",
        "письма внутри дня. На закрытом бенчмарке формулировки могут быть",
        "смысловыми и без даты — для этого случая в движке остаётся текстовый",
        "поиск по всему ящику (см. абляцию `text_only`).",
        "",
    ]
    (out_dir / f"{split}_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Бейзлайн поиска писем: ящик + дата + BM25")
    parser.add_argument("--data", default="data", help="каталог с documents/queries/qrels")
    parser.add_argument("--out", default="reports", help="каталог для отчётов")
    parser.add_argument("--split", default="dev", help="имя сплита для файлов отчёта")
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument(
        "--idf",
        choices=("global", "local"),
        default="global",
        help="IDF по всему корпусу или по кандидатам дня",
    )
    parser.add_argument("--ablation", action="store_true", help="прогнать варианты без даты/текста")
    args = parser.parse_args()

    data_dir = Path(args.data)
    documents = load_documents(data_dir / "documents.jsonl")
    queries = load_queries(data_dir / "queries_dev.jsonl")
    qrels = load_qrels(data_dir / "qrels_dev.jsonl")

    started = perf_counter()
    engine = SearchEngine(documents, k1=args.k1, b=args.b, idf_mode=args.idf)
    build_ms = (perf_counter() - started) * 1000.0

    print(
        f"Корпус: {len(documents)} писем, {len({d.mailbox for d in documents})} ящиков; "
        f"запросов: {len(queries)}; индекс собран за {build_ms / 1000:.2f} с"
    )
    print(f"k1={args.k1}, b={args.b}, idf={args.idf}, top_k=10\n")

    results = variant_rows(engine, queries, qrels, use_date_filter=True, use_text=True)
    results["config"] = {
        "k1": args.k1,
        "b": args.b,
        "idf_mode": args.idf,
        "use_date_filter": True,
        "use_text": True,
        "build_ms": build_ms,
    }
    print_summary("full", results["summary"])
    print(
        f"  покрытие qrels фильтром: "
        f"{sum(row['covers_all_relevant'] for row in results['rows'])}/{len(results['rows'])} запросов; "
        f"среднее кандидатов: {results['summary']['mean_candidates']:.1f}"
    )

    ablation = None
    if args.ablation:
        print()
        ablation = {}
        local_engine = SearchEngine(documents, k1=args.k1, b=args.b, idf_mode="local")
        ablation["full"] = results
        ablation["local_idf"] = variant_rows(
            local_engine, queries, qrels, use_date_filter=True, use_text=True
        )
        ablation["date_only"] = variant_rows(
            engine, queries, qrels, use_date_filter=True, use_text=False
        )
        ablation["text_only"] = variant_rows(
            engine, queries, qrels, use_date_filter=False, use_text=True
        )
        for name, item in ablation.items():
            if name == "full":
                continue
            print_summary(name, item["summary"])

    dataset = {
        "documents": len(documents),
        "queries": len(queries),
        "qrels": sum(len(value) for value in qrels.values()),
    }
    write_reports(
        Path(args.out),
        split=args.split,
        dataset=dataset,
        results=results,
        ablation=ablation,
    )
    print(f"\nОтчёты: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
