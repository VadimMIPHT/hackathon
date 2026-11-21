#!/usr/bin/env python3
"""Замер гибрида: BM25 + dense, RRF, RM3, дедуп и стратификация.

Пример:
    python3 scripts/hybrid_eval.py --data data
    python3 scripts/hybrid_eval.py --data data \
        --queries data_aug/queries.jsonl --qrels data_aug/qrels.jsonl --split aug
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartmail import metrics  # noqa: E402
from smartmail.dense import DenseIndex, DenseUnavailable  # noqa: E402
from smartmail.engine import SearchEngine  # noqa: E402
from smartmail.hybrid import HybridConfig, HybridSearcher  # noqa: E402
from smartmail.io import load_documents, load_qrels, load_queries  # noqa: E402


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(pct / 100.0 * len(ordered)) - 1)
    return ordered[index]


def run_variant(searcher, queries, qrels, top_k: int) -> dict:
    recalls, ndcgs, mrrs, latency, candidates = [], [], [], [], []
    for query in queries:
        result = searcher.search(query.text, query.mailbox, query.date, top_k=top_k)
        ranked = [hit.doc_id for hit in result.hits]
        relevant = qrels.get(query.query_id, set())
        recalls.append(metrics.recall_at_k(ranked, relevant, top_k))
        ndcgs.append(metrics.ndcg_at_k(ranked, relevant, top_k))
        mrrs.append(metrics.mrr_at_k(ranked, relevant, top_k))
        latency.append(result.elapsed_ms)
        candidates.append(result.candidates)
    total = len(queries)
    return {
        "recall": sum(recalls) / total,
        "ndcg": sum(ndcgs) / total,
        "mrr": sum(mrrs) / total,
        "p50": percentile(latency, 50),
        "p99": percentile(latency, 99),
        "candidates": sum(candidates) / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Гибридный поиск: BM25 + dense + RRF")
    parser.add_argument("--data", default="data")
    parser.add_argument("--queries", default=None)
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dense-model-dir", default=".models/multilingual-e5-small")
    parser.add_argument("--dense-cache", default=".cache/dense_e5_small.npz")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--all-boxes",
        action="store_true",
        help="кодировать все 16k писем, а не только ящики из запросов",
    )
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--subject-boost", type=int, default=1)
    args = parser.parse_args()

    data_dir = Path(args.data)
    all_documents = load_documents(data_dir / "documents.jsonl")
    queries_path = Path(args.queries) if args.queries else data_dir / "queries_dev.jsonl"
    qrels_path = Path(args.qrels) if args.qrels else data_dir / "qrels_dev.jsonl"
    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)

    boxes = {query.mailbox for query in queries}
    documents = [doc for doc in all_documents if args.all_boxes or doc.mailbox in boxes]

    print(
        f"Корпус: {len(documents)} из {len(all_documents)} писем "
        f"({len(boxes)} ящиков из запросов), запросов: {len(queries)}"
    )
    engine = SearchEngine(documents, subject_boost=args.subject_boost)

    dense = None
    if not args.no_dense:
        try:
            print("Считаю/загружаю эмбеддинги...", flush=True)
            dense = DenseIndex(
                documents,
                model_dir=args.dense_model_dir,
                cache_path=args.dense_cache,
                threads=args.threads,
                max_length=args.max_length,
                batch_size=args.batch_size,
                progress=lambda done, total: (
                    print(f"  {done}/{total}", flush=True) if done % 2048 == 0 or done == total else None
                ),
            )
            print(f"Эмбеддинги: {dense.matrix.shape}", flush=True)
        except DenseUnavailable as error:
            print(f"dense недоступен: {error}\nЗапускаю только BM25-варианты.\n")

    configs = [
        ("bm25", HybridConfig(expansion="none")),
        ("bm25+rm3", HybridConfig(expansion="rm3")),
    ]
    if dense is not None:
        configs += [
            ("hybrid", HybridConfig(expansion="none")),
            ("hybrid+rm3", HybridConfig(expansion="rm3")),
            (
                "hybrid+rm3+dedup+strat",
                HybridConfig(expansion="rm3", dedup=True, stratify=True),
            ),
        ]

    print(f"\n{'вариант':<24} {'R@10':>6} {'nDCG@10':>8} {'MRR@10':>7} {'p50':>7} {'p99':>7}")
    results = {}
    for name, config in configs:
        searcher = HybridSearcher(documents, dense, engine=engine, config=config)
        result = run_variant(searcher, queries, qrels, args.top_k)
        results[name] = result
        print(
            f"{name:<24} {result['recall']:6.3f} {result['ndcg']:8.3f} "
            f"{result['mrr']:7.3f} {result['p50']:6.1f}мс {result['p99']:6.1f}мс"
        )

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"hybrid_{args.split}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
