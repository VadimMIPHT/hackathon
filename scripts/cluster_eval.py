#!/usr/bin/env python3
"""Кластерная оценка аугментированного набора.

Идея: сгенерированный запрос часто описывает не одно письмо, а группу
похожих (шаблонных) писем. Если считать релевантным только исходное письмо,
метрика занижается: система не может отличить «Успешная оплата» №1 от
«Успешная оплата» №2, да и пользователю это не нужно.

Скрипт строит near-duplicate кластеры (ящик + тема + скелет тела, где
номера и суммы заменены на #), считает метрики по кластерам и может
вычистить false negatives из pairs.jsonl.

Пример:
    python3 scripts/cluster_eval.py --data data --pairs data_aug/pairs.jsonl \
        --out data_aug --filter-negatives
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartmail.engine import SearchEngine  # noqa: E402
from smartmail.io import Document, load_documents  # noqa: E402
from smartmail.text import split_subject_body  # noqa: E402

_ID_RE = re.compile(r"syn-[0-9a-f]+", re.IGNORECASE)
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
_NON_WORD_RE = re.compile(r"[^а-яa-z#]+")


def normalize(value: str, limit: int) -> str:
    value = value.lower().replace("ё", "е")
    value = _ID_RE.sub(" ", value)
    value = _NUM_RE.sub("#", value)
    value = _NON_WORD_RE.sub(" ", value)
    return " ".join(value.split())[:limit]


def cluster_key(document: Document, mode: str) -> tuple[str, ...]:
    subject, body = split_subject_body(document.text)
    if mode == "subject":
        return (document.mailbox, normalize(subject, 80))
    if mode == "body":
        return (document.mailbox, normalize(subject, 80), normalize(body, 160))
    raise ValueError(f"неизвестный режим кластеризации: {mode}")


def build_clusters(
    documents: list[Document], mode: str
) -> tuple[dict[tuple[str, ...], list[str]], dict[str, tuple[str, ...]]]:
    clusters: dict[tuple[str, ...], list[str]] = defaultdict(list)
    doc_key: dict[str, tuple[str, ...]] = {}
    for document in documents:
        key = cluster_key(document, mode)
        clusters[key].append(document.doc_id)
        doc_key[document.doc_id] = key
    return dict(clusters), doc_key


def evaluate(
    engine: SearchEngine,
    pairs: list[dict],
    doc_key: dict[str, tuple[str, str]],
    clusters: dict[tuple[str, str], list[str]],
    top_k: int = 10,
) -> dict[str, float]:
    single_hits: list[float] = []
    cluster_hits: list[float] = []
    precision: list[float] = []
    cluster_mrr: list[float] = []
    for pair in pairs:
        key = doc_key.get(pair["doc_id"])
        if key is None:
            continue
        cluster = set(clusters[key])
        hits = engine.search(
            pair["query"], pair["synthetic_user"], pair["query_date"], top_k=top_k
        ).hits
        ranked = [hit.doc_id for hit in hits]
        single_hits.append(1.0 if pair["doc_id"] in ranked else 0.0)
        found = [position for position, doc_id in enumerate(ranked, start=1) if doc_id in cluster]
        cluster_hits.append(1.0 if found else 0.0)
        precision.append(len(found) / top_k)
        cluster_mrr.append(1.0 / found[0] if found else 0.0)
    size = len(single_hits)
    return {
        "queries": size,
        "single_hit@10": sum(single_hits) / size,
        "cluster_hit@10": sum(cluster_hits) / size,
        "cluster_precision@10": sum(precision) / size,
        "cluster_mrr@10": sum(cluster_mrr) / size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Near-duplicate кластеры и честные метрики")
    parser.add_argument("--data", default="data")
    parser.add_argument("--pairs", default="data_aug/pairs.jsonl")
    parser.add_argument("--out", default="data_aug")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--filter-negatives", action="store_true")
    parser.add_argument(
        "--cluster-mode",
        choices=("subject", "body"),
        default="subject",
        help="subject — один кластер на тему; body — ещё и скелет тела (строже)",
    )
    args = parser.parse_args()

    documents = load_documents(Path(args.data) / "documents.jsonl")
    pairs = [json.loads(line) for line in Path(args.pairs).open(encoding="utf-8") if line.strip()]
    engines = {boost: SearchEngine(documents, subject_boost=boost) for boost in (1, 2)}

    for mode in ("subject", "body"):
        clusters, doc_key = build_clusters(documents, mode)
        sizes = sorted(len(value) for value in clusters.values())
        query_clusters = {doc_key[pair["doc_id"]] for pair in pairs if pair["doc_id"] in doc_key}
        print(
            f"\n[{mode}] кластеров: {len(sizes)}, медиана {statistics.median(sizes):.0f}, "
            f"p90 {sizes[int(0.9 * len(sizes))]}, max {sizes[-1]}, "
            f"затронуто запросами: {len(query_clusters)} (медиана "
            f"{statistics.median([len(clusters[key]) for key in query_clusters]):.0f})"
        )
        for boost, engine in engines.items():
            metrics = evaluate(engine, pairs, doc_key, clusters, top_k=args.top_k)
            print(
                f"  boost={boost}: single_hit={metrics['single_hit@10']:.3f}  "
                f"cluster_hit={metrics['cluster_hit@10']:.3f}  "
                f"P@10={metrics['cluster_precision@10']:.3f}  "
                f"cluster_MRR={metrics['cluster_mrr@10']:.3f}"
            )

    clusters, doc_key = build_clusters(documents, args.cluster_mode)

    out_dir = Path(args.out)
    summary = {
        "cluster_mode": args.cluster_mode,
        "clusters": len(clusters),
        "queries": len(pairs),
    }
    (out_dir / "cluster_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.filter_negatives:
        cleaned = []
        removed = 0
        for pair in pairs:
            key = doc_key.get(pair["doc_id"])
            cluster = set(clusters[key]) if key else set()
            negatives = [doc_id for doc_id in pair.get("negatives", []) if doc_id not in cluster]
            removed += len(pair.get("negatives", [])) - len(negatives)
            cleaned.append({**pair, "negatives": negatives})
        with (out_dir / "pairs_clean.jsonl").open("w", encoding="utf-8") as handle:
            for row in cleaned:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Убрано false negatives: {removed}; файл: {out_dir / 'pairs_clean.jsonl'}")


if __name__ == "__main__":
    main()
