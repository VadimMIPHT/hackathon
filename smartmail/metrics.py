"""Метрики качества выдачи: Recall@K, nDCG@K, MRR@K.

Релевантность бинарная (в qrels_dev значение 1), поэтому nDCG считается
по бинарным гейнам.
"""

from __future__ import annotations

import math

Ranked = list[str] | tuple[str, ...]


def recall_at_k(ranked: Ranked, relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(ranked[:k]) & relevant)
    return hits / len(relevant)


def dcg_at_k(ranked: Ranked, relevant: set[str], k: int) -> float:
    return sum(
        1.0 / math.log2(position + 2)
        for position, doc_id in enumerate(ranked[:k])
        if doc_id in relevant
    )


def ndcg_at_k(ranked: Ranked, relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    ideal = sum(1.0 / math.log2(position + 2) for position in range(min(len(relevant), k)))
    if ideal == 0:
        return 0.0
    return dcg_at_k(ranked, relevant, k) / ideal


def mrr_at_k(ranked: Ranked, relevant: set[str], k: int) -> float:
    for position, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def first_relevant_rank(ranked: Ranked, relevant: set[str], k: int) -> int | None:
    for position, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return position
    return None
