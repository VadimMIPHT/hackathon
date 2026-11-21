"""BM25, который умеет считать скоры только по подмножеству документов.

Ключевая идея бейзлайна: сначала метаданные (ящик + дата) сужают корпус
до 10-300 писем, и только потом BM25 ранжирует этот срез. Поэтому
полноценный поисковый движок с ANN здесь не нужен — точный перебор по
кандидатам быстрее и не теряет recall.

IDF считается в одном из двух режимов:
    global — по всему корпусу (16k писем);
    local  — по кандидатам запроса (письма за нужный день).
Локальный IDF лучше разделяет типы писем внутри одного дня, глобальный
стабильнее на маленьких выборках. Оба режима замеряются в evaluate.py.
"""

from __future__ import annotations

import math
from collections import Counter

IDF_MODES = ("global", "local")


class BM25Index:
    def __init__(
        self,
        token_lists: list[list[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 должен быть > 0")
        if not 0 <= b <= 1:
            raise ValueError("b должен быть в [0, 1]")

        self.k1 = k1
        self.b = b
        self.n_docs = len(token_lists)
        if self.n_docs == 0:
            raise ValueError("пустой корпус")

        self.doc_len = [len(tokens) for tokens in token_lists]
        self.avgdl = sum(self.doc_len) / self.n_docs

        postings: dict[str, dict[int, int]] = {}
        for doc_idx, tokens in enumerate(token_lists):
            for term, tf in Counter(tokens).items():
                postings.setdefault(term, {})[doc_idx] = tf
        self.postings = postings

        self.idf: dict[str, float] = {
            term: math.log(1 + (self.n_docs - len(docs) + 0.5) / (len(docs) + 0.5))
            for term, docs in postings.items()
        }

    def scores(
        self,
        terms: list[str] | tuple[str, ...],
        candidates: list[int] | set[int],
        *,
        idf_mode: str = "local",
    ) -> dict[int, float]:
        """BM25-скоры только для переданных индексов документов."""
        if idf_mode not in IDF_MODES:
            raise ValueError(f"idf_mode должен быть одним из {IDF_MODES}, получено {idf_mode!r}")

        candidate_set = candidates if isinstance(candidates, set) else set(candidates)
        if not candidate_set:
            return {}

        result: dict[int, float] = {doc_idx: 0.0 for doc_idx in candidate_set}
        n_candidates = len(candidate_set)

        for term in terms:
            docs = self.postings.get(term)
            if not docs:
                continue

            if idf_mode == "global":
                idf = self.idf[term]
            else:
                df = sum(1 for doc_idx in docs if doc_idx in candidate_set)
                if df == 0:
                    continue
                idf = math.log(1 + (n_candidates - df + 0.5) / (df + 0.5))

            for doc_idx, tf in docs.items():
                if doc_idx not in candidate_set:
                    continue
                dl = self.doc_len[doc_idx]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                result[doc_idx] += idf * tf * (self.k1 + 1) / denom

        return result
