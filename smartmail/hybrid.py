"""Гибридный поиск: BM25 + dense, RRF, query expansion, дедуп и стратификация.

Идея: BM25 хорошо ловит точные слова и сущности, dense — пересказ смысла.
Списки объединяются через Reciprocal Rank Fusion, после чего одинаковые
письма схлопываются, а выдача стратифицируется по группам (отправитель
или тема), чтобы топ не состоял из десяти копий одного письма.
"""

from __future__ import annotations

import re
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import date
from time import perf_counter

from . import text as textmod
from .dates import extract_date, strip_span
from .engine import SearchEngine, SearchHit, SearchResult

_NORM_RE = re.compile(r"[^а-яa-z#]+")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+")


@dataclass(frozen=True, slots=True)
class HybridConfig:
    rrf_k: int = 60
    bm25_top: int = 50
    dense_top: int = 50
    expansion: str = "none"  # none | rm3
    expansion_terms: int = 5
    expansion_docs: int = 5
    dedup: bool = False
    stratify: bool = False
    group_by: str | None = None  # sender | cluster | None (авто)
    sender_filter: bool = True


def _normalize_key(value: str, limit: int = 100) -> str:
    value = value.lower().replace("ё", "е")
    value = re.sub(r"\d+", "#", value)
    value = _NORM_RE.sub(" ", value)
    return " ".join(value.split())[:limit]


class HybridSearcher:
    def __init__(
        self,
        documents,
        dense=None,
        *,
        engine: SearchEngine | None = None,
        config: HybridConfig | None = None,
        subject_boost: int = 1,
    ) -> None:
        self.documents = list(documents)
        self.engine = engine or SearchEngine(self.documents, subject_boost=subject_boost)
        self.dense = dense
        self.config = config or HybridConfig()
        self._senders = [textmod.extract_sender(document.text) for document in self.documents]

    # ------------------------------------------------------------------ search

    def search(
        self,
        query: str,
        mailbox: str,
        query_date: date | str,
        top_k: int = 10,
        *,
        use_date_filter: bool = True,
        use_text: bool = True,
    ) -> SearchResult:
        started = perf_counter()
        query_date = query_date if isinstance(query_date, date) else date.fromisoformat(query_date)

        normalized = textmod.normalize(query)
        parsed = extract_date(normalized, query_date)
        body = strip_span(normalized, parsed.span)
        terms = tuple(textmod.query_terms(body, self.engine._stemmer)) if use_text else ()

        candidates = self.engine.candidates(
            mailbox, query_date, parsed.date if use_date_filter else None
        )

        sender = self._detect_sender(body) if self.config.sender_filter else None
        if sender:
            filtered = [
                index for index in candidates if (self._senders[index] or "").lower() == sender
            ]
            if filtered:
                candidates = filtered

        bm25_scores = self.engine.score_bm25(terms, candidates) if terms else {}
        if terms and self.config.expansion == "rm3":
            extra = self._rm3_terms(terms, candidates)
            if extra:
                terms = tuple(dict.fromkeys(terms + tuple(extra)))
                bm25_scores = self.engine.score_bm25(terms, candidates)

        bm25_ranking = self._ranking(bm25_scores)[: self.config.bm25_top]
        dense_ranking: list[int] = []
        if self.dense is not None and use_text:
            dense_hits = self.dense.search(body, candidates, self.config.dense_top)
            dense_ranking = [index for index, _ in dense_hits]

        fused = self._rrf([bm25_ranking, dense_ranking])
        ordered = self._ranking(fused)

        if self.config.dedup:
            ordered = self._dedup(ordered)
        if self.config.stratify:
            ordered = self._stratify(ordered, sender_detected=bool(sender))

        hits = tuple(
            SearchHit(
                doc_id=self.documents[index].doc_id,
                score=round(fused.get(index, 0.0), 6),
                doc_dt=self.documents[index].date,
                snippet=self.engine.snippet(index),
            )
            for index in ordered[:top_k]
        )
        return SearchResult(
            query=query,
            mailbox=mailbox,
            query_date=query_date,
            target_date=parsed.date,
            terms=terms,
            candidates=len(candidates),
            hits=hits,
            elapsed_ms=(perf_counter() - started) * 1000.0,
        )

    # ------------------------------------------------------------------ fusion

    def _ranking(self, scores: dict[int, float]) -> list[int]:
        return sorted(
            scores,
            key=lambda index: (
                -scores[index],
                -(self.documents[index].date.toordinal() if self.documents[index].date else 0),
                self.documents[index].doc_id,
            ),
        )

    def _rrf(self, rankings: list[list[int]]) -> dict[int, float]:
        fused: dict[int, float] = defaultdict(float)
        for ranking in rankings:
            for rank, index in enumerate(ranking, start=1):
                fused[index] += 1.0 / (self.config.rrf_k + rank)
        return dict(fused)

    def _rm3_terms(self, terms: tuple[str, ...], candidates: list[int]) -> list[str]:
        scores = self.engine.score_bm25(terms, candidates)
        top_docs = self._ranking(scores)[: self.config.expansion_docs]
        counts: Counter[str] = Counter()
        for index in top_docs:
            counts.update(token for token in self.engine.tokens(index) if len(token) >= 4)
        known = set(terms)
        scored = [
            (count * self.engine.idf.get(token, 0.0), token)
            for token, count in counts.items()
            if token not in known
        ]
        scored.sort(reverse=True)
        return [token for _, token in scored[: self.config.expansion_terms]]

    # ------------------------------------------------------------------- groups

    def _detect_sender(self, query: str) -> str | None:
        lower = query.lower()
        known = {sender.lower() for sender in self._senders if sender}
        found = {sender for sender in known if sender in lower}
        if not found:
            for match in _EMAIL_RE.findall(lower):
                if match in known:
                    found.add(match)
        return next(iter(found)) if len(found) == 1 else None

    def _cluster_key(self, index: int) -> tuple[str, str, str]:
        subject, _ = textmod.split_subject_body(self.documents[index].text)
        return (
            self.documents[index].mailbox,
            self._senders[index] or "",
            _normalize_key(subject, 80),
        )

    def _group_key(self, index: int, sender_detected: bool) -> object:
        mode = self.config.group_by or ("cluster" if sender_detected else "sender")
        if mode == "sender":
            return (self.documents[index].mailbox, self._senders[index] or "")
        return self._cluster_key(index)

    def _dedup(self, ordered: list[int]) -> list[int]:
        seen: set[tuple[str, str]] = set()
        result: list[int] = []
        for index in ordered:
            key = self._cluster_key(index)
            if key in seen:
                continue
            seen.add(key)
            result.append(index)
        return result

    def _stratify(self, ordered: list[int], *, sender_detected: bool) -> list[int]:
        groups: OrderedDict[object, list[int]] = OrderedDict()
        for index in ordered:
            groups.setdefault(self._group_key(index, sender_detected), []).append(index)
        result: list[int] = []
        depth = 0
        while len(result) < len(ordered):
            added = False
            for items in groups.values():
                if depth < len(items):
                    result.append(items[depth])
                    added = True
            if not added:
                break
            depth += 1
        return result
