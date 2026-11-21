"""Поисковый движок бейзлайна: фильтр по ящику и дате + BM25."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from time import perf_counter

from . import text as textmod
from .dates import extract_date, strip_span
from .index import BM25Index
from .io import Document


@dataclass(frozen=True, slots=True)
class SearchHit:
    doc_id: str
    score: float
    doc_dt: date | None
    snippet: str


@dataclass(frozen=True, slots=True)
class SearchResult:
    query: str
    mailbox: str
    query_date: date
    target_date: date | None
    terms: tuple[str, ...]
    candidates: int
    hits: tuple[SearchHit, ...]
    elapsed_ms: float
    target_end: date | None = None


def _as_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


class SearchEngine:
    """Индекс строится один раз, поиск — только по срезу кандидатов."""

    def __init__(
        self,
        documents: list[Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        idf_mode: str = "global",
        stemmer: textmod.RussianStemmer | None = None,
        soft_prefix_len: int = textmod.SOFT_PREFIX_LEN,
        subject_boost: int = 1,
        snippet_len: int = 220,
    ) -> None:
        if not documents:
            raise ValueError("пустой корпус")
        if subject_boost < 1:
            raise ValueError("subject_boost должен быть >= 1")

        self._docs = list(documents)
        self._stemmer = stemmer or textmod.RussianStemmer()
        self._idf_mode = idf_mode
        self._soft_prefix_len = soft_prefix_len
        self._subject_boost = subject_boost
        self._snippet_len = snippet_len

        self._by_box: dict[str, list[int]] = {}
        self._dates: list[date | None] = []
        self._snippets: list[str] = []

        token_lists: list[list[str]] = []
        for doc_idx, doc in enumerate(self._docs):
            self._by_box.setdefault(doc.mailbox, []).append(doc_idx)
            self._dates.append(doc.date)
            self._snippets.append(self._make_snippet(doc.text))
            source = doc.text
            if self._subject_boost > 1:
                subject, body = textmod.split_subject_body(doc.text)
                if subject:
                    source = (subject + " ") * self._subject_boost + body
            token_lists.append(
                textmod.tokenize(
                    source, self._stemmer, soft_prefix_len=self._soft_prefix_len
                )
            )

        self._tokens = token_lists
        self._index = BM25Index(token_lists, k1=k1, b=b)

    @property
    def documents(self) -> list[Document]:
        return self._docs

    @property
    def idf(self) -> dict[str, float]:
        return self._index.idf

    def tokens(self, doc_idx: int) -> list[str]:
        return self._tokens[doc_idx]

    def snippet(self, doc_idx: int) -> str:
        return self._snippets[doc_idx]

    def score_bm25(self, terms, candidates) -> dict[int, float]:
        return self._index.scores(terms, candidates, idf_mode=self._idf_mode)

    def candidates(
        self,
        mailbox: str,
        query_date: date | str,
        target_range: tuple[date, date] | None = None,
    ) -> list[int]:
        return self._candidates(
            mailbox, query_date=_as_date(query_date), target_range=target_range
        )

    def _make_snippet(self, value: str) -> str:
        flat = " ".join(value.split())
        if len(flat) <= self._snippet_len:
            return flat
        return flat[: self._snippet_len - 1] + "…"

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
        if top_k <= 0:
            raise ValueError("top_k должен быть > 0")

        started = perf_counter()
        query_date = _as_date(query_date)

        normalized = textmod.normalize(query)
        parsed = extract_date(normalized, query_date)
        body = strip_span(normalized, parsed.span)
        terms = (
            tuple(
                textmod.query_terms(
                    body, self._stemmer, soft_prefix_len=self._soft_prefix_len
                )
            )
            if use_text
            else ()
        )

        target_range = parsed.bounds() if use_date_filter else None
        candidates = self._candidates(
            mailbox,
            query_date=query_date,
            target_range=target_range,
        )

        if terms and candidates:
            scores = self._index.scores(terms, candidates, idf_mode=self._idf_mode)
        else:
            scores = {}

        ranked = sorted(
            candidates,
            key=lambda doc_idx: (
                -scores.get(doc_idx, 0.0),
                -(self._dates[doc_idx].toordinal() if self._dates[doc_idx] else 0),
                self._docs[doc_idx].doc_id,
            ),
        )[:top_k]

        hits = tuple(
            SearchHit(
                doc_id=self._docs[doc_idx].doc_id,
                score=round(scores.get(doc_idx, 0.0), 6),
                doc_dt=self._dates[doc_idx],
                snippet=self._snippets[doc_idx],
            )
            for doc_idx in ranked
        )
        elapsed_ms = (perf_counter() - started) * 1000.0

        return SearchResult(
            query=query,
            mailbox=mailbox,
            query_date=query_date,
            target_date=parsed.date,
            terms=terms,
            candidates=len(candidates),
            hits=hits,
            elapsed_ms=elapsed_ms,
            target_end=parsed.end_date,
        )

    def _candidates(
        self,
        mailbox: str,
        *,
        query_date: date,
        target_range: tuple[date, date] | None,
    ) -> list[int]:
        """Письма ящика, доступные на дату запроса.

        Письма с известной датой позже даты запроса недоступны по условию.
        Письма с неизвестной датой (doc_dt = null) не могут подтвердить
        попадание в конкретный день/диапазон, поэтому при точном фильтре
        отбрасываются.
        """
        result: list[int] = []
        for doc_idx in self._by_box.get(mailbox, []):
            doc_date = self._dates[doc_idx]
            if doc_date is None:
                if target_range is None:
                    result.append(doc_idx)
                continue
            if doc_date > query_date:
                continue
            if target_range is not None:
                start, end = target_range
                if not (start <= doc_date <= end):
                    continue
            result.append(doc_idx)
        return result
