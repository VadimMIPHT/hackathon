"""Пакет бейзлайна поиска писем: BM25 по срезу "ящик + дата"."""

from .engine import SearchEngine, SearchHit, SearchResult
from .io import Document, Query, load_documents, load_qrels, load_queries

__all__ = [
    "Document",
    "Query",
    "SearchEngine",
    "SearchHit",
    "SearchResult",
    "load_documents",
    "load_qrels",
    "load_queries",
]
