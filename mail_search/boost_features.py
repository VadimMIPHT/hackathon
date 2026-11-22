import argparse
import hashlib
import json
import re
from pathlib import Path
from time import perf_counter

import torch

from mail_search.benchmark import source_hashes
from mail_search.dataset import Dataset, Document, load_archive, read_split
from mail_search.e5 import E5
from mail_search.reranker import Reranker, write_metrics
from mail_search.reranker_data import CandidateQuery, load_candidates
from mail_search.retrieval import CALENDAR_DATE, RRF_DEPTH, searchable_text, tokens

FEATURES = (
    "e5_rank", "e5_cosine", "bm25_rank", "frozen_bge_logit",
    "query_term_coverage", "subject_term_coverage", "query_trigram_coverage",
    "identifier_coverage",
)
SUBJECT_HEADER = re.compile(r"^[ \t]*тема[ \t]*:[ \t]*([^\r\n]*)", re.I | re.M)
QUERY_STOP = frozenset(
    "найди покажи письма письмо уведомления уведомление полученные полученных вчера об о за на".split()
)
TOP_K = 20


def trigrams(terms: set[str]) -> set[str]:
    return {term[i:i + 3] for term in terms for i in range(len(term) - 2)}


def coverage(query_terms: set[str], document_terms: set[str]) -> float:
    return len(query_terms & document_terms) / len(query_terms) if query_terms else 0.0


def lexical_features(query: str, document: Document) -> list[float]:
    query_terms = set(tokens(CALENDAR_DATE.sub("", query))) - QUERY_STOP
    doc_terms = set(tokens(document.text))
    subject = SUBJECT_HEADER.search(document.text)
    subject_terms = set(tokens(subject[1])) if subject else set()
    identifiers = {term for term in query_terms
                   if "@" in term or (len(term) >= 5 and any(ch.isdigit() for ch in term))}
    return [
        coverage(query_terms, doc_terms),
        coverage(query_terms, subject_terms),
        coverage(trigrams(query_terms), trigrams(doc_terms)),
        coverage(identifiers, doc_terms),
    ]


def build_rows(
    dataset: Dataset, groups: list[CandidateQuery], candidate_path: Path,
    e5: E5, bge: Reranker,
) -> list[dict]:
    bm25_rows = json.loads(candidate_path.read_text())["runs"]["query_day"]["per_query"]
    bm25 = {row["query_id"]: {doc: rank for rank, doc in enumerate(row["ranked_doc_ids"], 1)}
            for row in bm25_rows if row["method"] == "bm25"}
    unique_ids = sorted({doc for group in groups for doc in group.ranked_doc_ids[:TOP_K]})
    texts = {doc_id: searchable_text(dataset.documents[doc_id]) for doc_id in unique_ids}
    matrix = e5.encode([texts[doc_id] for doc_id in unique_ids], "passage: ")
    doc_vectors = dict(zip(unique_ids, matrix, strict=True))
    query_vectors = e5.encode([group.query.text for group in groups], "query: ")

    rows = []
    for group, vector in zip(groups, query_vectors, strict=True):
        q = group.query
        ids = group.ranked_doc_ids[:TOP_K]
        started = perf_counter()
        bge_scores = bge.score(q.text, [texts[doc_id] for doc_id in ids])
        features = [
            [rank, float(doc_vectors[doc_id] @ vector), bm25[q.query_id].get(doc_id, RRF_DEPTH + 1),
             logit, *lexical_features(q.text, dataset.documents[doc_id])]
            for rank, (doc_id, logit) in enumerate(zip(ids, bge_scores, strict=True), 1)
        ]
        rows.append({
            "query_id": q.query_id, "mailbox": q.mailbox,
            "e5_ranked_doc_ids": group.ranked_doc_ids,
            "features": features,
            "feature_ms": (perf_counter() - started) * 1000,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed-candidate features for CatBoost ranker.")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--train-candidates", type=Path, required=True)
    parser.add_argument("--val-candidates", type=Path, required=True)
    parser.add_argument("--e5-model", type=Path, required=True)
    parser.add_argument("--bge-model", type=Path, required=True)
    parser.add_argument("--bge-revision", required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(2)
    dataset = load_archive(args.archive)
    e5 = E5(args.e5_model)
    bge = Reranker(args.bge_model, args.bge_revision, max_length=256, train_layers=2)
    bge.model.eval()
    bge.score("прогрев поиска", ["прогрев оценки релевантности"] * TOP_K)
    result: dict = {
        "config": {
            "archive_sha256": dataset.archive_sha256, "top_k": TOP_K,
            "features": FEATURES, "bge_revision": args.bge_revision,
            "code_base_commit": args.base_commit, "source_sha256": source_hashes(),
        },
        "splits": {},
    }
    for split, path in (("train", args.train_candidates), ("val", args.val_candidates)):
        users, _ = read_split(args.splits, dataset, split)
        candidates, retrieval_config = load_candidates(path, dataset, args.splits, users, split)
        if ("e5" in result["config"]) and result["config"]["e5"] != retrieval_config:
            raise ValueError("Train and validation were retrieved with different E5 models")
        result["config"]["e5"] = retrieval_config
        result["config"][f"{split}_candidates_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        result["splits"][split] = build_rows(dataset, candidates, path, e5, bge)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_metrics(args.output, result)


if __name__ == "__main__":
    main()
