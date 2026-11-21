import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from mail_search.dataset import Dataset, Document, Query
from mail_search.retrieval import RRF_DEPTH, day_from_query, eligible_indices, searchable_text


@dataclass(frozen=True)
class TrainingPair:
    query: str
    document: str
    label: int
    weight: float


@dataclass(frozen=True)
class CandidateQuery:
    query: Query
    ranked_doc_ids: list[str]
    retrieval_ms: float


def documents_by_mailbox(dataset: Dataset) -> dict[str, list[Document]]:
    by_mailbox: dict[str, list[Document]] = defaultdict(list)
    for document in dataset.documents.values():
        by_mailbox[document.mailbox].append(document)
    return by_mailbox


def load_training_groups(
    path: Path, dataset: Dataset, mailboxes: set[str],
) -> list[tuple[Query, list[Document], dict[str, int]]]:
    payload = json.loads(path.read_text())
    if payload["archive_sha256"] != dataset.archive_sha256:
        raise ValueError("Additional judgements belong to a different archive")
    seen = set(dataset.queries)
    groups = []
    for row in payload["queries"]:
        query = Query(row["query_id"], row["synthetic_user"], row["query"],
                      date.fromisoformat(row["query_date"]))
        if query.query_id in seen or query.mailbox not in mailboxes:
            raise ValueError(f"Duplicate or out-of-train query: {query.query_id}")
        seen.add(query.query_id)
        labels = row["labels"]
        documents = [dataset.documents[doc_id] for doc_id in labels]
        target = day_from_query(query)
        if (target is None or set(labels.values()) != {0, 1}
                or len(eligible_indices(documents, query, use_query_day=True)) != len(documents)):
            raise ValueError(f"Invalid labels, date or mailbox: {query.query_id}")
        groups.append((query, documents, {doc_id: label for doc_id, label in labels.items() if label}))
    return groups


def training_pairs(
    dataset: Dataset, mailboxes: set[str], extra_judgements: Path | None = None,
) -> list[TrainingPair]:
    by_mailbox = documents_by_mailbox(dataset)
    groups = []
    for query in sorted(dataset.queries.values(), key=lambda query: query.query_id):
        if query.mailbox not in mailboxes:
            continue
        mailbox_documents = by_mailbox[query.mailbox]
        indices = eligible_indices(mailbox_documents, query, use_query_day=True)
        documents = [mailbox_documents[index] for index in indices]
        relevant = dataset.qrels[query.query_id]
        if not relevant.keys() <= {document.doc_id for document in documents}:
            raise ValueError(f"A training positive fails the date filter: {query.query_id}")
        groups.append((query, documents, relevant))

    if extra_judgements is not None:
        groups.extend(load_training_groups(extra_judgements, dataset, mailboxes))
    pair_count = sum(len(documents) for _, documents, _ in groups)
    if not pair_count:
        raise ValueError("The training split contains no eligible pairs")
    return [
        TrainingPair(
            query.text, searchable_text(document), int(document.doc_id in relevant),
            pair_count / (len(groups) * len(documents)),
        )
        for query, documents, relevant in groups
        for document in documents
    ]


def load_candidates(
    path: Path, dataset: Dataset, manifest: Path, mailboxes: set[str], split: str,
) -> tuple[list[CandidateQuery], dict]:
    run = json.loads(path.read_text())
    config = run["config"]
    if (config["split"] != split or config["archive_sha256"] != dataset.archive_sha256
            or config["split_manifest_sha256"] != hashlib.sha256(manifest.read_bytes()).hexdigest()
            or config.get("e5") is None):
        raise ValueError("E5 candidate metadata does not match the requested archive and split")

    expected = {query.query_id for query in dataset.queries.values() if query.mailbox in mailboxes}
    by_mailbox = documents_by_mailbox(dataset)
    candidates = []
    seen = set()
    for row in run["runs"]["query_day"]["per_query"]:
        if row["method"] != "e5":
            continue
        query_id = row["query_id"]
        if query_id not in expected or query_id in seen:
            raise ValueError(f"Unexpected or duplicate candidate query: {query_id}")
        seen.add(query_id)
        query = dataset.queries[query_id]
        mailbox_documents = by_mailbox[query.mailbox]
        eligible = {mailbox_documents[i].doc_id for i in
                    eligible_indices(mailbox_documents, query, use_query_day=True)}
        ranking = row["ranked_doc_ids"]
        if len(ranking) != min(RRF_DEPTH, len(eligible)) or len(ranking) != len(set(ranking)):
            raise ValueError(f"Invalid E5 ranking length or duplicates: {query_id}")
        if not set(ranking) <= eligible:
            raise ValueError(f"Ineligible E5 candidate: {query_id}")
        candidates.append(CandidateQuery(query, ranking, row["elapsed_ms"]))

    if seen != expected or not candidates:
        raise ValueError("E5 candidates do not cover the complete evaluation split")
    return candidates, config["e5"]
