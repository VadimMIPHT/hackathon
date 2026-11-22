import argparse
import hashlib
import itertools
import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from zipfile import ZipFile


@dataclass(frozen=True)
class Document:
    doc_id: str
    mailbox: str
    text: str
    received: date | None


@dataclass(frozen=True)
class Query:
    query_id: str
    mailbox: str
    text: str
    as_of: date


@dataclass(frozen=True)
class Dataset:
    archive_sha256: str
    documents: dict[str, Document]
    queries: dict[str, Query]
    qrels: dict[str, dict[str, int]]


def load_archive(path: Path) -> Dataset:
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        contents = {}
        for name in ("README.md", "documents.jsonl", "queries_dev.jsonl", "qrels_dev.jsonl"):
            content = archive.read(name)
            if hashlib.sha256(content).hexdigest() != manifest["sha256"][name]:
                raise ValueError(f"Checksum mismatch: {name}")
            contents[name] = content

    documents = {}
    for line in contents["documents.jsonl"].splitlines():
        row = json.loads(line)
        document = Document(
            row["doc_id"], row["synthetic_user"], row["doc"],
            date.fromisoformat(row["doc_dt"]) if row["doc_dt"] else None,
        )
        if document.doc_id in documents:
            raise ValueError(f"Duplicate document: {document.doc_id}")
        documents[document.doc_id] = document

    queries = {}
    for line in contents["queries_dev.jsonl"].splitlines():
        row = json.loads(line)
        query = Query(
            row["query_id"], row["synthetic_user"], row["query"],
            date.fromisoformat(row["query_date"]),
        )
        if not query.text.strip() or query.query_id in queries:
            raise ValueError(f"Empty or duplicate query: {query.query_id}")
        queries[query.query_id] = query

    qrels: dict[str, dict[str, int]] = {}
    for line in contents["qrels_dev.jsonl"].splitlines():
        row = json.loads(line)
        query = queries[row["query_id"]]
        document = documents[row["doc_id"]]
        if query.mailbox != document.mailbox or (document.received and document.received > query.as_of):
            raise ValueError(f"Ineligible qrel: {query.query_id}, {document.doc_id}")
        relevance = qrels.setdefault(query.query_id, {})
        if document.doc_id in relevance or row["relevance"] != 1:
            raise ValueError(f"Duplicate or nonbinary qrel: {query.query_id}, {document.doc_id}")
        relevance[document.doc_id] = row["relevance"]

    if set(qrels) != set(queries):
        raise ValueError("Every public query needs at least one positive qrel")
    counts = (len(documents), len(queries), sum(len(rel) for rel in qrels.values()))
    if counts != tuple(manifest[key] for key in ("documents", "queries", "qrels")):
        raise ValueError(f"Archive row counts do not match manifest: {counts}")
    return Dataset(hashlib.sha256(path.read_bytes()).hexdigest(), documents, queries, qrels)


def manifest_for(dataset: Dataset, seed: int) -> dict:
    doc_counts = Counter(doc.mailbox for doc in dataset.documents.values())
    query_counts = Counter(query.mailbox for query in dataset.queries.values())
    labelled = sorted(query_counts)
    unlabelled = sorted(doc_counts.keys() - query_counts.keys())
    if len(labelled) != 7 or len(unlabelled) != 5:
        raise ValueError("This split is defined for the published 7 labelled and 5 unlabelled mailboxes")

    rng = random.Random(seed)
    rng.shuffle(labelled)
    rng.shuffle(unlabelled)
    options = []
    for train in itertools.combinations(labelled, 3):
        remaining = [mailbox for mailbox in labelled if mailbox not in train]
        for val in itertools.combinations(remaining, 2):
            test = tuple(mailbox for mailbox in remaining if mailbox not in val)
            counts = [sum(query_counts[mailbox] for mailbox in group) for group in (train, val, test)]
            query_imbalance = sum(abs(count * 100 - percent * len(dataset.queries))
                                  for count, percent in zip(counts, (55, 22, 23)))
            val_imbalance = abs(query_counts[val[0]] - query_counts[val[1]])
            options.append(((query_imbalance, val_imbalance), (train, val, test)))

    _, groups = min(options, key=lambda option: option[0])
    splits = {}
    for name, group, extra in zip(
        ("train", "val", "test"), groups, (unlabelled[:3], unlabelled[3:4], unlabelled[4:])
    ):
        mailboxes = sorted([*group, *extra])
        qids = [query.query_id for query in dataset.queries.values() if query.mailbox in mailboxes]
        splits[name] = {
            "mailboxes": mailboxes,
            "labelled_mailboxes": len(group),
            "documents": sum(doc_counts[mailbox] for mailbox in mailboxes),
            "queries": len(qids),
            "qrels": sum(len(dataset.qrels[qid]) for qid in qids),
            "singleton_queries": sum(len(dataset.qrels[qid]) == 1 for qid in qids),
        }
    return {"archive_sha256": dataset.archive_sha256, "seed": seed, "splits": splits}


def read_split(path: Path, dataset: Dataset, split: str) -> tuple[set[str], dict]:
    manifest = json.loads(path.read_text())
    if manifest != manifest_for(dataset, manifest["seed"]):
        raise ValueError("Split manifest does not match archive and seed")
    mailboxes = [mailbox for info in manifest["splits"].values() for mailbox in info["mailboxes"]]
    if len(set(mailboxes)) != len(mailboxes) or set(mailboxes) != {
        doc.mailbox for doc in dataset.documents.values()
    }:
        raise ValueError("Split must assign each mailbox exactly once")
    return set(manifest["splits"][split]["mailboxes"]), manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the VK archive and split whole mailboxes.")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = manifest_for(load_archive(args.archive), args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
