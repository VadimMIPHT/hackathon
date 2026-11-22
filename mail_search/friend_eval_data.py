import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from zipfile import ZipFile

from mail_search.dataset import Dataset, Document, Query, load_archive, read_split
from mail_search.friend_eval_dates import requested_dates
from mail_search.retrieval import day_from_query, eligible_indices

TOP_K = 50
PARAPHRASE_IDS = frozenset(f"SYN-10PLUS-{i:03d}" for i in range(1, 33))
RANGE_IDS = frozenset(f"SYN-10PLUS-{i:03d}" for i in range(33, 41))


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class FriendData:
    dataset: Dataset
    split_by_mailbox: dict[str, str]
    protocol: dict
    old_qrels: dict
    old_sources: dict[str, list[str]]

    def date_pool(self, query: Query, docs: list[Document], mode: str) -> tuple[dict, list[str]]:
        relevant = self.dataset.qrels[query.query_id]
        as_of = [docs[i] for i in eligible_indices(docs, query, use_query_day=False)]
        if mode == "frozen":
            allowed = [docs[i].doc_id for i in eligible_indices(docs, query, use_query_day=True)]
            day = day_from_query(query)
            dates = frozenset((day,)) if day else frozenset()
        elif mode == "date-aware":
            dates = requested_dates(query) or frozenset()
            allowed = [doc.doc_id for doc in as_of if not dates or doc.received in dates]
        else:
            raise ValueError(f"Unknown date mode: {mode}")
        as_of_relevant = len(relevant.keys() & {doc.doc_id for doc in as_of})
        eligible_relevant = len(relevant.keys() & set(allowed))
        return {
            "query_id": query.query_id,
            "mailbox": query.mailbox,
            "mailbox_split": self.split_by_mailbox[query.mailbox],
            "query_type": "paraphrase" if query.query_id in PARAPHRASE_IDS else "date_range",
            "requested_dates": [day.isoformat() for day in sorted(dates)],
            "relevant_documents": len(relevant),
            "as_of_documents": len(as_of),
            "as_of_relevant": as_of_relevant,
            "eligible_documents": len(allowed),
            "eligible_relevant": eligible_relevant,
            "excluded_by_query_day": as_of_relevant - eligible_relevant,
            "eligible_relevant_at_most_top50": min(eligible_relevant, TOP_K),
        }, allowed


def load_friend_data(original_path: Path, augmented_path: Path, splits_path: Path) -> FriendData:
    original = load_archive(original_path)
    _, split = read_split(splits_path, original, "train")
    with ZipFile(augmented_path) as archive, ZipFile(original_path) as original_zip:
        manifest = json.loads(archive.read("manifest.json"))
        contents = {name: archive.read(name) for name in (
            "README.md", "documents.jsonl", "queries_dev.jsonl", "qrels_dev.jsonl",
        )}
        for name, content in contents.items():
            if hashlib.sha256(content).hexdigest() != manifest["sha256"][name]:
                raise ValueError(f"Augmented checksum mismatch: {name}")
        if contents["documents.jsonl"] != original_zip.read("documents.jsonl"):
            raise ValueError("The two archives must have byte-identical documents")
        old_rows = {row["query_id"]: row for row in
                    map(json.loads, original_zip.read("queries_dev.jsonl").splitlines())}
    query_rows = list(map(json.loads, contents["queries_dev.jsonl"].splitlines()))
    relevance_rows = list(map(json.loads, contents["qrels_dev.jsonl"].splitlines()))
    if (len(original.documents), len(query_rows), len(relevance_rows)) != (
        manifest["documents"], manifest["queries"], manifest["qrels"],
    ):
        raise ValueError("Augmented archive counts do not match its manifest")
    if {row["query_id"]: row for row in query_rows if row["query_id"] in old_rows} != old_rows:
        raise ValueError("An original query was removed or changed")
    queries = {}
    for row in query_rows:
        query_id = row["query_id"]
        if query_id in old_rows:
            continue
        if query_id in queries or not row["query"].strip():
            raise ValueError(f"Duplicate or empty added query: {query_id}")
        queries[query_id] = Query(
            query_id, row["synthetic_user"], row["query"], date.fromisoformat(row["query_date"]),
        )
    if len({row["query_id"] for row in query_rows}) != len(query_rows):
        raise ValueError("Duplicate query IDs")
    if queries.keys() != PARAPHRASE_IDS | RANGE_IDS:
        raise ValueError("Unexpected new-query IDs in the augmented archive")

    qrels: dict[str, dict[str, int]] = {query_id: {} for query_id in queries}
    old = {"preserved": 0, "additional_positive": 0, "additional_negative": 0}
    new_negative = 0
    pairs = set()
    for row in relevance_rows:
        query_id, doc_id, label = row["query_id"], row["doc_id"], row["relevance"]
        pair = (query_id, doc_id)
        query = queries.get(query_id) or original.queries[query_id]
        doc = original.documents[doc_id]
        if pair in pairs or label not in (0, 1):
            raise ValueError(f"Duplicate qrel or invalid label: {pair}")
        if doc.mailbox != query.mailbox or (doc.received and doc.received > query.as_of):
            raise ValueError(f"Qrel is outside its mailbox or after its query date: {pair}")
        pairs.add(pair)
        if query_id in queries:
            if label:
                qrels[query_id][doc_id] = label
            else:
                new_negative += 1
        elif doc_id in original.qrels[query_id]:
            if label != original.qrels[query_id][doc_id]:
                raise ValueError(f"Original qrel label was changed: {pair}")
            old["preserved"] += 1
        else:
            old["additional_positive" if label else "additional_negative"] += 1
    if old["preserved"] != sum(map(len, original.qrels.values())):
        raise ValueError("An original qrel is missing")
    if new_negative:
        raise ValueError("Unexpected explicit negative in this added-query cohort")
    if not all(qrels.values()):
        raise ValueError("Every new query must have at least one positive qrel")

    old_sources = {}
    for query_id, query in queries.items():
        rel = qrels[query_id].keys()
        same_box = {qid: old_rel.keys() for qid, old_rel in original.qrels.items()
                    if original.queries[qid].mailbox == query.mailbox}
        if query_id in PARAPHRASE_IDS:
            sources = [qid for qid, old_rel in same_box.items() if rel == old_rel]
        else:
            sources = [qid for qid, old_rel in same_box.items() if rel & old_rel]
        if not sources or not rel <= set().union(*(same_box[qid] for qid in sources)):
            raise ValueError(f"The original labels do not cover {query_id}")
        old_sources[query_id] = sorted(sources)

    dataset = Dataset(sha256(augmented_path), original.documents, queries, qrels)
    protocol = {
        "original_archive_sha256": original.archive_sha256,
        "augmented_archive_sha256": dataset.archive_sha256,
        "split_manifest_sha256": sha256(splits_path),
        "split_seed": split["seed"],
        "cohort": "added SYN-10PLUS queries only",
        "date_modes": ["frozen", "date-aware"],
        "top_k": TOP_K,
        "tie_break": "doc_id",
    }
    split_by_mailbox = {
        mailbox: partition for partition, info in split["splits"].items()
        for mailbox in info["mailboxes"]
    }
    return FriendData(dataset, split_by_mailbox, protocol, old, old_sources)


def audit_manual(path: Path, data: FriendData) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    fields = sorted(set().union(*(row.keys() for row in rows)))
    matching = [row for row in rows if row["doc_id"] in data.dataset.documents]
    known_positive = set().union(*(row.keys() for row in data.dataset.qrels.values()))
    differences = {"text": 0, "mailbox": 0, "date": 0}
    for row in matching:
        doc = data.dataset.documents[row["doc_id"]]
        differences["text"] += row["doc"] != doc.text
        differences["mailbox"] += row["synthetic_user"] != doc.mailbox
        differences["date"] += (date.fromisoformat(row["doc_dt"]) if row["doc_dt"] else None) != doc.received
    positive_labels = Counter(
        label for row in rows for label, value in row["labels"].items() if value == 1
    )
    return {
        "sha256": sha256(path),
        "primary_label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "positive_label_counts": dict(sorted(positive_labels.items())),
        "rows": len(rows),
        "fields": fields,
        "unique_doc_ids": len({row["doc_id"] for row in rows}),
        "doc_ids_matched": len(matching),
        "metadata_mismatch": differences,
        "doc_ids_appearing_in_new_positives": sum(row["doc_id"] in known_positive for row in rows),
        "query_fields": [field for field in fields if "query" in field.casefold()],
        "suitable_for_retrieval": False,
    }
