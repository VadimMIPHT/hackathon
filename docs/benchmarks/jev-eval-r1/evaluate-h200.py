import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

from mail_search.benchmark import measure
from mail_search.dataset import Dataset, load_archive
from mail_search.depth_benchmark import write_json
from mail_search.friend_eval import group_results, retrieve, score
from mail_search.friend_eval_data import FriendData, load_friend_data, sha256
from mail_search.reranker_data import documents_by_mailbox


@dataclass(frozen=True)
class JevData(FriendData):
    query_ids: dict[str, str] | None = None

    def date_pool(self, query, docs, mode: str) -> tuple[dict, list[str]]:
        row, allowed = super().date_pool(query, docs, mode)
        if not query.query_id.startswith("SYN-10PLUS-"):
            row["query_type"] = "original"
        return row, allowed


def read_jev(args: argparse.Namespace) -> tuple[FriendData, JevData, JevData, dict]:
    original = load_friend_data(args.original_archive, args.augmented_archive, args.splits)
    with ZipFile(args.jev_archive) as new, ZipFile(args.original_archive) as initial:
        prefix, = {Path(name).parts[0] for name in new.namelist()}
        manifest = json.loads(new.read(f"{prefix}/manifest.json"))
        for name in ("README.md", "documents.jsonl", "queries_dev.jsonl", "qrels_dev.jsonl"):
            content = new.read(f"{prefix}/{name}")
            if hashlib.sha256(content).hexdigest() != manifest["sha256"][name]:
                raise ValueError(f"Jev checksum failed: {name}")
        if new.read(f"{prefix}/documents.jsonl") != initial.read("documents.jsonl"):
            raise ValueError("Jev documents are different from the original corpus")
        read_rows = lambda archive, name: [json.loads(line) for line in archive.open(name)]
        old_queries = {q["query_id"]: q for q in read_rows(initial, "queries_dev.jsonl")}
        new_query_rows = read_rows(new, f"{prefix}/queries_dev.jsonl")
        new_queries = {q["query_id"]: q for q in new_query_rows}
        if len(new_queries) != len(new_query_rows) or len(new_queries) != manifest["queries"]:
            raise ValueError("Unexpected Jev queries")
        if len(original.dataset.documents) != manifest["documents"]:
            raise ValueError("Unexpected Jev document count")
        if {qid: new_queries[qid] for qid in old_queries} != old_queries:
            raise ValueError("Original query text or metadata changed")
        mapping = {qid: f"N-{qid.rsplit('-', 1)[1]}" for qid in original.dataset.queries}
        if set(new_queries) != old_queries.keys() | set(mapping.values()):
            raise ValueError("Unexpected cohort")
        for old_id, new_id in mapping.items():
            q = original.dataset.queries[old_id]
            if new_queries[new_id] != {
                "query_id": new_id, "query": q.text, "synthetic_user": q.mailbox,
                "query_date": q.as_of.isoformat(),
            }:
                raise ValueError(f"Cannot reuse scores for modified query {new_id}")
        positive_rows = read_rows(new, f"{prefix}/qrels_dev.jsonl")
        original_positive = read_rows(initial, "qrels_dev.jsonl")
    if len(positive_rows) != manifest["qrels"]:
        raise ValueError("Unexpected Jev positive count")
    positives: dict[str, dict[str, int]] = {qid: {} for qid in new_queries}
    for row in positive_rows:
        qid, did = row["query_id"], row["doc_id"]
        if row["relevance"] != 1 or did not in original.dataset.documents or did in positives[qid]:
            raise ValueError(f"Invalid Jev positive {qid}, {did}")
        positives[qid][did] = 1
    if not all(positives.values()):
        raise ValueError("Found a query without relevance labels")

    archive_sha = sha256(args.jev_archive)
    common = {
        "dataset": original.dataset,
        "split_by_mailbox": original.split_by_mailbox,
        "old_qrels": original.old_qrels,
        "old_sources": original.old_sources,
    }
    new = JevData(**{**common, "dataset": Dataset(
        archive_sha, original.dataset.documents, original.dataset.queries,
        {old_id: positives[new_id] for old_id, new_id in mapping.items()},
    )}, protocol={**original.protocol, "jev_archive_sha256": archive_sha}, query_ids=mapping)
    old_dataset = load_archive(args.original_archive)
    old = JevData(**{**common, "dataset": Dataset(
        archive_sha, original.dataset.documents, old_dataset.queries,
        {qid: positives[qid] for qid in old_queries},
    )}, protocol={**original.protocol, "jev_archive_sha256": archive_sha, "cohort": "99 original queries"})
    meta = {"prefix": prefix, "manifest": manifest, "positives": positives,
            "original_positive": original_positive}
    return original, new, old, meta


def audit_full(path: Path, meta: dict, documents: dict, new: JevData) -> dict:
    positives = meta["positives"]
    query_index = {qid: n for n, qid in enumerate(positives)}
    document_index = {did: n for n, did in enumerate(documents)}
    n_docs = len(documents)
    total = n_docs * len(positives)
    seen = bytearray(total)
    counts: Counter[int] = Counter()
    digest = hashlib.sha256()
    with ZipFile(path) as archive, archive.open(f"{meta['prefix']}/qrels_full.jsonl") as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            qid, did, label = row["query_id"], row["doc_id"], row["relevance"]
            index = query_index[qid] * n_docs + document_index[did]
            if seen[index] or type(label) is not int or label != positives[qid].get(did, 0):
                raise ValueError(f"Conflicting, invalid or duplicated annotation {qid}, {did}")
            seen[index] = 1
            counts[label] += 1
    assert sum(counts.values()) == total == meta["manifest"]["qrels_full"]
    assert digest.hexdigest() == meta["manifest"]["sha256"]["qrels_full.jsonl"]
    return {"full_pairs": total, "label_counts": dict(counts), "missing_pairs": 0,
            "full_sha256": digest.hexdigest(), "archive_sha256": new.dataset.archive_sha256}


def differences(before: dict, after: dict) -> dict:
    added = [(q, d) for q, old in before.items() for d in sorted(after[q].keys() - old.keys())]
    removed = [(q, d) for q, old in before.items() for d in sorted(old.keys() - after[q].keys())]
    return {"previous_positives": sum(map(len, before.values())),
            "jev_positives": sum(map(len, after.values())),
            "added_positive": added, "removed_positive": removed}


def replay(cache: Path, original: FriendData, new: JevData, old: JevData, meta: dict) -> dict:
    docs = documents_by_mailbox(new.dataset)
    files = {"E5": "candidates.json", "BGE": "bge.json", "Qwen": "qwen.json", "Qwen FT": "qwen-ft.json"}
    original_positive: dict = {qid: {} for qid in old.dataset.queries}
    for r in meta["original_positive"]:
        original_positive[r["query_id"]][r["doc_id"]] = 1
    data = {
        "protocol": new.protocol, "query_id_mapping": new.query_ids,
        "annotation_audit": audit_full(Path(new.protocol["archive_path"]), meta, new.dataset.documents, new),
        "changed_qrels": {"new40": differences(original.dataset.qrels, new.dataset.qrels),
                          "original99": differences(original_positive, old.dataset.qrels)},
        "sources": {name: sha256(cache / filename) for name, filename in files.items()}, "models": {},
    }
    for name, filename in files.items():
        artifact = json.loads((cache / filename).read_text())
        if artifact["protocol"] != original.protocol:
            raise ValueError(f"Artifact protocol mismatch: {filename}")
        if name != "E5" and artifact["candidates_sha256"] != data["sources"]["E5"]:
            raise ValueError(f"Candidate SHA-256 mismatch: {filename}")
        data["models"][name] = {"environment": artifact["environment"],
                                "config": artifact.get("model", artifact.get("e5")), "modes": {}}
        for mode, section in artifact["modes"].items():
            rows = []
            for row in section["per_query"]:
                qid = row["query_id"]
                query = new.dataset.queries[qid]
                updated, allowed = new.date_pool(query, docs[query.mailbox], mode)
                relevant = new.dataset.qrels[qid]
                ranking = row.get("reranked_doc_ids", row["ranked_doc_ids"])
                assert len(set(ranking)) == len(ranking) and set(ranking) <= set(allowed)
                old_metrics = measure(ranking, original.dataset.qrels[qid])
                assert old_metrics == row.get("reranking_metrics", row["retrieval_metrics"])
                rows.append({**updated, "query_id": new.query_ids[qid],
                             "prior_query_id": qid, "ranked_doc_ids": ranking,
                             "retrieved_relevant": len(set(ranking) & relevant.keys()),
                             "previous_metrics": old_metrics,
                             "reranking_metrics": measure(ranking, relevant),
                             "prior_zero_now_positive_at10": sum(
                                 d in relevant and d not in original.dataset.qrels[qid] for d in ranking[:10]),
                             "prior_positive_now_zero_at10": sum(
                                 d not in relevant and d in original.dataset.qrels[qid] for d in ranking[:10]),
                             })
            assert len(rows) == len({r["query_id"] for r in rows}) == len(new.dataset.queries)
            data["models"][name]["modes"][mode] = {**group_results(rows), "per_query": rows}
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("replay", "retrieve", "score"))
    for key in ("original-archive", "augmented-archive", "jev-archive", "splits", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    for key in ("cache", "e5-model", "candidates", "model-path", "weights"):
        parser.add_argument(f"--{key}", type=Path)
    parser.add_argument("--gpu-memory-limit-gib", type=float, default=5)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--e5-revision")
    parser.add_argument("--model", choices=("bge", "qwen", "qwen-ft"))
    parser.add_argument("--model-revision")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--train-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--batch-pause-ms", type=float, default=0)
    args = parser.parse_args()
    previous, new, old, meta = read_jev(args)
    if args.action == "replay":
        new.protocol["archive_path"] = str(args.jev_archive)
        payload = replay(args.cache, previous, new, old, meta)
    elif args.action == "retrieve":
        payload = retrieve(args, old)
    else:
        payload = score(args, old)
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
