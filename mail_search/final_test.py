import argparse
import hashlib
import json
import os
import platform
import statistics
from importlib.metadata import version
from pathlib import Path
from time import perf_counter, sleep

from mail_search.benchmark import METRICS, measure, source_hashes
from mail_search.dataset import Dataset, load_archive, read_split
from mail_search.reranker_data import documents_by_mailbox
from mail_search.retrieval import day_from_query, eligible_indices, searchable_text


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_protocol(args: argparse.Namespace, dataset: Dataset, manifest: dict) -> dict:
    protocol = json.loads(args.protocol.read_text())
    expected = {
        "split": "test", "split_seed": manifest["seed"], "archive_sha256": dataset.archive_sha256,
        "split_manifest_sha256": sha256(args.splits), "bge_checkpoint_sha256": sha256(args.bge_weights),
        "e5_weights_sha256": sha256(args.e5_model / "model.safetensors"),
        "bge_base_weights_sha256": sha256(args.bge_model / "model.safetensors"),
        "date_filter": "query_day", "tie_break": "doc_id", "top_k": 50,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Test input or parameter differs from the locked protocol: {key}")
    return protocol


def retrieve(args: argparse.Namespace, dataset: Dataset, users: set[str], protocol: dict) -> tuple:
    import torch

    from mail_search.e5 import DOCUMENT_BATCH_SIZE, MAX_LENGTH, E5, rank

    if (protocol["e5"]["max_length"], protocol["e5"]["document_batch_size"]) != (
        MAX_LENGTH, DOCUMENT_BATCH_SIZE,
    ):
        raise ValueError("E5 model length or batch size differs from val")
    boxes = documents_by_mailbox(dataset)
    queries = sorted((query for query in dataset.queries.values() if query.mailbox in users),
                     key=lambda query: (query.mailbox, query.query_id))
    active = {query.mailbox for query in queries}
    docs = [document for mailbox in sorted(active)
            for document in sorted(boxes[mailbox], key=lambda doc: doc.doc_id)]
    doc_ids = [document.doc_id for document in docs]
    indices = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    e5 = E5(args.e5_model)
    started = perf_counter()
    matrix = e5.encode([searchable_text(doc) for doc in docs], "passage: ")
    indexing_seconds = perf_counter() - started
    e5.encode(["прогрев поиска"], "query: ")
    rows = []
    for query in queries:
        started = perf_counter()
        mailbox_docs = boxes[query.mailbox]
        allowed = [mailbox_docs[i].doc_id for i in eligible_indices(
            mailbox_docs, query, use_query_day=True,
        )]
        vector = e5.encode([query.text], "query: ")[0]
        ranking = rank(matrix, vector, [indices[d] for d in allowed], doc_ids, protocol["top_k"])
        rows.append((query, allowed, ranking, (perf_counter() - started) * 1000))
    del e5
    torch.cuda.empty_cache()
    return rows, {"documents": len(docs), "seconds": indexing_seconds}


def rerank(args: argparse.Namespace, dataset: Dataset, protocol: dict, candidates: list) -> list[dict]:
    from mail_search.reranker import Reranker

    bge = Reranker(
        args.bge_model, protocol["bge_revision"], protocol["bge_max_length"],
        protocol["bge_train_layers"],
    )
    bge.load_weights(args.bge_weights)
    bge.model.eval()
    batch_size = protocol["bge_batch_size"]
    bge.score("прогрев поиска", ["прогрев оценки релевантности"] * batch_size)
    rows = []
    for query, eligible, e5, elapsed_e5 in candidates:
        started = perf_counter()
        scoring_ids = sorted(e5)
        scores = []
        for n in range(0, len(scoring_ids), batch_size):
            batch = [searchable_text(dataset.documents[d]) for d in scoring_ids[n:n + batch_size]]
            scores.extend(bge.score(query.text, batch))
            if args.batch_pause_ms:
                sleep(args.batch_pause_ms / 1000)
        logits = dict(zip(scoring_ids, scores, strict=True))
        reranked = sorted(e5, key=lambda d: (-logits[d], d))
        elapsed_bge = (perf_counter() - started) * 1000

        relevant = dataset.qrels[query.query_id]
        ideal = sorted(e5, key=lambda d: (-relevant.get(d, 0), d))
        rows.append({
            "query_id": query.query_id, "mailbox": query.mailbox,
            "requested_day": (day.isoformat() if (day := day_from_query(query)) is not None else None),
            "eligible_documents": len(eligible),
            "relevant_documents": len(relevant),
            "eligible_relevant": len(relevant.keys() & set(eligible)),
            "retrieved_relevant": len(relevant.keys() & set(e5)),
            "e5_ranked_doc_ids": e5, "bge_ranked_doc_ids": reranked,
            "bge_logits": logits, "e5_ms": elapsed_e5, "bge_ms": elapsed_bge,
            "e5_metrics": measure(e5, relevant),
            "bge_metrics": measure(reranked, relevant),
            "oracle_metrics": measure(ideal, relevant),
        })
    return rows


def aggregate(rows: list[dict]) -> dict:
    summary = {
        "queries": len(rows), "mailboxes": len({r["mailbox"] for r in rows}),
        "singleton_queries": sum(r["relevant_documents"] == 1 for r in rows),
        "known_relevant": sum(r["relevant_documents"] for r in rows),
        "eligible_relevant": sum(r["eligible_relevant"] for r in rows),
        "retrieved_relevant": sum(r["retrieved_relevant"] for r in rows),
    }
    for name in ("e5", "bge", "oracle"):
        summary[name] = {
            metric: statistics.mean(values) if (
                values := [r[f"{name}_metrics"][metric] for r in rows
                           if r[f"{name}_metrics"][metric] is not None]
            ) else None
            for metric in METRICS
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the locked E5 top-50 + BGE cascade on test.")
    for name in ("archive", "splits", "protocol", "e5-model", "bge-model", "bge-weights", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--gpu-memory-limit-gib", type=float, default=6)
    parser.add_argument("--batch-pause-ms", type=float, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("The test result already exists")
    if args.batch_pause_ms < 0 or args.gpu_memory_limit_gib <= 0:
        parser.error("Invalid GPU limit or batch pause")

    import torch

    from mail_search.reranker import write_metrics

    dataset = load_archive(args.archive)
    users, manifest = read_split(args.splits, dataset, "test")
    protocol = load_protocol(args, dataset, manifest)
    torch.set_num_threads(2)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    memory_limit = args.gpu_memory_limit_gib * (1 << 30)
    torch.cuda.set_per_process_memory_fraction(min(memory_limit / total_memory, 1.0), device=0)
    candidates, indexing = retrieve(args, dataset, users, protocol)
    rows = rerank(args, dataset, protocol, candidates)
    if len(rows) != manifest["splits"]["test"]["queries"]:
        raise ValueError("Test query count differs from the locked split")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_metrics(args.output, {
        "protocol": protocol, "protocol_sha256": sha256(args.protocol),
        "environment": {
            "host": platform.node(), "python": platform.python_version(),
            "torch": torch.__version__, "transformers": version("transformers"),
            "tokenizers": version("tokenizers"), "numpy": version("numpy"), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "gpu_memory_limit_gib": args.gpu_memory_limit_gib, "batch_pause_ms": args.batch_pause_ms,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(0) / (1 << 30),
            "code_base_commit": args.base_commit, "source_sha256": source_hashes(),
        },
        "indexing": indexing,
        "summary": aggregate(rows),
        "by_mailbox": {
            mailbox: aggregate([row for row in rows if row["mailbox"] == mailbox])
            for mailbox in sorted({r["mailbox"] for r in rows})
        },
        "per_query": rows,
    })


if __name__ == "__main__":
    main()
