import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import statistics
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from mail_search.dataset import load_archive, read_split
from mail_search.retrieval import (
    BM25, BM25_B, BM25_K1, RRF_DEPTH, RRF_K, SUBJECT_WEIGHT,
    day_from_query, eligible_indices, fuse, searchable_text,
)

METRICS = ("ndcg@10", "recall@10", "recall@50", "mrr@10_singleton")


def measure(ranking: list[str], relevant: dict[str, int]) -> dict[str, float | None]:
    top10 = ranking[:10]
    positions = [rank for rank, doc_id in enumerate(top10, 1) if doc_id in relevant]
    dcg = sum(1 / math.log2(rank + 1) for rank in positions)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(10, len(relevant)) + 1))
    return {
        "ndcg@10": dcg / ideal,
        "recall@10": len(positions) / len(relevant),
        "recall@50": len(set(ranking[:50]) & relevant.keys()) / len(relevant),
        "mrr@10_singleton": (1 / positions[0] if positions else 0.0) if len(relevant) == 1 else None,
    }


def summarize(rows: list[dict]) -> dict:
    result = {"queries": len(rows), "mailboxes": len({row["mailbox"] for row in rows})}
    for key in METRICS:
        values = [value for row in rows if (value := row["metrics"][key]) is not None]
        result[key] = statistics.mean(values) if values else None
    elapsed = sorted(row["elapsed_ms"] for row in rows)
    result["p50_ms_offline"] = statistics.median(elapsed)
    result["p99_ms_offline"] = elapsed[math.ceil(0.99 * len(elapsed)) - 1]
    return result


def source_hashes() -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))}


def benchmark(args: argparse.Namespace) -> dict:
    dataset = load_archive(args.archive)
    mailboxes, split_manifest = read_split(args.splits, dataset, args.split)
    queries = sorted((query for query in dataset.queries.values() if query.mailbox in mailboxes),
                     key=lambda query: (query.mailbox, query.query_id))
    active_mailboxes = {query.mailbox for query in queries}
    documents = sorted((doc for doc in dataset.documents.values() if doc.mailbox in active_mailboxes),
                       key=lambda doc: (doc.mailbox, doc.doc_id))
    doc_ids = [doc.doc_id for doc in documents]
    bm25 = BM25(documents)
    by_mailbox = {mailbox: list(group) for mailbox, group in
                  itertools.groupby(documents, key=lambda doc: doc.mailbox)}
    offsets = {}
    offset = 0
    for mailbox, group in by_mailbox.items():
        offsets[mailbox] = offset
        offset += len(group)

    runner = {
        "utc_started": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "code_base_commit": args.base_commit,
        "source_sha256": source_hashes(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    config = {
        "split": args.split,
        "archive_sha256": dataset.archive_sha256,
        "split_seed": split_manifest["seed"],
        "split_manifest_sha256": hashlib.sha256(args.splits.read_bytes()).hexdigest(),
        "indexed_documents": len(documents),
        "bm25": {"k1": BM25_K1, "b": BM25_B, "subject_weight": SUBJECT_WEIGHT},
        "rrf": {"k": RRF_K, "depth_per_leg": RRF_DEPTH},
        "e5": None,
    }
    e5 = None
    if args.e5_model:
        import numpy as np
        import torch

        from mail_search.e5 import DOCUMENT_BATCH_SIZE, MAX_LENGTH, E5, rank

        torch.set_num_threads(2)
        e5 = E5(args.e5_model)
        runner.update({"torch": torch.__version__, "cuda": torch.version.cuda,
                       "gpu": torch.cuda.get_device_name(0), "numpy": np.__version__,
                       "transformers": version("transformers"), "tokenizers": version("tokenizers")})
        config["e5"] = {
            "repo": "intfloat/multilingual-e5-small", "revision": args.e5_revision,
            "max_length": MAX_LENGTH, "document_batch_size": DOCUMENT_BATCH_SIZE,
            "model_dtype": "fp16", "pooling_dtype": "fp32",
        }
        started = perf_counter()
        matrix = e5.encode([searchable_text(doc) for doc in documents], "passage: ")
        runner["document_encoding_seconds"] = perf_counter() - started
        e5.encode(["прогрев поиска", "прогрев поиска по письмам"], "query: ")

    runs = {}
    for mode in ("as_of", "query_day"):
        rows = []
        candidate_counts = []
        for query in queries:
            relevant = dataset.qrels[query.query_id]
            start = perf_counter()
            indices = [offsets[query.mailbox] + index for index in
                       eligible_indices(by_mailbox[query.mailbox], query, mode == "query_day")]
            filter_ms = (perf_counter() - start) * 1000
            start = perf_counter()
            lexical = bm25.rank(query.text, indices)
            lexical_ms = (perf_counter() - start) * 1000
            rankings = {"bm25": (lexical, filter_ms + lexical_ms)}
            if e5 is not None:
                start = perf_counter()
                vector = e5.encode([query.text], "query: ")[0]
                dense = rank(matrix, vector, indices, doc_ids, RRF_DEPTH)
                dense_ms = (perf_counter() - start) * 1000
                start = perf_counter()
                hybrid = fuse(lexical, dense)
                fusion_ms = (perf_counter() - start) * 1000
                rankings.update({"e5": (dense, filter_ms + dense_ms),
                                 "rrf": (hybrid, filter_ms + lexical_ms + dense_ms + fusion_ms)})

            allowed_positives = len({doc_ids[i] for i in indices} & relevant.keys())
            candidate_counts.append({
                "query_id": query.query_id,
                "mailbox": query.mailbox,
                "eligible": len(indices),
                "relevant": len(relevant),
                "eligible_relevant": allowed_positives,
                "parsed_day": day_from_query(query) is not None,
            })
            for method, (ranking, elapsed_ms) in rankings.items():
                rows.append({
                    "query_id": query.query_id,
                    "mailbox": query.mailbox,
                    "method": method,
                    "ranked_doc_ids": ranking,
                    "metrics": measure(ranking, relevant),
                    "elapsed_ms": elapsed_ms,
                })

        methods = sorted({row["method"] for row in rows})
        runs[mode] = {
            "summary": {method: summarize([row for row in rows if row["method"] == method])
                        for method in methods},
            "by_mailbox": {
                mailbox: {
                    method: summarize([row for row in rows
                                       if row["method"] == method and row["mailbox"] == mailbox])
                    for method in methods
                }
                for mailbox in sorted(active_mailboxes)
            },
            "candidate_counts": candidate_counts,
            "per_query": rows,
        }
    return {"environment": runner, "config": config, "runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare VK mail-search retrieval baselines.")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--e5-model", type=Path)
    parser.add_argument("--e5-revision")
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.e5_model and not args.e5_revision:
        parser.error("--e5-revision is required with --e5-model")
    result = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
