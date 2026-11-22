import argparse
import json
import math
import os
import platform
import statistics
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter, sleep

from mail_search.benchmark import METRICS, measure, source_hashes
from mail_search.depth_benchmark import configure_cuda, write_json
from mail_search.friend_eval_data import TOP_K, FriendData, audit_manual, load_friend_data, sha256
from mail_search.reranker_data import documents_by_mailbox
from mail_search.retrieval import searchable_text

MODES = ("frozen", "date-aware")


def aggregate(rows: list[dict]) -> dict:
    result = {
        "queries": len(rows),
        "mailboxes": len({row["mailbox"] for row in rows}),
        "parsed_date_queries": sum(bool(row["requested_dates"]) for row in rows),
        "date_excluded_queries": sum(row["excluded_by_query_day"] > 0 for row in rows),
    }
    for key in (
        "relevant_documents", "as_of_documents", "as_of_relevant", "eligible_documents",
        "eligible_relevant", "excluded_by_query_day", "eligible_relevant_at_most_top50",
        "retrieved_relevant", "excluded_by_top50_capacity", "excluded_by_e5_ordering",
    ):
        if key in rows[0]:
            result[key] = sum(row[key] for row in rows)
    for key in ("retrieval_metrics", "candidate_oracle_metrics", "reranking_metrics"):
        if key in rows[0]:
            result[key] = {
                metric: statistics.mean(values) if (
                    values := [r[key][metric] for r in rows if r[key][metric] is not None]
                ) else None
                for metric in METRICS
            }
    return result


def group_results(rows: list[dict]) -> dict:
    report = {"summary": aggregate(rows)}
    for key in ("query_type", "mailbox_split", "mailbox"):
        report[f"by_{key}"] = {
            group: aggregate([row for row in rows if row[key] == group])
            for group in sorted({row[key] for row in rows})
        }
    return report


def environment(args: argparse.Namespace) -> dict:
    return {
        "code_base_commit": args.base_commit,
        "source_sha256": source_hashes(),
        "utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
    }


def gpu_environment(args: argparse.Namespace) -> dict:
    import torch

    return {
        **environment(args),
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "gpu_memory_limit_gib": args.gpu_memory_limit_gib,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(0) / (1 << 30),
    }


def audit(args: argparse.Namespace, data: FriendData) -> dict:
    boxes = documents_by_mailbox(data.dataset)
    result = {
        "protocol": data.protocol,
        "environment": environment(args),
        "original_qrels": data.old_qrels,
        "modes": {},
    }
    for mode in MODES:
        rows = []
        for query in sorted(data.dataset.queries.values(), key=lambda q: q.query_id):
            row, _ = data.date_pool(query, boxes[query.mailbox], mode)
            rows.append({**row, "related_original_queries": data.old_sources[query.query_id]})
        result["modes"][mode] = {**group_results(rows), "per_query": rows}
    if args.manual_documents:
        result["manual_letter_classification"] = audit_manual(args.manual_documents, data)
    return result


def retrieve(args: argparse.Namespace, data: FriendData) -> dict:
    from mail_search.e5 import DOCUMENT_BATCH_SIZE, MAX_LENGTH, E5, rank

    configure_cuda(args.gpu_memory_limit_gib)
    docs = sorted(
        (doc for doc in data.dataset.documents.values() if doc.mailbox in {
            query.mailbox for query in data.dataset.queries.values()
        }),
        key=lambda doc: (doc.mailbox, doc.doc_id),
    )
    boxes = documents_by_mailbox(data.dataset)
    doc_ids = [doc.doc_id for doc in docs]
    indices = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    e5 = E5(args.e5_model)
    started = perf_counter()
    matrix = e5.encode([searchable_text(doc) for doc in docs], "passage: ")
    indexing_seconds = perf_counter() - started
    e5.encode(["прогрев поиска"], "query: ")
    results: dict[str, list[dict]] = {mode: [] for mode in MODES}
    for query in sorted(data.dataset.queries.values(), key=lambda q: (q.mailbox, q.query_id)):
        started = perf_counter()
        vector = e5.encode([query.text], "query: ")[0]
        embed_ms = (perf_counter() - started) * 1000
        relevant = data.dataset.qrels[query.query_id]
        for mode in MODES:
            started = perf_counter()
            row, allowed = data.date_pool(query, boxes[query.mailbox], mode)
            ranking = rank(matrix, vector, [indices[doc_id] for doc_id in allowed], doc_ids, TOP_K)
            elapsed = (perf_counter() - started) * 1000 + embed_ms
            retrieved = len(relevant.keys() & set(ranking))
            oracle = sorted(ranking, key=lambda d: (d not in relevant, d))
            results[mode].append({
                **row,
                "retrieved_relevant": retrieved,
                "excluded_by_top50_capacity": row["eligible_relevant"] - row["eligible_relevant_at_most_top50"],
                "excluded_by_e5_ordering": row["eligible_relevant_at_most_top50"] - retrieved,
                "ranked_doc_ids": ranking,
                "retrieval_metrics": measure(ranking, relevant),
                "candidate_oracle_metrics": measure(oracle, relevant),
                "retrieval_ms_offline": elapsed,
            })
    return {
        "protocol": data.protocol,
        "environment": gpu_environment(args),
        "e5": {
            "repo": "intfloat/multilingual-e5-small",
            "revision": args.e5_revision,
            "weights_sha256": sha256(args.e5_model / "model.safetensors"),
            "max_length": MAX_LENGTH,
            "document_batch_size": DOCUMENT_BATCH_SIZE,
            "indexed_documents": len(docs),
            "document_encoding_seconds": indexing_seconds,
        },
        "modes": {mode: {**group_results(rows), "per_query": rows} for mode, rows in results.items()},
    }


def get_scorer(args: argparse.Namespace):
    if args.model == "bge":
        from mail_search.reranker import Reranker
        scorer = Reranker(args.model_path, args.model_revision, args.max_length, args.train_layers)
    elif args.model == "qwen-ft":
        from mail_search.qwen_train import TrainableQwen
        scorer = TrainableQwen(args.model_path, args.model_revision, args.max_length, args.train_layers)
    else:
        from mail_search.zero_shot import QwenScorer
        return QwenScorer(args.model_path, args.max_length)
    scorer.load_weights(args.weights)
    scorer.model.eval()
    return scorer


def score(args: argparse.Namespace, data: FriendData) -> dict:
    candidates = json.loads(args.candidates.read_text())
    if candidates["protocol"] != data.protocol or candidates["modes"].keys() != set(MODES):
        raise ValueError("Candidate file has a different dataset, split or date protocol")
    boxes = documents_by_mailbox(data.dataset)
    rows: dict[str, dict[str, dict]] = {}
    for mode in MODES:
        subgroup = candidates["modes"][mode]["per_query"]
        rows[mode] = {row["query_id"]: row for row in subgroup}
        if len(subgroup) != len(data.dataset.queries) or rows[mode].keys() != data.dataset.queries.keys():
            raise ValueError(f"Wrong query coverage in {mode} candidate file")

    configure_cuda(args.gpu_memory_limit_gib)
    scorer = get_scorer(args)
    scorer.score("прогрев поиска", ["прогрев оценки релевантности"] * args.batch_size)
    results: dict[str, list[dict]] = {mode: [] for mode in MODES}
    for query_id, query in sorted(data.dataset.queries.items()):
        relevant = data.dataset.qrels[query_id]
        mode_rows = {mode: rows[mode][query_id] for mode in MODES}
        for mode, row in mode_rows.items():
            expected, allowed = data.date_pool(query, boxes[query.mailbox], mode)
            ids = row["ranked_doc_ids"]
            if expected != {key: row[key] for key in expected} or (
                len(ids) != min(len(allowed), TOP_K) or len(set(ids)) != len(ids)
                or not set(ids) <= set(allowed)
            ):
                raise ValueError(f"Invalid candidates for {mode} {query_id}")

        scoring_ids = sorted(set().union(*(row["ranked_doc_ids"] for row in mode_rows.values())))
        scores = []
        start = perf_counter()
        for offset in range(0, len(scoring_ids), args.batch_size):
            batch_ids = scoring_ids[offset:offset + args.batch_size]
            batch = [searchable_text(data.dataset.documents[doc_id]) for doc_id in batch_ids]
            scores.extend(scorer.score(query.text, batch))
            if args.batch_pause_ms:
                sleep(args.batch_pause_ms / 1000)
        elapsed = (perf_counter() - start) * 1000
        if not all(map(math.isfinite, scores)):
            raise ValueError(f"Non-finite reranker score for {query_id}")
        logits = dict(zip(scoring_ids, scores, strict=True))
        for mode, row in mode_rows.items():
            reranked = sorted(row["ranked_doc_ids"], key=lambda d: (-logits[d], d))
            results[mode].append({
                **row,
                "reranked_doc_ids": reranked,
                "reranker_logits": {doc_id: logits[doc_id] for doc_id in row["ranked_doc_ids"]},
                "reranking_metrics": measure(reranked, relevant),
                "shared_reranking_ms_offline": elapsed,
                "shared_reranking_pairs": len(scoring_ids),
            })
    return {
        "protocol": data.protocol,
        "environment": gpu_environment(args),
        "candidates_sha256": sha256(args.candidates),
        "model": {
            "name": args.model,
            "revision": args.model_revision,
            "base_weights_sha256": sha256(args.model_path / "model.safetensors"),
            "checkpoint_sha256": sha256(args.weights) if args.weights else None,
            "max_length": args.max_length,
            "train_layers": args.train_layers if args.weights else None,
            "batch_size": args.batch_size,
            "batch_pause_ms": args.batch_pause_ms,
        },
        "modes": {mode: {**group_results(rows), "per_query": rows} for mode, rows in results.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the 40 added VK queries on shared candidate pools.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--manual-documents", type=Path)
    retrieve_parser = subparsers.add_parser("retrieve")
    retrieve_parser.add_argument("--e5-model", type=Path, required=True)
    retrieve_parser.add_argument("--e5-revision", required=True)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--candidates", type=Path, required=True)
    score_parser.add_argument("--model", choices=("bge", "qwen", "qwen-ft"), required=True)
    score_parser.add_argument("--model-path", type=Path, required=True)
    score_parser.add_argument("--model-revision", required=True)
    score_parser.add_argument("--weights", type=Path)
    score_parser.add_argument("--max-length", type=int, default=256)
    score_parser.add_argument("--train-layers", type=int, default=2)
    score_parser.add_argument("--batch-size", type=int, default=20)
    score_parser.add_argument("--batch-pause-ms", type=float, default=0)
    for subparser in (audit_parser, retrieve_parser, score_parser):
        for name in ("original-archive", "augmented-archive", "splits", "output"):
            subparser.add_argument(f"--{name}", type=Path, required=True)
        subparser.add_argument("--base-commit", required=True)
    for subparser in (retrieve_parser, score_parser):
        subparser.add_argument("--gpu-memory-limit-gib", type=float, default=6)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("The result already exists")
    if args.command != "audit" and args.gpu_memory_limit_gib <= 0:
        parser.error("GPU limit must be positive")
    if args.command == "score":
        if (args.model != "qwen") != (args.weights is not None):
            parser.error("--weights is required for fine-tuned models and unavailable for frozen Qwen")
        if args.batch_size < 1 or args.batch_pause_ms < 0:
            parser.error("Invalid batch size or pause")
    data = load_friend_data(args.original_archive, args.augmented_archive, args.splits)
    operation = {"audit": audit, "retrieve": retrieve, "score": score}[args.command]
    write_json(args.output, operation(args, data))


if __name__ == "__main__":
    main()
