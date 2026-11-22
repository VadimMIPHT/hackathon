import argparse
import hashlib
import json
import os
import platform
import statistics
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter, sleep

from mail_search.benchmark import measure, source_hashes, summarize
from mail_search.dataset import Dataset, Query, load_archive, read_split
from mail_search.reranker_data import documents_by_mailbox, load_candidates
from mail_search.retrieval import eligible_indices, searchable_text

DEPTHS = (20, 50, None)
BATCH_SIZE = 20


@dataclass(frozen=True)
class QueryPool:
    query: Query
    e5: list[str]
    documents: list[str]

    def candidates(self, depth: int | None) -> list[str]:
        return self.documents if depth is None else self.e5[:depth]


def load_pools(
    dataset: Dataset, split_manifest: Path, e5_path: Path,
) -> tuple[list[QueryPool], dict]:
    users, manifest = read_split(split_manifest, dataset, "val")
    candidates, e5_config = load_candidates(e5_path, dataset, split_manifest, users, "val")
    by_mailbox = documents_by_mailbox(dataset)
    pools = []
    for group in candidates:
        query = group.query
        docs = by_mailbox[query.mailbox]
        eligible = sorted(docs[i].doc_id for i in eligible_indices(docs, query, use_query_day=True))
        if not dataset.qrels[query.query_id].keys() <= set(eligible):
            raise ValueError(f"Date filter excludes a relevant document: {query.query_id}")
        pools.append(QueryPool(query, group.ranked_doc_ids, eligible))
    return pools, {"split_seed": manifest["seed"], "e5": e5_config}


def coverage_report(dataset: Dataset, pools: list[QueryPool]) -> dict:
    result = {}
    for depth in DEPTHS:
        rows = []
        for pool in pools:
            relevant = dataset.qrels[pool.query.query_id]
            available = pool.candidates(depth)
            positives = [doc_id for doc_id in available if doc_id in relevant]
            rows.append({
                "query_id": pool.query.query_id,
                "mailbox": pool.query.mailbox,
                "available": len(available), "found_relevant": len(positives),
                "relevant": len(relevant), "candidate_recall": len(positives) / len(relevant),
                "oracle": measure(positives, relevant),
            })
        result[str(depth or "all")] = {
            "pairs": sum(row["available"] for row in rows),
            "median_pairs_per_query": statistics.median(row["available"] for row in rows),
            "max_pairs_per_query": max(row["available"] for row in rows),
            "relevant_found": sum(row["found_relevant"] for row in rows),
            "relevant_total": sum(row["relevant"] for row in rows),
            "macro_candidate_recall": statistics.mean(row["candidate_recall"] for row in rows),
            "oracle_ndcg@10": statistics.mean(row["oracle"]["ndcg@10"] for row in rows),
            "oracle_recall@10": statistics.mean(row["oracle"]["recall@10"] for row in rows),
            "per_query": rows,
        }
    return result


def evaluate_depths(dataset: Dataset, pools: list[QueryPool], score_rows: list[dict]) -> dict:
    scores_by_query = {row["query_id"]: row for row in score_rows}
    if set(scores_by_query) != {pool.query.query_id for pool in pools}:
        raise ValueError("Full scoring does not cover all evaluation queries")
    results = {}
    for depth in DEPTHS:
        rows = []
        for pool in pools:
            query = pool.query
            source = scores_by_query[query.query_id]
            scores = dict(zip(source["doc_ids"], source["scores"], strict=True))
            if set(scores) != set(pool.documents):
                raise ValueError(f"Scored documents differ from date pool: {query.query_id}")
            started = perf_counter()
            ranked = sorted(pool.candidates(depth), key=lambda doc_id: (-scores[doc_id], doc_id))
            if depth is not None:
                ranked += pool.e5[depth:]
            ranked = ranked[:50]
            sort_ms = (perf_counter() - started) * 1000
            rows.append({
                "query_id": query.query_id, "mailbox": query.mailbox, "ranked_doc_ids": ranked,
                "metrics": measure(ranked, dataset.qrels[query.query_id]),
                "elapsed_ms": source["score_ms"] + sort_ms,
            })
        summary = summarize(rows)
        if depth is not None:
            # Restricted rankings reuse full-pool scores. The raw elapsed time is for the full pass.
            summary.pop("p50_ms_offline")
            summary.pop("p99_ms_offline")
            for row in rows:
                del row["elapsed_ms"]
        results[str(depth or "all")] = {"summary": summary, "per_query": rows}
    return results


def write_json(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def base_config(args: argparse.Namespace, dataset: Dataset, metadata: dict) -> dict:
    return {
        "archive_sha256": dataset.archive_sha256, "split": "val", **metadata,
        "split_manifest_sha256": hashlib.sha256(args.splits.read_bytes()).hexdigest(),
        "e5_candidates_sha256": hashlib.sha256(args.e5_candidates.read_bytes()).hexdigest(),
        "code_base_commit": args.base_commit, "source_sha256": source_hashes(),
        "filters": "same mailbox and requested day, accessible at query date",
        "tie_break": "doc_id",
    }


def configure_cuda(memory_limit_gib: float | None) -> None:
    import torch

    torch.set_num_threads(2)
    if memory_limit_gib is not None:
        limit = memory_limit_gib * (1 << 30)
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(limit / total, 1.0), device=0)


def score_all_model(args: argparse.Namespace, dataset: Dataset, pools: list[QueryPool]) -> dict:
    import torch

    from mail_search.reranker import Reranker
    from mail_search.zero_shot import QWEN_INSTRUCTION, QwenScorer, RuBertScorer

    configure_cuda(args.gpu_memory_limit_gib)
    if args.name == "bge":
        model = Reranker(args.model, args.revision, max_length=256, train_layers=2)
        if args.weights:
            model.load_weights(args.weights)
    elif args.name == "qwen":
        model = QwenScorer(args.model, 256)
    else:
        model = RuBertScorer(args.model, 256)
    model.model.eval()
    model.score("прогрев поиска", ["прогрев оценки релевантности"] * BATCH_SIZE)
    rows = []
    for group in pools:
        query = group.query
        started = perf_counter()
        docs = [searchable_text(dataset.documents[doc_id]) for doc_id in group.documents]
        scores = []
        for start in range(0, len(docs), BATCH_SIZE):
            scores.extend(model.score(query.text, docs[start:start + BATCH_SIZE]))
            if args.batch_pause_ms:
                sleep(args.batch_pause_ms / 1000)
        rows.append({
            "query_id": query.query_id, "mailbox": query.mailbox, "doc_ids": group.documents,
            "scores": scores, "score_ms": (perf_counter() - started) * 1000,
        })
    return {
        "environment": {
            "python": platform.python_version(), "host": platform.node(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "transformers": version("transformers"),
            "tokenizers": version("tokenizers"), "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_memory_limit_gib": args.gpu_memory_limit_gib,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(0) / (1 << 30),
        },
        "model": {"name": args.name, "revision": args.revision, "max_length": 256,
                  "batch_size": BATCH_SIZE, "batch_pause_ms": args.batch_pause_ms,
                  "instruction": QWEN_INSTRUCTION if args.name == "qwen" else None,
                  "checkpoint_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest() if args.weights else None},
        "scores": rows, "depths": evaluate_depths(dataset, pools, rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare E5 depth limits with full-date scoring.")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--e5-candidates", type=Path, required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--name", choices=("coverage", "bge", "qwen", "rubert"), required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--gpu-memory-limit-gib", type=float)
    parser.add_argument("--batch-pause-ms", type=float, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.name != "coverage" and not (args.model and args.revision):
        parser.error("--model and --revision are required for neural scoring")
    if args.weights and args.name != "bge":
        parser.error("--weights only applies to BGE")
    if (args.gpu_memory_limit_gib is not None and args.gpu_memory_limit_gib <= 0) or args.batch_pause_ms < 0:
        parser.error("GPU memory limit must be positive and batch pause non-negative")

    dataset = load_archive(args.archive)
    pools, metadata = load_pools(dataset, args.splits, args.e5_candidates)
    result = {"config": base_config(args, dataset, metadata)}
    if args.name == "coverage":
        result["depths"] = coverage_report(dataset, pools)
    else:
        result.update(score_all_model(args, dataset, pools))
    write_json(args.output, result)


if __name__ == "__main__":
    main()
