import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

from mail_search.dataset import Dataset, load_archive
from mail_search.depth_benchmark import (
    QueryPool, base_config, configure_cuda, evaluate_depths, load_pools, write_json,
)
from mail_search.retrieval import RRF_DEPTH, searchable_text


def prepare(args: argparse.Namespace, dataset: Dataset, pools: list[QueryPool]) -> dict:
    from mail_search.boost_features import FEATURES, lexical_features
    from mail_search.e5 import E5

    configure_cuda(args.gpu_memory_limit_gib)
    bge = json.loads(args.frozen_bge_scores.read_text())
    if (bge["config"]["archive_sha256"] != dataset.archive_sha256
            or bge["config"]["e5_candidates_sha256"]
            != hashlib.sha256(args.e5_candidates.read_bytes()).hexdigest()
            or bge["model"]["name"] != "bge" or bge["model"]["checkpoint_sha256"] is not None):
        raise ValueError("Expected frozen BGE scores on the same corpus and candidates")
    frozen = {row["query_id"]: row for row in bge["scores"]}
    bm25 = json.loads(args.e5_candidates.read_text())["runs"]["query_day"]["per_query"]
    bm25_ranks = {
        row["query_id"]: {doc_id: rank for rank, doc_id in enumerate(row["ranked_doc_ids"], 1)}
        for row in bm25 if row["method"] == "bm25"
    }
    model = E5(args.e5_model)
    ids = sorted({doc_id for pool in pools for doc_id in pool.documents})
    embeddings = model.encode([searchable_text(dataset.documents[doc_id]) for doc_id in ids], "passage: ")
    vectors = dict(zip(ids, embeddings, strict=True))
    queries = model.encode([pool.query.text for pool in pools], "query: ")

    rows = []
    for pool, query_vector in zip(pools, queries, strict=True):
        qid = pool.query.query_id
        score_row = frozen[qid]
        frozen_logits = dict(zip(score_row["doc_ids"], score_row["scores"], strict=True))
        if set(frozen_logits) != set(pool.documents):
            raise ValueError(f"Frozen scores have an unexpected candidate pool: {qid}")
        cosines = {doc_id: float(vectors[doc_id] @ query_vector) for doc_id in pool.documents}
        rest = set(pool.documents) - set(pool.e5)
        ranking = pool.e5 + sorted(rest, key=lambda doc_id: (-cosines[doc_id], doc_id))
        started = perf_counter()
        features = [
            [rank, cosines[doc_id], bm25_ranks[qid].get(doc_id, RRF_DEPTH + 1), frozen_logits[doc_id],
             *lexical_features(pool.query.text, dataset.documents[doc_id])]
            for rank, doc_id in enumerate(ranking, 1)
        ]
        rows.append({
            "query_id": qid, "doc_ids": ranking, "features": features,
            "feature_ms": (perf_counter() - started) * 1000 + score_row["score_ms"],
        })
    return {"features": FEATURES, "rows": rows,
            "frozen_bge_scores_sha256": hashlib.sha256(args.frozen_bge_scores.read_bytes()).hexdigest()}


def predict(args: argparse.Namespace, dataset: Dataset, pools: list[QueryPool]) -> dict:
    import catboost

    payload = json.loads(args.features.read_text())
    if (payload["config"]["archive_sha256"] != dataset.archive_sha256
            or payload["config"]["e5_candidates_sha256"]
            != hashlib.sha256(args.e5_candidates.read_bytes()).hexdigest()):
        raise ValueError("Feature table was computed for a different candidate set")
    ranker = catboost.CatBoostRanker()
    ranker.load_model(args.model)
    if ranker.feature_names_ != payload["features"]:
        raise ValueError("CatBoost feature names do not match its fitted model")
    ranker.predict(payload["rows"][0]["features"], thread_count=2)
    rows = []
    for row in payload["rows"]:
        started = perf_counter()
        scores = ranker.predict(row["features"], thread_count=2).tolist()
        rows.append({
            "query_id": row["query_id"], "doc_ids": row["doc_ids"], "scores": scores,
            "score_ms": row["feature_ms"] + (perf_counter() - started) * 1000,
        })
    return {
        "environment": {"python": platform.python_version(), "catboost": catboost.__version__},
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "latency_scope": "frozen BGE, lexical features, CatBoost; E5/BM25 already prepared",
        "scores": rows, "depths": evaluate_depths(dataset, pools, rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the fitted CatBoost ranker beyond the E5 cutoff.")
    parser.add_argument("stage", choices=("prepare", "predict"))
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--e5-candidates", type=Path, required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--e5-model", type=Path)
    parser.add_argument("--frozen-bge-scores", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--gpu-memory-limit-gib", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "prepare" and not (args.e5_model and args.frozen_bge_scores):
        parser.error("prepare requires --e5-model and --frozen-bge-scores")
    if args.stage == "predict" and not (args.features and args.model):
        parser.error("predict requires --features and --model")
    if args.gpu_memory_limit_gib is not None and args.gpu_memory_limit_gib <= 0:
        parser.error("GPU memory limit must be positive")
    dataset = load_archive(args.archive)
    pools, metadata = load_pools(dataset, args.splits, args.e5_candidates)
    output = prepare(args, dataset, pools) if args.stage == "prepare" else predict(args, dataset, pools)
    write_json(args.output, {"config": base_config(args, dataset, metadata), **output})


if __name__ == "__main__":
    main()
