import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import catboost

from mail_search.benchmark import measure, source_hashes, summarize
from mail_search.dataset import Dataset, load_archive, read_split


def fit_ranker(dataset: Dataset, groups: list[dict], features: list[str]) -> tuple:
    x, labels, query_ids, group_weights = [], [], [], []
    mixed_groups = 0
    for group in groups:
        qid = group["query_id"]
        relevant = dataset.qrels[qid]
        y = [int(doc_id in relevant) for doc_id in group["e5_ranked_doc_ids"][:len(group["features"])]]
        positives = sum(y)
        if not positives or positives == len(y):
            continue
        mixed_groups += 1
        x.extend(group["features"])
        labels.extend(y)
        query_ids.extend([qid] * len(y))
        group_weights.extend([1 / (positives * (len(y) - positives))] * len(y))

    fit_parameters = {
        "loss_function": "PairLogit", "depth": 2, "iterations": 100,
        "learning_rate": 0.08, "l2_leaf_reg": 5, "random_seed": 42,
        "thread_count": 2, "verbose": False, "allow_writing_files": False,
    }
    model = catboost.CatBoostRanker(**fit_parameters)
    started = perf_counter()
    model.fit(catboost.Pool(
        x, label=labels, group_id=query_ids, group_weight=group_weights, feature_names=features,
    ))
    result = {
        "mixed_queries": mixed_groups, "pairs_in_mixed_queries": len(labels),
        "positives": sum(labels), "parameters": fit_parameters,
        "train_seconds": perf_counter() - started,
    }
    return model, result


def evaluate_ranker(model: catboost.CatBoostRanker, dataset: Dataset, groups: list[dict]) -> dict:
    model.predict(groups[0]["features"], thread_count=2)
    rows = []
    for group in groups:
        qid = group["query_id"]
        original = group["e5_ranked_doc_ids"]
        features = group["features"]
        started = perf_counter()
        scores = model.predict(features, thread_count=2).tolist()
        depth = len(features)
        head = sorted(range(depth), key=lambda position: (-scores[position], position))
        ranking = [original[i] for i in head] + original[depth:]
        scoring_ms = (perf_counter() - started) * 1000
        rows.append({
            "query_id": qid, "mailbox": group["mailbox"], "ranked_doc_ids": ranking,
            "in_e5_order_logits": scores, "boost_scoring_ms": scoring_ms,
            "elapsed_ms": group["feature_ms"] + scoring_ms,
            "metrics": measure(ranking, dataset.qrels[qid]),
        })
    return {
        "latency_scope": "frozen_bge_and_pair_features_plus_catboost_after_bm25_and_e5",
        "summary": summarize(rows),
        "by_mailbox": {
            mailbox: summarize([row for row in rows if row["mailbox"] == mailbox])
            for mailbox in sorted({row["mailbox"] for row in rows})
        },
        "per_query": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a shallow CatBoost pairwise ranker.")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = load_archive(args.archive)
    payload = json.loads(args.input.read_text())
    metadata = payload["config"]
    if metadata["archive_sha256"] != dataset.archive_sha256:
        raise ValueError("Feature table archive does not match")
    for split in ("train", "val"):
        users, _ = read_split(args.splits, dataset, split)
        expected = {q.query_id for q in dataset.queries.values() if q.mailbox in users}
        rows = payload["splits"][split]
        if len(rows) != len(expected) or {row["query_id"] for row in rows} != expected:
            raise ValueError("Feature table does not match query split")
    model, train = fit_ranker(dataset, payload["splits"]["train"], metadata["features"])
    result = {
        "environment": {
            "python": platform.python_version(), "host": platform.node(),
            "catboost": catboost.__version__, "source_sha256": source_hashes(),
        },
        "config": {
            **metadata, "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "split_manifest_sha256": hashlib.sha256(args.splits.read_bytes()).hexdigest(),
        },
        "train": train,
        **evaluate_ranker(model, dataset, payload["splits"]["val"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.output.with_suffix(".cbm"))
    temporary = args.output.with_suffix(".partial")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
