import argparse
import json
import math
from pathlib import Path
from statistics import mean

import evaluate
from mail_search.benchmark import METRICS, measure
from mail_search.dataset import load_archive
from mail_search.depth_benchmark import write_json
from mail_search.friend_eval_data import sha256
from transfer import NEW_IDS

ROOT = Path(__file__).resolve().parent


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def summary(rows: dict, ids: set) -> dict:
    values = {metric: [rows[qid][metric] for qid in sorted(ids) if rows[qid][metric] is not None]
              for metric in METRICS}
    return {metric: {"mean": mean(seq) if seq else None, "n": len(seq)} for metric, seq in values.items()}


def check(computed: dict, saved: dict) -> None:
    for metric in METRICS:
        a, b = computed[metric], saved[metric]
        assert a is b if a is None or b is None else math.isclose(a, b, abs_tol=1e-12)


def compare(args: argparse.Namespace) -> dict:
    _, _, old, _ = evaluate.read_jev(args)
    original = load_archive(args.original_archive)
    files = {name: ROOT / name for name in ("train.json", "test.json", "train_manifest.json", "split.json",
                                           "transfer.py", "compare.py", "run_h200.sh")}
    tr, test, split, manifest = [read(files[name]) for name in ("train.json", "test.json", "split.json", "train_manifest.json")]
    e5, previous = read(args.candidates), read(args.previous)
    assert tr["complete"] and tr["config"]["epochs"] == len(tr["epochs"]) == 2
    assert tr["config"]["archive_sha256"] == manifest["archive_sha256"] == old.dataset.archive_sha256
    assert set(tr["train_query_ids"]) == manifest["train"].keys() == NEW_IDS
    assert tr["config"]["train_manifest_sha256"] == sha256(files["train_manifest.json"])
    assert tr["script_sha256"] == sha256(files["transfer.py"])
    assert tr["pairs"] == sum(map(len, manifest["train"].values()))
    assert tr["positives"] == sum(y for labels in manifest["train"].values() for y in labels.values())
    assert test["train_log_sha256"] == sha256(files["train.json"])
    assert test["test_split_sha256"] == sha256(files["split.json"])
    assert test["protocol"] == e5["protocol"] == previous["protocol"] == old.protocol
    assert test["candidate_sha256"] == previous["candidates_sha256"] == sha256(args.candidates)
    assert test["models"].keys() == {"bge_base", "bge_new_only"}
    assert split["test"].keys() == original.queries.keys() == old.dataset.queries.keys()
    assert not NEW_IDS & split["test"].keys()
    all_ids = set(split["test"])
    groups = {"all": all_ids,
              "related_originals": {q for q in all_ids if split["test"][q]["related_original_query"]},
              "unseen_mailboxes": {q for q in all_ids if split["test"][q]["unseen_mailbox"]}}
    groups["other_originals"] = all_ids - groups["related_originals"]
    train_docs = {doc for labels in manifest["train"].values() for doc in labels}
    output = {"source_sha256": {**{name: sha256(path) for name, path in files.items()},
                                 "evaluate_local.py": sha256(Path(evaluate.__file__)),
                                 "e5_candidates": sha256(args.candidates), "previous_bge": sha256(args.previous)},
              "log": tr, "test_environment": test["environment"], "metrics": {}}
    for mode in e5["modes"]:
        models = {"e5": e5["modes"][mode]["per_query"], "bge_previous": previous["modes"][mode]["per_query"],
                  **{name: modes[mode] for name, modes in test["models"].items()}}
        candidate_sets = {row["query_id"]: set(row["ranked_doc_ids"]) for row in models["e5"]}
        output["metrics"][mode] = {}
        for label_set, qrels in (("jev", old.dataset.qrels), ("public_vk", original.qrels)):
            entries = {}
            for model, rows in models.items():
                result = {}
                for row in rows:
                    qid = row["query_id"]
                    ranking = row.get("reranked_doc_ids", row["ranked_doc_ids"])
                    assert qid not in result and set(ranking) == candidate_sets[qid] and len(set(ranking)) == len(ranking)
                    if "logits" in row:
                        assert ranking == sorted(row["logits"], key=lambda did: (-row["logits"][did], did))
                    scores = measure(ranking, qrels[qid])
                    if label_set in row:
                        check(scores, row[label_set])
                    elif label_set == "jev":
                        check(scores, row.get("reranking_metrics", row["retrieval_metrics"]))
                    result[qid] = scores
                assert result.keys() == all_ids
                entries[model] = result
            no_overlap = {qid for qid, relevant in qrels.items() if not relevant.keys() & train_docs}
            data = {}
            for group, qids in {**groups, "no_seen_relevant_letters": no_overlap}.items():
                delta = [entries["bge_new_only"][qid]["ndcg@10"] - entries["bge_base"][qid]["ndcg@10"]
                         for qid in sorted(qids)]
                data[group] = {"queries": len(qids), "mailboxes": len({split["test"][q]["mailbox"] for q in qids}),
                               "metrics": {name: summary(rows, qids) for name, rows in entries.items()},
                               "new_only_vs_base_ndcg": {"mean_delta": mean(delta), "wins": sum(v > 1e-12 for v in delta),
                                                        "ties": sum(abs(v) <= 1e-12 for v in delta),
                                                        "losses": sum(v < -1e-12 for v in delta)}}
            output["metrics"][mode][label_set] = data
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("original-archive", "augmented-archive", "jev-archive", "splits", "candidates", "previous"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    write_json(ROOT / "comparison.json", compare(parser.parse_args()))
