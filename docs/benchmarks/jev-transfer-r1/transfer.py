import argparse
import json
import logging
import math
import random
from collections import Counter
from datetime import date
from pathlib import Path
from time import perf_counter, sleep
from zipfile import ZipFile

from evaluate import read_jev
from mail_search.benchmark import measure
from mail_search.dataset import Document, load_archive
from mail_search.depth_benchmark import configure_cuda, write_json
from mail_search.friend_eval import MODES, gpu_environment
from mail_search.friend_eval_data import sha256
from mail_search.reranker_data import TrainingPair, documents_by_mailbox
from mail_search.retrieval import searchable_text

LOGGER = logging.getLogger(__name__)
NEW_IDS = {f"N-{i:03d}" for i in range(1, 41)}
REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
BASE_SHA = "d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286"
TOP_K = 50
SEED = 42
EPOCHS = 2
BATCH_SIZE = 16
SCORING_BATCH = 20
MAX_LENGTH = 256
TRAIN_LAYERS = 2
LEARNING_RATE = 2e-5
BATCH_PAUSE_MS = 30
CAP_GIB = 6


def prepare(args: argparse.Namespace) -> None:
    previous, new, old, meta = read_jev(args)
    audit = json.loads(args.audit.read_text())["annotation_audit"]
    if new.dataset.archive_sha256 != audit["archive_sha256"] or audit["missing_pairs"]:
        raise ValueError("The Jev archive differs from the verified complete annotation")
    if set(new.query_ids.values()) != NEW_IDS or set(old.dataset.queries) & NEW_IDS:
        raise ValueError("Unexpected train/test IDs")
    boxes = documents_by_mailbox(new.dataset)
    queries = {}
    for old_id, query in sorted(new.dataset.queries.items()):
        _, eligible = new.date_pool(query, boxes[query.mailbox], "date-aware")
        relevant = new.dataset.qrels[old_id]
        if not relevant.keys() <= set(eligible):
            raise ValueError(f"Date filter removed a positive for {old_id}")
        queries[new.query_ids[old_id]] = {doc_id: int(doc_id in relevant) for doc_id in sorted(eligible)}
    train_docs = set().union(*(set(labels) for labels in queries.values()))
    train_positives = {d for labels in queries.values() for d, y in labels.items() if y}
    train_boxes = {query.mailbox for query in new.dataset.queries.values()}
    source_ids = set().union(*map(set, previous.old_sources.values()))
    old_public = public_positive_sets(meta["original_positive"])
    split = {
        "protocol": new.protocol, "manifest": meta["manifest"],
        "test": {
            qid: {"mailbox": query.mailbox, "related_original_query": qid in source_ids,
                  "public_positive_overlap_with_any_train_letter": len(old_public[qid] & train_docs),
                  "public_positive_overlap_with_positive_train_letter": len(old_public[qid] & train_positives),
                  "jev_positive_overlap_with_any_train_letter": len(old.dataset.qrels[qid].keys() & train_docs),
                  "unseen_mailbox": query.mailbox not in train_boxes}
            for qid, query in sorted(old.dataset.queries.items())
        },
    }
    write_json(args.output_dir / "train_manifest.json",
               {"archive_sha256": new.dataset.archive_sha256, "train": queries})
    write_json(args.output_dir / "split.json", split)
    LOGGER.info("Training query IDs: %d. Label counts: %s. Held-out query IDs: %d", len(queries),
                dict(Counter(y for labels in queries.values() for y in labels.values())), len(split["test"]))


def public_positive_sets(rows: list[dict]) -> dict[str, set[str]]:
    positives: dict[str, set[str]] = {}
    for row in rows:
        positives.setdefault(row["query_id"], set()).add(row["doc_id"])
    return positives


def load_pairs(args: argparse.Namespace) -> tuple[list[TrainingPair], dict]:
    manifest = json.loads(args.train_manifest.read_text())
    if manifest["archive_sha256"] != sha256(args.jev_archive) or manifest["train"].keys() != NEW_IDS:
        raise ValueError("Training manifest must contain exactly the 40 new Jev query IDs")
    requested = set().union(*(labels.keys() for labels in manifest["train"].values()))
    with ZipFile(args.jev_archive) as archive:
        prefix = "vk-jev-dataset/"
        queries = {row["query_id"]: row for row in map(json.loads, archive.open(prefix + "queries_dev.jsonl"))
                   if row["query_id"] in NEW_IDS}
        docs = {row["doc_id"]: row for row in map(json.loads, archive.open(prefix + "documents.jsonl"))
                if row["doc_id"] in requested}
    if queries.keys() != NEW_IDS or docs.keys() != requested:
        raise ValueError("Missing training query or document")
    texts = {did: searchable_text(Document(did, row["synthetic_user"], row["doc"],
                                          date.fromisoformat(row["doc_dt"]) if row["doc_dt"] else None))
             for did, row in docs.items()}
    sizes = {qid: len(labels) for qid, labels in manifest["train"].items()}
    count = sum(sizes.values())
    pairs = []
    for qid, labels in sorted(manifest["train"].items()):
        query = queries[qid]
        weight = count / (len(queries) * sizes[qid])
        for did, label in sorted(labels.items()):
            if label not in (0, 1) or docs[did]["synthetic_user"] != query["synthetic_user"]:
                raise ValueError(f"Invalid training pair {qid}/{did}")
            pairs.append(TrainingPair(query["query"], texts[did], label, weight))
    return pairs, manifest


def get_bge(args: argparse.Namespace):
    from mail_search.reranker import Reranker

    if sha256(args.model / "model.safetensors") != BASE_SHA:
        raise ValueError("Unexpected BGE base weights")
    configure_cuda(CAP_GIB)
    return Reranker(args.model, REVISION, MAX_LENGTH, TRAIN_LAYERS)


def train(args: argparse.Namespace) -> None:
    import torch

    if args.original_archive or args.augmented_archive or args.candidates or args.split:
        raise ValueError("Pass only training inputs to the training stage")
    pairs, manifest = load_pairs(args)
    torch.manual_seed(SEED)
    bge = get_bge(args)
    optimizer = torch.optim.AdamW(bge.weights.values(), lr=LEARNING_RATE)
    rng = random.Random(SEED)
    checkpoint = args.output_dir / "epoch-2.safetensors"
    metrics_file = args.output_dir / "train.json"
    if checkpoint.exists() or metrics_file.exists():
        raise FileExistsError("Training outputs already exist")
    metrics = {
        "environment": gpu_environment(args), "script_sha256": sha256(Path(__file__)),
        "config": {"seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "max_length": MAX_LENGTH,
                   "train_layers": TRAIN_LAYERS, "batch_pause_ms": BATCH_PAUSE_MS,
                   "repo": "BAAI/bge-reranker-v2-m3", "revision": REVISION, "base_weights_sha256": BASE_SHA,
                   "archive_sha256": manifest["archive_sha256"], "train_manifest_sha256": sha256(args.train_manifest),
                   "loss": "BCE with logits, equal total weight per query", "optimizer": "AdamW",
                   "learning_rate": LEARNING_RATE, "weight_decay": 0.01, "autocast_dtype": "bf16"},
        "train_query_ids": sorted(manifest["train"]), "pairs": len(pairs),
        "positives": sum(pair.label for pair in pairs), "epochs": [], "complete": False,
    }
    for epoch in range(1, EPOCHS + 1):
        start = perf_counter()
        loss = bge.fit_epoch(pairs, optimizer, BATCH_SIZE, rng, BATCH_PAUSE_MS)
        if not math.isfinite(loss):
            raise ValueError("Non-finite training loss")
        elapsed = perf_counter() - start
        metrics["epochs"].append({"epoch": epoch, "loss": loss, "seconds": elapsed})
        write_json(metrics_file, metrics)
        LOGGER.info("Epoch %d/%d. Train BCE: %.5f. %.1f seconds", epoch, EPOCHS, loss, elapsed)
    bge.save_weights(checkpoint)
    metrics["checkpoint_sha256"] = sha256(checkpoint)
    metrics["complete"] = True
    metrics["environment"] = gpu_environment(args)
    write_json(metrics_file, metrics)


def test(args: argparse.Namespace) -> None:
    _, _, dataset, _ = read_jev(args)
    original = load_archive(args.original_archive)
    split = json.loads(args.split.read_text())
    result = json.loads((args.output_dir / "train.json").read_text())
    if not result["complete"] or set(result["train_query_ids"]) != NEW_IDS:
        raise ValueError("No complete new-only training run")
    artifact = json.loads(args.candidates.read_text())
    if (artifact["protocol"] != dataset.protocol or dataset.dataset.queries.keys() != split["test"].keys()
            or dataset.dataset.queries.keys() & NEW_IDS):
        raise ValueError("The test candidates must cover only the 99 original queries")
    boxes = documents_by_mailbox(dataset.dataset)
    cached = {mode: {row["query_id"]: row for row in artifact["modes"][mode]["per_query"]} for mode in MODES}
    if any(rows.keys() != dataset.dataset.queries.keys() for rows in cached.values()):
        raise ValueError("Incomplete E5 candidate cache")
    path = args.output_dir / "epoch-2.safetensors"
    if sha256(path) != result["checkpoint_sha256"]:
        raise ValueError("Unexpected trained checkpoint")
    bge = get_bge(args)
    output = {"candidate_sha256": sha256(args.candidates), "train_log_sha256": sha256(args.output_dir / "train.json"),
              "test_split_sha256": sha256(args.split), "protocol": dataset.protocol,
              "scoring_batch_size": SCORING_BATCH, "max_length": MAX_LENGTH, "models": {}}
    for name in ("bge_base", "bge_new_only"):
        if name == "bge_new_only":
            bge.load_weights(path)
        bge.model.eval()
        bge.score("прогрев поиска", ["прогрев оценки релевантности"] * SCORING_BATCH)
        per_mode = {mode: [] for mode in MODES}
        for qid, query in sorted(dataset.dataset.queries.items()):
            candidates = {}
            for mode in MODES:
                metadata, allowed = dataset.date_pool(query, boxes[query.mailbox], mode)
                row = cached[mode][qid]
                ids = row["ranked_doc_ids"]
                if (metadata != {key: row[key] for key in metadata} or len(ids) != min(TOP_K, len(allowed))
                        or len(set(ids)) != len(ids) or not set(ids) <= set(allowed)):
                    raise ValueError(f"Invalid candidates for {qid}/{mode}")
                candidates[mode] = ids
            doc_ids = sorted(set().union(*candidates.values()))
            values = []
            for start in range(0, len(doc_ids), SCORING_BATCH):
                batch = doc_ids[start:start + SCORING_BATCH]
                values.extend(bge.score(query.text, [searchable_text(dataset.dataset.documents[did]) for did in batch]))
                sleep(BATCH_PAUSE_MS / 1000)
            logits = dict(zip(doc_ids, values, strict=True))
            for mode, ids in candidates.items():
                ranking = sorted(ids, key=lambda did: (-logits[did], did))
                per_mode[mode].append({"query_id": qid, "ranked_doc_ids": ranking,
                                      "logits": {did: logits[did] for did in ids},
                                      "jev": measure(ranking, dataset.dataset.qrels[qid]),
                                      "public_vk": measure(ranking, original.qrels[qid])})
        LOGGER.info("Evaluated %s on %d original queries", name, len(dataset.dataset.queries))
        output["models"][name] = per_mode
    output["environment"] = gpu_environment(args)
    write_json(args.output_dir / "test.json", output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train", "test"))
    for key in ("original-archive", "augmented-archive", "jev-archive", "splits", "output-dir", "audit",
                "train-manifest", "model", "candidates", "split"):
        parser.add_argument(f"--{key}", type=Path)
    parser.add_argument("--base-commit")
    args = parser.parse_args()
    args.gpu_memory_limit_gib = CAP_GIB
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    {"prepare": prepare, "train": train, "test": test}[args.action](args)


if __name__ == "__main__":
    main()
