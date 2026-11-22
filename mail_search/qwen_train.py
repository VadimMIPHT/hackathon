import argparse
import json
import logging
import math
import os
import platform
import random
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter, sleep

import torch

from mail_search.benchmark import measure, source_hashes, summarize
from mail_search.dataset import Dataset, load_archive, read_split
from mail_search.depth_benchmark import configure_cuda
from mail_search.final_test import sha256
from mail_search.reranker import PairFineTuner, write_metrics
from mail_search.reranker_data import CandidateQuery, load_candidates, training_pairs
from mail_search.retrieval import searchable_text
from mail_search.zero_shot import QWEN_INSTRUCTION, QwenScorer

LOGGER = logging.getLogger(__name__)
BASE_WEIGHTS_SHA256 = "27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b"
BATCH_SIZE = 20


class TrainableQwen(QwenScorer, PairFineTuner):
    def __init__(
        self, model_path: Path, revision: str, max_length: int = 256, train_layers: int = 2,
    ) -> None:
        super().__init__(model_path, max_length, dtype=torch.float32)
        if not 0 < train_layers <= len(self.model.model.layers):
            raise ValueError("Invalid number of unfrozen layers")
        self.model.requires_grad_(False)
        self.train_modules = list(self.model.model.layers[-train_layers:])
        for module in self.train_modules:
            module.requires_grad_(True)
        self.weights = {
            name: weight for name, weight in self.model.named_parameters() if weight.requires_grad
        }
        self.checkpoint_metadata = {
            "base_model": "Qwen/Qwen3-Reranker-0.6B", "base_revision": revision,
            "train_layers": str(train_layers), "max_length": str(max_length),
            "head": "frozen_yes_minus_no",
        }

    def logits(self, queries: list[str], documents: list[str]) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return super().logits(queries, documents)


def evaluate(
    dataset: Dataset, candidates: list[CandidateQuery], model: TrainableQwen, batch_pause_ms: float,
) -> dict:
    model.model.eval()
    model.score("прогрев поиска", ["прогрев оценки релевантности"] * BATCH_SIZE)
    by_depth: dict[str, list[dict]] = {"20": [], "50": []}
    for candidate in candidates:
        query = candidate.query
        started = perf_counter()
        baseline = candidate.ranked_doc_ids
        scores = []
        for start in range(0, len(baseline), BATCH_SIZE):
            documents = [searchable_text(dataset.documents[did])
                         for did in baseline[start:start+BATCH_SIZE]]
            scores.extend(model.score(query.text, documents))
            if batch_pause_ms:
                sleep(batch_pause_ms / 1000)
        logits = dict(zip(baseline, scores, strict=True))
        elapsed = (perf_counter() - started) * 1000
        for depth in (20, 50):
            if depth == 20:
                head = sorted(baseline[:depth], key=lambda did: -logits[did])
            else:
                head = sorted(baseline[:depth], key=lambda did: (-logits[did], did))
            ranked = head + baseline[depth:]
            by_depth[str(depth)].append({
                "query_id": query.query_id, "mailbox": query.mailbox,
                "ranked_doc_ids": ranked, "e5_order_logits": scores,
                "elapsed_ms": elapsed,
                "metrics": measure(ranked, dataset.qrels[query.query_id]),
            })
    return {
        depth: {"summary": summarize(rows), "per_query": rows}
        for depth, rows in by_depth.items()
    }


def training_reference(args: argparse.Namespace, dataset: Dataset, manifest: dict) -> tuple[dict, dict]:
    bge = json.loads(args.bge_reference.read_text())
    config = bge["config"]
    expected = {
        "archive_sha256": dataset.archive_sha256,
        "split_manifest_sha256": sha256(args.splits),
        "split_seed": manifest["seed"],
        "extra_judgements_sha256": sha256(args.extra_judgements),
        "e5_candidates_sha256": sha256(args.e5_candidates),
        "eval_split": "val",
        "top_k": 20,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Training or validation differs from the BGE reference: {key}")
    if config["optimizer"]["name"] != "AdamW" or config["loss"] != "BCE with logits, equal total weight per query":
        raise ValueError("Unexpected BGE training objective")
    return config, bge["train"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen reranker on the exact BGE train.")
    for name in ("archive", "splits", "extra-judgements", "bge-reference", "e5-candidates",
                 "model", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--gpu-memory-limit-gib", type=float, default=8)
    parser.add_argument("--batch-pause-ms", type=float, default=0)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory must be new")
    if args.batch_pause_ms < 0 or args.gpu_memory_limit_gib <= 0:
        parser.error("Invalid GPU memory limit or batch pause")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    dataset = load_archive(args.archive)
    train_users, manifest = read_split(args.splits, dataset, "train")
    val_users = set(manifest["splits"]["val"]["mailboxes"])
    config, reference_counts = training_reference(args, dataset, manifest)
    candidates, _ = load_candidates(args.e5_candidates, dataset, args.splits, val_users, "val")
    pairs = training_pairs(dataset, train_users, args.extra_judgements)
    if len(pairs) != reference_counts["pairs"] or sum(pair.label for pair in pairs) != reference_counts["positives"]:
        raise ValueError("Train pair counts differ from BGE")
    if sha256(args.model / "model.safetensors") != BASE_WEIGHTS_SHA256:
        raise ValueError("Qwen base weights differ from the expected revision")

    configure_cuda(args.gpu_memory_limit_gib)
    torch.manual_seed(config["seed"])
    rng = random.Random(config["seed"])
    model = TrainableQwen(args.model, args.revision, config["max_length"], config["train_layers"])
    optimizer = torch.optim.AdamW(
        model.weights.values(), lr=config["optimizer"]["learning_rate"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "metrics.json"
    result = {
        "environment": {
            "host": platform.node(), "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "transformers": version("transformers"),
            "tokenizers": version("tokenizers"), "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "code_base_commit": args.base_commit, "source_sha256": source_hashes(),
            "utc_started": datetime.now(timezone.utc).isoformat(),
        },
        "config": {
            **{key: config[key] for key in ("archive_sha256", "split_manifest_sha256", "split_seed",
               "extra_judgements_sha256", "e5_candidates_sha256", "eval_split", "top_k",
               "max_length", "seed", "train_layers", "epochs", "batch_size", "optimizer", "loss")},
            "bge_reference_sha256": sha256(args.bge_reference),
            "qwen_revision": args.revision, "base_weights_sha256": BASE_WEIGHTS_SHA256,
            "instruction": QWEN_INSTRUCTION, "autocast": "bf16", "parameter_dtype": "fp32",
            "head": "frozen yes-minus-no logit", "batch_pause_ms": args.batch_pause_ms,
            "gpu_memory_limit_gib": args.gpu_memory_limit_gib,
        },
        "train": {
            "queries": reference_counts["queries"], "pairs": len(pairs),
            "positives": sum(pair.label for pair in pairs),
            "trainable_parameters": sum(weight.numel() for weight in model.weights.values()),
        },
        "qwen_pretrained": evaluate(dataset, candidates, model, args.batch_pause_ms),
        "epochs": [],
    }
    LOGGER.info("Qwen base nDCG@10 on val top-20 %.5f",
                result["qwen_pretrained"]["20"]["summary"]["ndcg@10"])
    write_metrics(output, result)
    best_score = -math.inf
    checkpoint = args.output_dir / "best-finetuned.safetensors"
    for epoch in range(1, config["epochs"] + 1):
        started = perf_counter()
        loss = model.fit_epoch(pairs, optimizer, config["batch_size"], rng, args.batch_pause_ms)
        elapsed = perf_counter() - started
        evaluation = evaluate(dataset, candidates, model, args.batch_pause_ms)
        ndcg = evaluation["20"]["summary"]["ndcg@10"]
        result["epochs"].append({"epoch": epoch, "loss": loss, "train_seconds": elapsed,
                                 "depths": evaluation})
        if ndcg > best_score:
            best_score = ndcg
            model.save_weights(checkpoint)
            result["best_epoch"] = epoch
            result["checkpoint_sha256"] = sha256(checkpoint)
        result["environment"]["peak_reserved_gib"] = torch.cuda.max_memory_reserved(0) / (1 << 30)
        write_metrics(output, result)
        LOGGER.info("Qwen epoch %d, BCE %.5f, val nDCG@10 %.5f, %.1fs", epoch, loss, ndcg, elapsed)


if __name__ == "__main__":
    main()
