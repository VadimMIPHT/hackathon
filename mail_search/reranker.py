import argparse
import hashlib
import json
import logging
import math
import os
import platform
import random
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Protocol

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from mail_search.benchmark import measure, source_hashes, summarize
from mail_search.dataset import Dataset, load_archive, read_split
from mail_search.reranker_data import (
    CandidateQuery, TrainingPair, load_candidates, load_training_groups, training_pairs,
)
from mail_search.retrieval import searchable_text

LOGGER = logging.getLogger(__name__)


class PairScorer(Protocol):
    model: torch.nn.Module

    def score(self, query: str, documents: list[str]) -> list[float]: ...


class Reranker:
    def __init__(self, model_path: Path, revision: str, max_length: int, train_layers: int) -> None:
        self.max_length = max_length
        self.checkpoint_metadata = {"base_revision": revision, "train_layers": str(train_layers)}
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True,
        ).to("cuda")
        self.model.requires_grad_(False)
        self.train_modules = [*self.model.roberta.encoder.layer[-train_layers:], self.model.classifier]
        for module in self.train_modules:
            module.requires_grad_(True)
        self.weights = {
            name: parameter for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }

    def logits(self, queries: list[str], documents: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            queries, documents, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(**inputs).logits.squeeze(-1).float()

    @torch.inference_mode()
    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        scores = self.logits([query] * len(documents), documents).cpu().tolist()
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("BGE produced a nonfinite score")
        return scores

    def fit_epoch(
        self, pairs: list[TrainingPair], optimizer: torch.optim.Optimizer,
        batch_size: int, rng: random.Random,
    ) -> float:
        self.model.eval()
        for module in self.train_modules:
            module.train()
        order = list(range(len(pairs)))
        rng.shuffle(order)
        loss_sum = torch.zeros((), device="cuda")
        for start in range(0, len(order), batch_size):
            batch = [pairs[index] for index in order[start:start + batch_size]]
            logits = self.logits([x.query for x in batch], [x.document for x in batch])
            labels = torch.tensor([x.label for x in batch], dtype=torch.float32, device="cuda")
            weights = torch.tensor([x.weight for x in batch], dtype=torch.float32, device="cuda")
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels, weight=weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.weights.values(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.detach() * len(batch)
        return loss_sum.item() / len(pairs)

    def save_weights(self, path: Path) -> None:
        temporary = path.with_suffix(".partial")
        save_file(
            {name: weight.detach().cpu() for name, weight in self.weights.items()},
            temporary, metadata=self.checkpoint_metadata,
        )
        temporary.replace(path)

    def load_weights(self, path: Path) -> None:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            if checkpoint.metadata() != self.checkpoint_metadata:
                raise ValueError("Checkpoint was saved for a different BGE revision or layer count")
            if set(checkpoint.keys()) != self.weights.keys():
                raise ValueError("Checkpoint parameters do not match the selected trainable layers")
            saved = {name: checkpoint.get_tensor(name) for name in checkpoint.keys()}
        self.model.load_state_dict(saved, strict=False)


def evaluate(
    dataset: Dataset, candidates: list[CandidateQuery], reranker: PairScorer | None, top_k: int,
) -> dict:
    if reranker is not None:
        reranker.model.eval()
        reranker.score("прогрев поиска", ["прогрев оценки релевантности"] * top_k)

    rows = []
    for candidate in candidates:
        query = candidate.query
        ranking = candidate.ranked_doc_ids
        scores = []
        if reranker is None:
            elapsed = candidate.retrieval_ms
        else:
            head, tail = ranking[:top_k], ranking[top_k:]
            started = perf_counter()
            scores = reranker.score(
                query.text, [searchable_text(dataset.documents[doc_id]) for doc_id in head],
            )
            reordered = sorted(range(len(head)), key=lambda index: (-scores[index], index))
            ranking = [head[index] for index in reordered] + tail
            elapsed = (perf_counter() - started) * 1000
        rows.append({
            "query_id": query.query_id,
            "mailbox": query.mailbox,
            "ranked_doc_ids": ranking,
            "in_e5_order_logits": scores,
            "elapsed_ms": elapsed,
            "metrics": measure(ranking, dataset.qrels[query.query_id]),
        })

    return {
        "latency_scope": "cached_e5_retrieval" if reranker is None else "rerank_stage_only",
        "summary": summarize(rows),
        "by_mailbox": {
            mailbox: summarize([row for row in rows if row["mailbox"] == mailbox])
            for mailbox in sorted({row["mailbox"] for row in rows})
        },
        "per_query": rows,
    }


def run(args: argparse.Namespace) -> None:
    dataset = load_archive(args.archive)
    train_users, split_manifest = read_split(args.splits, dataset, "train")
    eval_users = set(split_manifest["splits"][args.eval_split]["mailboxes"])
    candidates, retrieval_config = load_candidates(
        args.e5_candidates, dataset, args.splits, eval_users, args.eval_split,
    )
    if args.epochs and args.eval_split != "val":
        raise ValueError("Fine-tuning can only use val for model selection")

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    result = {
        "environment": {
            "utc_started": datetime.now(timezone.utc).isoformat(),
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": version("transformers"),
            "tokenizers": version("tokenizers"),
            "safetensors": version("safetensors"),
            "cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": torch.cuda.get_device_name(0),
            "code_base_commit": args.base_commit,
            "source_sha256": source_hashes(),
        },
        "config": {
            "archive_sha256": dataset.archive_sha256,
            "split_seed": split_manifest["seed"],
            "split_manifest_sha256": hashlib.sha256(args.splits.read_bytes()).hexdigest(),
            "eval_split": args.eval_split,
            "e5_candidates_sha256": hashlib.sha256(args.e5_candidates.read_bytes()).hexdigest(),
            "e5": retrieval_config,
            "top_k": args.top_k,
            "bge_repo": "BAAI/bge-reranker-v2-m3",
            "bge_revision": args.bge_revision,
            "max_length": args.max_length,
            "autocast_dtype": "bf16",
            "seed": args.seed,
            "train_layers": args.train_layers,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": {"name": "AdamW", "learning_rate": args.learning_rate, "weight_decay": 0.01},
            "loss": "BCE with logits, equal total weight per query",
        },
        "e5": evaluate(dataset, candidates, None, args.top_k),
        "epochs": [],
    }

    reranker = Reranker(args.model, args.bge_revision, args.max_length, args.train_layers)
    result["bge_pretrained"] = evaluate(dataset, candidates, reranker, args.top_k)
    LOGGER.info("Pretrained BGE %s nDCG@10 %.5f", args.eval_split,
                result["bge_pretrained"]["summary"]["ndcg@10"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = args.output_dir / "metrics.json"
    write_metrics(metrics_path, result)

    if args.weights:
        reranker.load_weights(args.weights)
        result["weights_evaluation"] = evaluate(dataset, candidates, reranker, args.top_k)
        result["config"]["checkpoint_sha256"] = hashlib.sha256(args.weights.read_bytes()).hexdigest()
        write_metrics(metrics_path, result)
    if not args.epochs:
        return

    pairs = training_pairs(dataset, train_users, args.extra_judgements)
    extra_queries = 0
    if args.extra_judgements:
        result["config"]["extra_judgements_sha256"] = hashlib.sha256(
            args.extra_judgements.read_bytes()
        ).hexdigest()
        extra_queries = len(load_training_groups(args.extra_judgements, dataset, train_users))
    result["train"] = {
        "queries": sum(query.mailbox in train_users for query in dataset.queries.values()) + extra_queries,
        "extra_queries": extra_queries,
        "pairs": len(pairs),
        "positives": sum(pair.label for pair in pairs),
        "trainable_parameters": sum(weight.numel() for weight in reranker.weights.values()),
    }
    optimizer = torch.optim.AdamW(reranker.weights.values(), lr=args.learning_rate)
    best_metric = -math.inf
    for epoch in range(1, args.epochs + 1):
        start = perf_counter()
        loss = reranker.fit_epoch(pairs, optimizer, args.batch_size, rng)
        elapsed = perf_counter() - start
        evaluation = evaluate(dataset, candidates, reranker, args.top_k)
        metric = evaluation["summary"]["ndcg@10"]
        record = {"epoch": epoch, "loss": loss, "train_seconds": elapsed, **evaluation}
        result["epochs"].append(record)
        if metric > best_metric:
            best_metric = metric
            result["best_finetuned_epoch"] = epoch
            reranker.save_weights(args.output_dir / "best-finetuned.safetensors")
        write_metrics(metrics_path, result)
        LOGGER.info("Epoch %d completed in %.1fs. Loss %.5f, val nDCG@10 %.5f", epoch, elapsed, loss, metric)


def write_metrics(path: Path, metrics: dict) -> None:
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and fine-tune BGE on VK mail relevance pairs.")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--e5-candidates", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bge-revision", required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-split", choices=("val", "test"), default="val")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--extra-judgements", type=Path)
    parser.add_argument("--top-k", type=int, choices=range(10, 51), default=20, metavar="[10-50]")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-layers", type=int, choices=range(1, 25), default=2, metavar="[1-24]")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (args.epochs < 0 or args.batch_size <= 0 or args.max_length < 16
            or not 0 < args.learning_rate < 1):
        parser.error("Invalid training configuration")
    if args.weights and args.epochs:
        parser.error("Use --epochs 0 when evaluating a saved checkpoint")
    if args.extra_judgements and not args.epochs:
        parser.error("Additional judgements require training")
    if args.output_dir.exists():
        parser.error("--output-dir must be new")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
