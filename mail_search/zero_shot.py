import argparse
import hashlib
import math
import os
import platform
from importlib.metadata import version
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from mail_search.benchmark import source_hashes
from mail_search.dataset import load_archive, read_split
from mail_search.reranker import evaluate, write_metrics
from mail_search.reranker_data import load_candidates

QWEN_INSTRUCTION = (
    "Given an email search query, rank messages that match the event or message requested. "
    "The mailbox and received date have already been filtered."
)


class RuBertScorer:
    def __init__(self, path: Path, max_length: int) -> None:
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, local_files_only=True,
        ).to("cuda").eval()

    @torch.inference_mode()
    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        inputs = self.tokenizer(
            [query] * len(documents), documents, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = self.model(**inputs).logits.squeeze(-1).float().cpu().tolist()
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("RuBERT produced a nonfinite score")
        return scores


class QwenScorer:
    def __init__(self, path: Path, max_length: int, dtype: torch.dtype = torch.bfloat16) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, padding_side="left", local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, dtype=dtype,
        ).to("cuda").eval()
        prefix = (
            '<|im_start|>system\nJudge whether the Document meets the requirements '
            'based on the Query and the Instruct provided. Note that the answer '
            'can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix = self.tokenizer.encode(suffix, add_special_tokens=False)
        self.input_limit = max_length - len(self.prefix) - len(self.suffix)
        if self.input_limit <= 0:
            raise ValueError("Context limit is too short for the Qwen scoring prompt")
        self.yes = self.tokenizer.convert_tokens_to_ids("yes")
        self.no = self.tokenizer.convert_tokens_to_ids("no")

    def logits(self, queries: list[str], documents: list[str]) -> torch.Tensor:
        pairs = [
            f"<Instruct>: {QWEN_INSTRUCTION}\n<Query>: {query}\n<Document>: {document}"
            for query, document in zip(queries, documents, strict=True)
        ]
        encoded = self.tokenizer(
            pairs, add_special_tokens=False, truncation=True, max_length=self.input_limit,
            return_attention_mask=False,
        )["input_ids"]
        inputs = self.tokenizer.pad(
            {"input_ids": [self.prefix + ids + self.suffix for ids in encoded]},
            padding=True, return_attention_mask=True, return_tensors="pt",
        ).to("cuda")
        logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]
        return logits[:, self.yes].float() - logits[:, self.no].float()

    @torch.inference_mode()
    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        scores = self.logits([query] * len(documents), documents).cpu().tolist()
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("Qwen produced a nonfinite score")
        return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare frozen cross-encoders on fixed E5 candidates.")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--e5-candidates", type=Path, required=True)
    parser.add_argument("--name", choices=("qwen", "rubert"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(2)
    dataset = load_archive(args.archive)
    users, split = read_split(args.splits, dataset, "val")
    candidates, e5 = load_candidates(args.e5_candidates, dataset, args.splits, users, "val")
    top_k = 20
    max_length = 256
    model = QwenScorer(args.model, max_length) if args.name == "qwen" else RuBertScorer(
        args.model, max_length,
    )
    result = {
        "environment": {
            "python": platform.python_version(), "host": platform.node(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "transformers": version("transformers"),
            "tokenizers": version("tokenizers"), "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "code_base_commit": args.base_commit, "source_sha256": source_hashes(),
        },
        "config": {
            "archive_sha256": dataset.archive_sha256, "split": "val", "split_seed": split["seed"],
            "e5": e5, "e5_candidates_sha256": hashlib.sha256(args.e5_candidates.read_bytes()).hexdigest(),
            "name": args.name, "revision": args.revision, "top_k": top_k, "max_length": max_length,
            "instruction": QWEN_INSTRUCTION if args.name == "qwen" else None,
        },
        **evaluate(dataset, candidates, model, top_k),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_metrics(args.output, result)


if __name__ == "__main__":
    main()
