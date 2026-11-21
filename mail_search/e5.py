from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


MAX_LENGTH = 256
DOCUMENT_BATCH_SIZE = 32


class E5:
    def __init__(self, model_path: Path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).to(
            device="cuda", dtype=torch.float16
        ).eval()

    @torch.inference_mode()
    def encode(self, texts: list[str], prefix: str) -> np.ndarray:
        batches = []
        for start in range(0, len(texts), DOCUMENT_BATCH_SIZE):
            batch = self.tokenizer(
                [prefix + text for text in texts[start:start + DOCUMENT_BATCH_SIZE]],
                max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt",
            ).to("cuda")
            encoded = self.model(**batch).last_hidden_state.float()
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1)
            batches.append(torch.nn.functional.normalize(pooled).cpu().numpy())
        return np.concatenate(batches, axis=0)


def rank(
    matrix: np.ndarray, vector: np.ndarray, indices: list[int], doc_ids: list[str], limit: int,
) -> list[str]:
    if not indices:
        return []
    scores = matrix[indices] @ vector
    order = sorted(zip(indices, scores), key=lambda item: (-item[1], doc_ids[item[0]]))[:limit]
    return [doc_ids[i] for i, _ in order]
