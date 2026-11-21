"""Dense-ретривер на ONNX Runtime без torch.

Модель по умолчанию — intfloat/multilingual-e5-small в ONNX-квантовке
(118 МБ, 384 измерения, 512 токенов). Эмбеддинги писем считаются один раз
и кэшируются в .cache/; в рантайме — один вектор запроса и матмуль.

Почему ONNX, а не sentence-transformers: не нужен torch (~1 ГБ), а на
CPU разница в скорости для 118M-модели небольшая.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / ".models" / "multilingual-e5-small"
CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"


class DenseUnavailable(RuntimeError):
    """Нет onnxruntime, tokenizer или файлов модели."""


class DenseIndex:
    def __init__(
        self,
        documents,
        *,
        model_dir: str | Path = MODEL_DIR,
        cache_path: str | Path | None = None,
        batch_size: int = 32,
        max_length: int = 512,
        threads: int | None = None,
        progress=None,
    ) -> None:
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as error:  # pragma: no cover - зависит от окружения
            raise DenseUnavailable(
                "нужны onnxruntime, tokenizers и numpy: "
                "python3 -m venv .venv && .venv/bin/pip install numpy tokenizers onnxruntime"
            ) from error

        model_dir = Path(model_dir)
        model_file = model_dir / "onnx" / "model_quantized.onnx"
        tokenizer_file = model_dir / "tokenizer.json"
        if not model_file.exists() or not tokenizer_file.exists():
            raise DenseUnavailable(f"нет файлов модели в {model_dir}")

        self.documents = list(documents)
        self.doc_ids = [document.doc_id for document in self.documents]
        self.max_length = max_length

        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads or 8
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_file), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_names = {item.name for item in self.session.get_inputs()}

        cache_path = Path(cache_path) if cache_path else CACHE_DIR / "dense_e5_small.npz"
        self.matrix = self._load_or_build(cache_path, batch_size, progress)

    # ------------------------------------------------------------------ encode

    def _load_or_build(self, cache_path: Path, batch_size: int, progress) -> np.ndarray:
        expected = len(self.documents)
        if cache_path.exists():
            with np.load(cache_path) as data:
                matrix = data["matrix"]
                ids = data["doc_ids"].tolist()
            if matrix.shape[0] == expected and ids == self.doc_ids:
                return matrix.astype(np.float32)

        texts = [document.text for document in self.documents]
        matrix = self._encode(texts, prefix="passage: ", batch_size=batch_size, progress=progress)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path, matrix=matrix, doc_ids=np.array(self.doc_ids, dtype=str)
        )
        return matrix

    def _encode(self, texts: list[str], *, prefix: str, batch_size: int, progress=None) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            batch = [prefix + text[:4000] for text in texts[start : start + batch_size]]
            vectors[start : start + len(batch)] = self._encode_batch(batch)
            if progress:
                progress(min(start + len(batch), len(texts)), len(texts))
        return vectors

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(texts)
        input_ids = np.array([item.ids for item in encodings], dtype=np.int64)
        attention = np.array([item.attention_mask for item in encodings], dtype=np.int64)
        feeds = {"input_ids": input_ids, "attention_mask": attention}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)

        output = self.session.run(None, feeds)[0]
        mask = attention[..., None].astype(np.float32)
        summed = (output * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        vectors = summed / counts
        norms = np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9, None)
        return (vectors / norms).astype(np.float32)

    @property
    def dimension(self) -> int:
        shape = self.session.get_outputs()[0].shape
        # [batch, seq, hidden] либо [batch, hidden]
        if isinstance(shape[-1], int):
            return shape[-1]
        return 384

    # ------------------------------------------------------------------- search

    def query_vector(self, query: str) -> np.ndarray:
        return self._encode_batch(["query: " + query])[0]

    def scores(self, query: str, candidates) -> dict[int, float]:
        if not candidates:
            return {}
        indices = np.fromiter(candidates, dtype=np.int64)
        vector = self.query_vector(query)
        values = self.matrix[indices] @ vector
        return {int(index): float(value) for index, value in zip(indices, values)}

    def search(self, query: str, candidates, top_k: int) -> list[tuple[int, float]]:
        scores = self.scores(query, candidates)
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]


class HashingDense:
    """Детерминированный fallback для тестов: хеширование токенов, без модели."""

    def __init__(self, documents, *, dimension: int = 128) -> None:
        from .text import tokenize

        self.documents = list(documents)
        self.dimension = dimension
        self.matrix = np.zeros((len(self.documents), dimension), dtype=np.float32)
        for row, document in enumerate(self.documents):
            self.matrix[row] = self._vector(document.text, tokenize)

    def _vector(self, text: str, tokenize) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in tokenize(text):
            index = hash(token) % self.dimension
            vector[index] += 1.0
        norm = np.linalg.norm(vector) or 1.0
        return vector / norm

    def scores(self, query: str, candidates) -> dict[int, float]:
        from .text import tokenize

        if not candidates:
            return {}
        vector = self._vector(query, tokenize)
        indices = list(candidates)
        values = self.matrix[indices] @ vector
        return {int(index): float(value) for index, value in zip(indices, values)}

    def search(self, query: str, candidates, top_k: int) -> list[tuple[int, float]]:
        return sorted(self.scores(query, candidates).items(), key=lambda item: (-item[1], item[0]))[:top_k]
