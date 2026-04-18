from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


class EmbeddingModel:
    """Lazy-loaded sentence-transformer wrapper."""

    _DETERMINISTIC_MODELS = {"fake", "fake-model", "deterministic", "offline-demo"}

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Any | None = None
        self._fallback_to_deterministic = False

    @property
    def model(self) -> Any:
        if self.uses_deterministic_mode:
            return None
        if self._model is None:
            try:
                self._model = self._load_transformer_model()
            except Exception as exc:
                self._fallback_to_deterministic = True
                LOGGER.warning(
                    "Falling back to deterministic local embeddings after transformer initialization failed: %s",
                    exc,
                )
                return None
        return self._model

    @property
    def uses_deterministic_mode(self) -> bool:
        return self._fallback_to_deterministic or self.model_name.strip().lower() in self._DETERMINISTIC_MODELS

    @property
    def model_version(self) -> str:
        if self.uses_deterministic_mode:
            return "deterministic-v1"
        model = self.model
        if model is None:
            return "deterministic-v1"
        version = getattr(model, "__version__", None)
        if version:
            return str(version)
        return f"{model.__class__.__module__}.{model.__class__.__name__}"

    def _load_transformer_model(self) -> Any:
        from sentence_transformers import SentenceTransformer

        try:
            return SentenceTransformer(self.model_name, local_files_only=True)
        except Exception as exc:
            self._fallback_to_deterministic = True
            LOGGER.warning(
                "Embedding model '%s' is not cached locally; falling back to deterministic local embeddings: %s",
                self.model_name,
                exc,
            )
            return None

    def _embed_deterministically(self, text: str) -> np.ndarray:
        vector = np.zeros(256, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for offset in range(0, len(digest), 4):
                bucket = digest[offset] % len(vector)
                weight = 1.0 + (digest[offset + 1] / 255.0)
                vector[bucket] += weight
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            return vector
        return vector / norm

    def embed(self, text: str) -> np.ndarray:
        normalized = text.strip()
        if not normalized:
            raise ValueError("Cannot embed empty text.")
        if self.uses_deterministic_mode:
            return self._embed_deterministically(normalized)
        model = self.model
        if model is None:
            return self._embed_deterministically(normalized)
        return np.asarray(
            model.encode(
                normalized,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        normalized = [text.strip() for text in texts]
        if any(not text for text in normalized):
            raise ValueError("Cannot embed empty text values.")
        if self.uses_deterministic_mode:
            return np.asarray([self._embed_deterministically(text) for text in normalized], dtype=np.float32)
        model = self.model
        if model is None:
            return np.asarray([self._embed_deterministically(text) for text in normalized], dtype=np.float32)
        return np.asarray(
            model.encode(
                normalized,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a_vec = np.asarray(a, dtype=np.float32)
        b_vec = np.asarray(b, dtype=np.float32)
        if a_vec.size == 0 or b_vec.size == 0:
            return 0.0
        if a_vec.shape != b_vec.shape:
            return 0.0
        a_norm = float(np.linalg.norm(a_vec))
        b_norm = float(np.linalg.norm(b_vec))
        if a_norm == 0.0 or b_norm == 0.0:
            return 0.0
        return float(np.dot(a_vec, b_vec) / (a_norm * b_norm))

    @staticmethod
    def to_bytes(embedding: np.ndarray) -> bytes:
        return np.asarray(embedding, dtype=np.float32).tobytes()

    @staticmethod
    def from_bytes(data: bytes) -> np.ndarray:
        return np.frombuffer(data, dtype=np.float32)
