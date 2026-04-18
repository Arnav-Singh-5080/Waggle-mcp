from __future__ import annotations

import numpy as np
import pytest

from waggle.embeddings import EmbeddingModel


def test_embedding_bytes_round_trip() -> None:
    vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    encoded = EmbeddingModel.to_bytes(vector)
    decoded = EmbeddingModel.from_bytes(encoded)
    assert np.allclose(decoded, vector)


def test_cosine_similarity_handles_orthogonal_vectors() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert EmbeddingModel.cosine_similarity(a, b) == 0.0


def test_cosine_similarity_returns_zero_for_shape_mismatch() -> None:
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0], dtype=np.float32)
    assert EmbeddingModel.cosine_similarity(a, b) == 0.0


def test_fake_model_is_deterministic_and_normalized() -> None:
    model = EmbeddingModel("fake-model")
    other_model = EmbeddingModel("fake-model")
    a = model.embed("PostgreSQL over MySQL")
    b = other_model.embed("PostgreSQL over MySQL")
    c = model.embed("Dark mode UI")

    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a), 1.0)
    assert EmbeddingModel.cosine_similarity(a, c) < 1.0


def test_uncached_transformer_falls_back_to_deterministic_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    def uncached(_: EmbeddingModel) -> None:
        raise OSError("model cache missing")

    monkeypatch.setattr(EmbeddingModel, "_load_transformer_model", uncached)

    model = EmbeddingModel("all-MiniLM-L6-v2")
    vector = model.embed("Backend uses FastAPI")

    assert model.uses_deterministic_mode is True
    assert vector.shape == (256,)
    assert np.isclose(np.linalg.norm(vector), 1.0)
