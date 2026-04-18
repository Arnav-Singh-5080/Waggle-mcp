from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from waggle.longmemeval_benchmark import evaluate_longmemeval


class FakeEmbeddingModel:
    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(8, dtype=np.float32)
        for token in text.lower().split():
            index = sum(ord(character) for character in token) % len(vector)
            vector[index] += 1.0
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            return vector
        return vector / norm

    def to_bytes(self, embedding: np.ndarray) -> bytes:
        return embedding.astype(np.float32).tobytes()

    def from_bytes(self, data: bytes) -> np.ndarray:
        return np.frombuffer(data, dtype=np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0.0 or b_norm == 0.0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))


class CountingEmbeddingModel(FakeEmbeddingModel):
    def __init__(self) -> None:
        self.embedded_text_count = 0

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        self.embedded_text_count += len(texts)
        return np.asarray([self.embed(text) for text in texts], dtype=np.float32)


def test_evaluate_longmemeval_graph_modes(tmp_path: Path) -> None:
    dataset = [
        {
            "id": "entry_1",
            "question": "what database are we using in production",
            "haystack_sessions": [
                [
                    {"role": "user", "content": "We are using SQLite locally."},
                    {"role": "assistant", "content": "SQLite sounds fine for local work."},
                ],
                [
                    {"role": "user", "content": "Production uses PostgreSQL for safer migrations."},
                    {"role": "assistant", "content": "PostgreSQL is the production choice."},
                ],
            ],
            "haystack_session_ids": ["sess_local", "sess_prod"],
            "haystack_dates": ["2024/01/05 (Fri) 09:00", "2024/02/10 (Sat) 09:00"],
            "correct_session_ids": ["sess_prod"],
        }
    ]
    dataset_path = tmp_path / "longmemeval.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    raw_report = evaluate_longmemeval(dataset_path, embedding_model=FakeEmbeddingModel(), mode="graph_raw")
    hybrid_report = evaluate_longmemeval(dataset_path, embedding_model=FakeEmbeddingModel(), mode="graph_hybrid")

    assert raw_report.case_count == 1
    assert hybrid_report.case_count == 1
    assert raw_report.r_at_5 == 1.0
    assert hybrid_report.r_at_5 == 1.0
    assert raw_report.per_case[0].retrieved_session_ids


def test_evaluate_longmemeval_caches_repeated_session_embeddings(tmp_path: Path) -> None:
    dataset = [
        {
            "id": "entry_1",
            "question": "what database are we using in production",
            "haystack_sessions": [
                [{"role": "user", "content": "Production uses PostgreSQL for safer migrations."}],
                [{"role": "user", "content": "Local development uses SQLite."}],
            ],
            "haystack_session_ids": ["sess_prod", "sess_local"],
            "haystack_dates": ["2024/02/10 (Sat) 09:00", "2024/01/05 (Fri) 09:00"],
            "correct_session_ids": ["sess_prod"],
        },
        {
            "id": "entry_2",
            "question": "what database do we use locally",
            "haystack_sessions": [
                [{"role": "user", "content": "Production uses PostgreSQL for safer migrations."}],
                [{"role": "user", "content": "Feature flags live in Redis."}],
            ],
            "haystack_session_ids": ["sess_prod_repeat", "sess_flags"],
            "haystack_dates": ["2024/02/10 (Sat) 09:00", "2024/03/01 (Fri) 09:00"],
            "correct_session_ids": ["sess_prod_repeat"],
        },
    ]
    dataset_path = tmp_path / "longmemeval.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    embedding_model = CountingEmbeddingModel()
    report = evaluate_longmemeval(dataset_path, embedding_model=embedding_model, mode="graph_raw")

    assert report.case_count == 2
    assert embedding_model.embedded_text_count == 5


def test_evaluate_longmemeval_reuses_disk_cache(tmp_path: Path) -> None:
    dataset = [
        {
            "id": "entry_1",
            "question": "what database are we using in production",
            "haystack_sessions": [
                [{"role": "user", "content": "Production uses PostgreSQL for safer migrations."}],
                [{"role": "user", "content": "Local development uses SQLite."}],
            ],
            "haystack_session_ids": ["sess_prod", "sess_local"],
            "haystack_dates": ["2024/02/10 (Sat) 09:00", "2024/01/05 (Fri) 09:00"],
            "correct_session_ids": ["sess_prod"],
        }
    ]
    dataset_path = tmp_path / "longmemeval.json"
    cache_dir = tmp_path / "cache"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    first_model = CountingEmbeddingModel()
    second_model = CountingEmbeddingModel()

    first_report = evaluate_longmemeval(
        dataset_path,
        embedding_model=first_model,
        mode="graph_raw",
        cache_dir=cache_dir,
    )
    second_report = evaluate_longmemeval(
        dataset_path,
        embedding_model=second_model,
        mode="graph_raw",
        cache_dir=cache_dir,
    )

    assert first_report.case_count == 1
    assert second_report.case_count == 1
    assert first_model.embedded_text_count == 3
    assert second_model.embedded_text_count == 0
    assert Path(first_report.cache_path).suffix == ".json"
    assert Path(first_report.cache_path).exists()
    assert Path(first_report.cache_path).with_suffix(".npz").exists()
    assert not Path(first_report.cache_path).with_suffix(".pkl").exists()
