from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from waggle.oolong_benchmark import answers_match, evaluate_oolong, load_oolong_examples


class FakeEmbeddingModel:
    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(1024, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vector[sum(ord(character) for character in token) % len(vector)] += 1.0
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


def test_load_oolong_examples_normalizes_list_answers(tmp_path: Path) -> None:
    dataset_path = tmp_path / "oolong.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "row-1",
                "question": "Which topping was selected?",
                "answer": "['ham']",
                "context_window_text": "Order summary\nSelected topping: ham",
                "task_group": "single_hop",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples = load_oolong_examples(dataset_path)

    assert len(examples) == 1
    assert examples[0].dataset_kind == "synth"
    assert examples[0].answer == "ham"


def test_evaluate_oolong_retrieval_only_reports_context_usage(tmp_path: Path) -> None:
    dataset = [
        {
            "id": "real-1",
            "context_window_id": "ctx-1",
            "question": "Which drink does Bob like?",
            "answer": "coffee",
            "question_type": "single_hop",
            "context_window_text": "\n".join(
                [
                    "[START OF EPISODE]",
                    "Alice likes tea.",
                    "[END OF EPISODE]",
                    "[START OF EPISODE]",
                    "Bob likes coffee.",
                    "[END OF EPISODE]",
                ]
            ),
        }
    ]
    dataset_path = tmp_path / "oolong-real.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    report = evaluate_oolong(
        dataset_path,
        embedding_model=FakeEmbeddingModel(),
        eval_mode="retrieval_only",
    )

    assert report.case_count == 1
    assert report.scored_case_count == 0
    assert report.accuracy is None
    assert report.per_case[0].retrieved_node_count >= 1
    assert report.per_case[0].retrieved_tokens > 0


def test_evaluate_oolong_with_llm_answerer_scores_predictions(tmp_path: Path) -> None:
    dataset = [
        {
            "id": "real-1",
            "context_window_id": "ctx-1",
            "question": "Which drink does Bob like? Return only the drink.",
            "answer": "coffee",
            "question_type": "single_hop",
            "context_window_text": "Bob likes coffee.",
        }
    ]
    dataset_path = tmp_path / "oolong-real.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    def fake_llm(prompt: str) -> str:
        return "coffee" if "Bob likes coffee" in prompt else "tea"

    report = evaluate_oolong(
        dataset_path,
        embedding_model=FakeEmbeddingModel(),
        eval_mode="waggle_llm",
        llm_answerer=fake_llm,
    )

    assert report.case_count == 1
    assert report.scored_case_count == 1
    assert report.accuracy == 1.0
    assert report.per_case[0].predicted_answer == "coffee"
    assert report.per_case[0].correct is True


def test_answers_match_handles_literal_lists() -> None:
    assert answers_match("ham", "['ham']")
    assert answers_match("42", "42")
    assert not answers_match("tea", "coffee")
