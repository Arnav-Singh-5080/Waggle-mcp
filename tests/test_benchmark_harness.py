from __future__ import annotations

import json
import sqlite3

import numpy as np

from waggle.benchmark_harness import (
    _build_comparative_graph_with_sessions,
    build_markdown_summary,
    choose_best_dedup_threshold,
    load_benchmark_fixtures,
    run_benchmarks,
)


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


def test_fixture_loading_is_auditable() -> None:
    fixtures = load_benchmark_fixtures()

    assert len(fixtures["extraction_cases"]) == 25
    assert len(fixtures["retrieval_cases"]["nodes"]) == 18
    assert len(fixtures["retrieval_cases"]["queries"]) == 18
    assert len(fixtures["dedup_cases"]) == 22
    assert len(fixtures["comparative_eval"]["scenarios"]) >= 20
    assert len(fixtures["comparative_eval"]["queries"]) == 69
    comparative_gold_ids = {
        support_id
        for query in fixtures["comparative_eval"]["queries"]
        for support_id in query["gold_support_ids"]
    }
    assert {"ctx_postgres_decision", "ctx_acid_reason", "ctx_sqlite_old", "ctx_postgres_new", "ctx_concurrency_reason", "ctx_fastapi_decision", "ctx_realtime_req", "ctx_concurrency_fact"} <= comparative_gold_ids
    assert len(fixtures["query_stress_cases"]["queries"]) >= 40
    assert sum(1 for case in fixtures["query_stress_cases"]["queries"] if case["task_family"] == "adversarial_paraphrase") >= 20
    assert sum(1 for case in fixtures["query_stress_cases"]["queries"] if case["task_family"] == "temporal_latest") >= 20
    assert any(not case["should_dedup"] for case in fixtures["dedup_cases"])
    assert any(case["should_dedup"] for case in fixtures["dedup_cases"])


def test_benchmark_report_includes_backend_labels_and_case_counts() -> None:
    report = run_benchmarks(
        extraction_backend="regex",
        embedding_model=FakeEmbeddingModel(),
        systems=["waggle", "rag_naive", "rag_tuned"],
    )

    extraction = next(metric for metric in report.metrics if metric.metric == "extraction")
    retrieval = next(metric for metric in report.metrics if metric.metric == "retrieval")
    dedup = next(metric for metric in report.metrics if metric.metric == "deduplication")

    assert extraction.backend == "regex"
    assert extraction.case_count == 25
    assert retrieval.backend == "semantic-query"
    assert retrieval.case_count == 18
    assert "corpus_nodes" in retrieval.metadata
    assert dedup.backend == "semantic-dedup"
    assert dedup.case_count == 22
    assert "threshold" in dedup.metadata
    assert report.comparative["corpus"]["scenario_count"] >= 20
    assert report.comparative["corpus"]["query_count"] == 69
    assert report.fixtures["query_stress_cases"] >= 40
    assert set(report.stress_eval["systems"]) == {"graph_raw", "graph_hybrid"}
    assert set(report.comparative["systems"]) == {"waggle", "rag_naive", "rag_tuned"}
    assert len(report.comparative["per_case"]) == 207
    assert set(report.comparative["systems"]["waggle"]["by_retrieval_mode"]) == {"flat", "graph"}
    assert set(report.comparative["systems"]["waggle"]["query_policy"]) == {"flat", "graph"}


def test_markdown_summary_includes_comparative_systems() -> None:
    report = run_benchmarks(
        extraction_backend="regex",
        embedding_model=FakeEmbeddingModel(),
        systems=["waggle", "rag_naive"],
    )

    markdown = build_markdown_summary(report)

    assert "# Waggle Comparative Evaluation" in markdown
    assert "| waggle |" in markdown
    assert "| rag_naive |" in markdown
    assert "## Waggle Query Policy" in markdown
    assert "## Waggle Mode Breakdown" in markdown
    assert "Failure Protocol" in markdown
    assert "## Query Stress Eval" in markdown


def test_comparative_graph_builder_ingests_sessions_and_temporal_edges() -> None:
    fixtures = load_benchmark_fixtures()

    graph = _build_comparative_graph_with_sessions(fixtures["comparative_eval"], embedding_model=FakeEmbeddingModel())
    with sqlite3.connect(graph.db_path) as connection:
        connection.row_factory = sqlite3.Row
        transcript_count = int(connection.execute("SELECT COUNT(*) FROM transcript_records").fetchone()[0])
        assert transcript_count > 0

        rows = connection.execute(
            "SELECT id, tags FROM nodes WHERE tags LIKE ? OR tags LIKE ?",
            ("%fact_id:db_sqlite_local%", "%fact_id:db_postgres_prod%"),
        ).fetchall()

        node_ids_by_fact: dict[str, str] = {}
        for row in rows:
            tags = json.loads(row["tags"])
            for tag in tags:
                if tag.startswith("fact_id:"):
                    node_ids_by_fact[tag.split(":", 1)[1]] = row["id"]

        older_id = node_ids_by_fact["db_sqlite_local"]
        newer_id = node_ids_by_fact["db_postgres_prod"]
        edge_rows = connection.execute(
            """
            SELECT relationship
            FROM edges
            WHERE source_id = ? AND target_id = ?
            ORDER BY relationship
            """,
            (newer_id, older_id),
        ).fetchall()
        relationships = {row["relationship"] for row in edge_rows}
        assert "updates" in relationships
        assert "contradicts" in relationships
def test_benchmark_report_runs_regex_only_extraction() -> None:
    report = run_benchmarks(extraction_backend="regex", embedding_model=FakeEmbeddingModel())

    extraction_metrics = [metric for metric in report.metrics if metric.metric == "extraction"]
    assert len(extraction_metrics) == 1
    assert extraction_metrics[0].backend == "regex"
    assert extraction_metrics[0].metadata["runtime"] == "deterministic-regex"


def test_dedup_threshold_sweep_tracks_positive_and_negative_cases() -> None:
    fixtures = load_benchmark_fixtures()

    best, sweep = choose_best_dedup_threshold(
        fixtures["dedup_cases"],
        embedding_model=FakeEmbeddingModel(),
    )

    assert sweep
    assert best.metadata["positive_cases"] == 11
    assert best.metadata["negative_cases"] == 11
    assert best.metadata["true_negatives"] + best.metadata["false_positives"] == 11
    assert best.metadata["true_positives"] + best.metadata["false_negatives"] == 11
