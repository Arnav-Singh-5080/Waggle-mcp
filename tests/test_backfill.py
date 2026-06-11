from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from waggle.backfill import backfill_context_windows
from waggle.graph import MemoryGraph
from waggle.models import NodeType


class FakeEmbeddingModel:
    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(8, dtype=np.float32)
        for token in text.lower().split():
            vector[sum(ord(character) for character in token) % len(vector)] += 1.0
        norm = np.linalg.norm(vector)
        return vector if norm == 0.0 else vector / norm

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


def make_graph(tmp_path: Path) -> MemoryGraph:
    return MemoryGraph(tmp_path / "memory.db", FakeEmbeddingModel(), dedup_similarity_threshold=1.1)


def _make_legacy_nodes(graph: MemoryGraph) -> list[str]:
    first = graph.add_node(
        label="Dog",
        content="Dog is named X",
        node_type=NodeType.ENTITY,
        project="alpha",
        session_id="sess-1",
    ).node
    second = graph.add_node(
        label="Dog",
        content="Dog is named Y",
        node_type=NodeType.ENTITY,
        project="alpha",
        session_id="sess-2",
    ).node
    with sqlite3.connect(graph.db_path) as connection:
        connection.execute("UPDATE nodes SET context_window_id = NULL")
        connection.execute("DELETE FROM context_window_edges")
        connection.execute("DELETE FROM context_windows")
        connection.execute("DELETE FROM repos")
    return [first.id, second.id]


def test_backfill_dry_run_does_not_assign_nodes(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    _make_legacy_nodes(graph)

    stats = backfill_context_windows(graph, dry_run=True)

    assert stats.dry_run is True
    assert stats.nodes_scanned == 2
    assert stats.nodes_assigned == 0
    assert len(graph.get_nodes_without_window()) == 2


def test_backfill_assigns_nodes_and_creates_windows(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    _make_legacy_nodes(graph)

    stats = backfill_context_windows(graph)

    assert stats.errors == []
    assert stats.nodes_scanned == 2
    assert stats.nodes_assigned == 2
    assert stats.repos_created == 1
    assert stats.windows_created == 2
    assert stats.embeddings_computed == 2
    assert len(graph.get_nodes_without_window()) == 0
    assert len(graph.list_context_windows(project="alpha")) == 2


def test_backfill_derives_cross_window_edges(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    _make_legacy_nodes(graph)

    stats = backfill_context_windows(graph)
    windows = graph.list_context_windows(project="alpha")
    edge_types = {edge.edge_type for window in windows for edge in graph.get_context_window_edges(window.id)}

    assert stats.window_edges_created >= 1
    assert "supersedes" in edge_types


def test_backfill_is_idempotent(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    _make_legacy_nodes(graph)

    first = backfill_context_windows(graph)
    second = backfill_context_windows(graph)

    assert first.nodes_assigned == 2
    assert second.nodes_scanned == 0
    assert second.nodes_assigned == 0
