from __future__ import annotations

from pathlib import Path

import numpy as np

from waggle.graph import MemoryGraph
from waggle.models import NodeType


class FakeEmbeddingModel:
    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(16, dtype=np.float32)
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


def make_graph(tmp_path: Path, *, tiered: bool = False) -> MemoryGraph:
    return MemoryGraph(
        tmp_path / "memory.db",
        FakeEmbeddingModel(),
        tiered_retrieval=tiered,
        tiered_retrieval_top_k_windows=1,
    )


def test_tiered_routes_to_matching_window(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    graph.add_node(
        label="Database Decision",
        content="Postgres is the production database for invoices",
        node_type=NodeType.DECISION,
        project="waggle",
        session_id="database-session",
    )
    graph.add_node(
        label="Frontend Decision",
        content="React renders the dashboard navigation",
        node_type=NodeType.DECISION,
        project="waggle",
        session_id="frontend-session",
    )

    result = graph.tiered_query(query="postgres database invoices", project="waggle", max_nodes=3, top_k_windows=1)

    assert result.retrieval_mode == "tiered"
    assert result.nodes
    assert result.nodes[0].session_id == "database-session"
    assert result.nodes[0].context_window_id is not None


def test_tiered_query_falls_back_without_windows(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    graph.add_node(
        label="Legacy Node",
        content="Legacy flat retrieval should still answer",
        node_type=NodeType.FACT,
        project="other-project",
        session_id="legacy-session",
    )

    result = graph.tiered_query(query="legacy retrieval", project="missing-project", max_nodes=3)

    assert result.retrieval_mode == "flat_fallback"


def test_query_uses_tiered_when_enabled_and_project_scoped(tmp_path: Path) -> None:
    graph = make_graph(tmp_path, tiered=True)
    graph.add_node(
        label="API Decision",
        content="FastAPI serves the tenant scoped API",
        node_type=NodeType.DECISION,
        project="waggle",
        session_id="api-session",
    )

    result = graph.query(query="fastapi api", project="waggle", max_nodes=3)

    assert result.retrieval_mode == "tiered"
    assert result.nodes[0].session_id == "api-session"


def test_query_keeps_flat_mode_when_tiered_disabled(tmp_path: Path) -> None:
    graph = make_graph(tmp_path, tiered=False)
    graph.add_node(
        label="API Decision",
        content="FastAPI serves the tenant scoped API",
        node_type=NodeType.DECISION,
        project="waggle",
        session_id="api-session",
    )

    result = graph.query(query="fastapi api", project="waggle", max_nodes=3)

    assert result.retrieval_mode == "graph"


def test_debug_retrieval_returns_window_and_flat_comparison(tmp_path: Path) -> None:
    graph = make_graph(tmp_path, tiered=True)
    graph.add_node(
        label="Database Decision",
        content="Postgres is the production database for invoices",
        node_type=NodeType.DECISION,
        project="waggle",
        session_id="database-session",
    )
    graph.add_node(
        label="Frontend Decision",
        content="React renders the dashboard navigation",
        node_type=NodeType.DECISION,
        project="waggle",
        session_id="frontend-session",
    )

    debug = graph.debug_retrieval(query="postgres database invoices", project="waggle", max_nodes=3)

    assert debug["repo_id"].endswith(":waggle")
    assert debug["embedding_preview"]
    assert debug["windows_evaluated"] == 2
    assert debug["selected_windows"]
    assert debug["selected_windows"][0]["session_id"] == "database-session"
    assert debug["flat_top_nodes"]
    assert debug["tiered_top_nodes"]
    assert debug["tiered_result_mode"] == "tiered"
