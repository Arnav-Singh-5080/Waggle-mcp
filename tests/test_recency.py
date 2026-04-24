from __future__ import annotations

import math
import sqlite3
import time
from pathlib import Path

import numpy as np

from waggle.graph import MemoryGraph, recency_weight, score_node
from waggle.models import NodeType, RelationType


class ConstantEmbeddingModel:
    def embed(self, text: str) -> np.ndarray:
        del text
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def to_bytes(self, embedding: np.ndarray) -> bytes:
        return embedding.astype(np.float32).tobytes()

    def from_bytes(self, data: bytes) -> np.ndarray:
        return np.frombuffer(data, dtype=np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        del a, b
        return 1.0


def make_graph(tmp_path: Path) -> MemoryGraph:
    return MemoryGraph(
        tmp_path / "recency-memory.db",
        ConstantEmbeddingModel(),
        dedup_similarity_threshold=1.1,
        dedup_same_label_threshold=1.1,
    )


def _set_updated_at(graph: MemoryGraph, node_id: str, updated_at_epoch: float) -> None:
    updated_at = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(updated_at_epoch))
    with sqlite3.connect(graph.db_path) as connection:
        connection.execute("UPDATE nodes SET updated_at = ? WHERE id = ?", (updated_at, node_id))


def test_recency_weight_curve() -> None:
    now = 1_000_000.0
    assert recency_weight(now, now=now) == 1.0
    assert recency_weight(now - (30.0 * 86400.0), now=now) == math.exp(-0.693)
    assert recency_weight(now - (150.0 * 86400.0), now=now) < 0.05


def test_score_node_weights_components() -> None:
    now = 1_000_000.0
    score = score_node(
        0.8,
        now - (30.0 * 86400.0),
        edge_weight=0.4,
        now=now,
        half_life_days=30.0,
    )
    expected = (0.8 * 0.6) + (math.exp(-0.693) * 0.3) + (0.4 * 0.1)
    assert score == expected


def test_score_node_superseded_penalty_lowers_rank() -> None:
    now = 1_000_000.0
    current = score_node(0.9, now, edge_weight=0.8, now=now, superseded=False)
    superseded = score_node(0.9, now, edge_weight=0.8, now=now, superseded=True)
    assert superseded < current
    assert superseded == current * 0.2


def test_query_graph_prefers_newer_nodes_when_similarity_matches(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    old_node = graph.add_node(
        label="Alpha memory",
        content="legacy zebra payload",
        node_type=NodeType.FACT,
    ).node
    new_node = graph.add_node(
        label="Beta memory",
        content="modern quartz payload",
        node_type=NodeType.FACT,
    ).node
    graph.add_edge(
        source_id=new_node.id,
        target_id=old_node.id,
        relationship=RelationType.UPDATES,
        weight=1.0,
    )

    now = time.time()
    _set_updated_at(graph, old_node.id, now - (120.0 * 86400.0))
    _set_updated_at(graph, new_node.id, now - 3600.0)

    result = graph.query(query="freshness ranking probe", max_nodes=2, max_depth=1)

    assert [node.label for node in result.nodes[:2]] == ["Beta memory", "Alpha memory"]
    assert result.nodes[0].final_score is not None
    assert result.nodes[0].recency_score > result.nodes[1].recency_score


def test_superseded_nodes_rank_below_current_version(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    old_node = graph.add_node(
        label="Dog memory old",
        content="legacy zebra release note",
        node_type=NodeType.FACT,
    ).node
    new_node = graph.add_node(
        label="Dog memory new",
        content="modern quartz release note",
        node_type=NodeType.FACT,
    ).node
    graph.add_edge(
        source_id=new_node.id,
        target_id=old_node.id,
        relationship=RelationType.UPDATES,
        weight=1.0,
    )

    now = time.time()
    _set_updated_at(graph, old_node.id, now - 300.0)
    _set_updated_at(graph, new_node.id, now - 300.0)

    result = graph.get_related(node_id=new_node.id, max_depth=1)
    ranked = {node.id: node for node in result.nodes}

    assert ranked[old_node.id].metadata["superseded_by"] == new_node.id
    assert ranked[new_node.id].final_score > ranked[old_node.id].final_score
