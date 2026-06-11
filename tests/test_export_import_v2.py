from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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


def test_export_graph_backup_includes_context_window_hierarchy(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    node = graph.add_node(
        label="Hierarchy Node",
        content="Hierarchy exports should preserve context windows",
        node_type=NodeType.FACT,
        project="alpha",
        session_id="sess-1",
    ).node

    backup = graph.export_graph_backup(output_path=tmp_path / "backup.json")
    payload = json.loads(Path(backup.output_path).read_text())

    assert payload["schema_version"] >= 5
    assert payload["repos"]
    assert payload["context_windows"]
    assert payload["nodes"][0]["context_window_id"] == node.context_window_id
    assert payload["context_windows"][0]["embedding_stale"] is True


def test_import_graph_backup_recreates_context_window_hierarchy(tmp_path: Path) -> None:
    source = make_graph(tmp_path / "source")
    first = source.add_node(
        label="Dog",
        content="Dog is named X",
        node_type=NodeType.ENTITY,
        project="alpha",
        session_id="sess-1",
    ).node
    second = source.add_node(
        label="Dog",
        content="Dog is named Y",
        node_type=NodeType.ENTITY,
        project="alpha",
        session_id="sess-2",
    ).node
    assert first.context_window_id is not None
    assert second.context_window_id is not None
    source.derive_context_window_edges(second.context_window_id, source.ensure_repo("alpha"))
    backup = source.export_graph_backup(output_path=tmp_path / "backup.json")

    target = make_graph(tmp_path / "target")
    imported = target.import_graph_backup(input_path=backup.output_path)

    assert imported.nodes_created == 2
    windows = target.list_context_windows(project="alpha")
    assert len(windows) == 2
    imported_first = target.get_node(first.id)
    assert imported_first.context_window_id == first.context_window_id
    edge_types = {edge.edge_type for window in windows for edge in target.get_context_window_edges(window.id)}
    assert "supersedes" in edge_types


def test_import_legacy_backup_without_hierarchy_still_works(tmp_path: Path) -> None:
    source = make_graph(tmp_path / "source")
    source.add_node(
        label="Legacy Node",
        content="Legacy backups should still import",
        node_type=NodeType.FACT,
    )
    backup = source.export_graph_backup(output_path=tmp_path / "backup.json")
    payload = json.loads(Path(backup.output_path).read_text())
    payload.pop("repos", None)
    payload.pop("context_windows", None)
    payload.pop("context_window_edges", None)
    for node in payload["nodes"]:
        node.pop("context_window_id", None)
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(payload))

    target = make_graph(tmp_path / "target")
    imported = target.import_graph_backup(input_path=legacy_path)

    assert imported.nodes_created == 1
    assert target.get_stats().total_nodes == 1
