"""Tests for valid_to / valid_from temporal validity enforcement.

Covers:
- Default queries exclude expired nodes (valid_to in the past).
- include_invalidated=True returns expired nodes.
- as_of=<past time> returns nodes valid at that point.
- resolve_conflict(winner=...) sets valid_to on the losing node only.
- resolve_conflict with a winner not on the edge raises ValueError.
- WAGGLE_ENFORCE_VALID_TO=false disables enforcement (legacy behaviour).
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from waggle.graph import MemoryGraph
from waggle.models import NodeType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeEmbeddingModel:
    model_name = "fake-model"
    model_id = "fake-model:deterministic-v1"

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(8, dtype=np.float32)
        for token in text.lower().split():
            index = sum(ord(c) for c in token) % len(vector)
            vector[index] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

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
    return MemoryGraph(tmp_path / "memory.db", FakeEmbeddingModel())


def utc(dt: datetime) -> datetime:
    """Ensure *dt* is UTC-aware."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NOW = datetime.now(UTC)
T0 = NOW - timedelta(hours=3)  # 3 h ago  — "created"
T1 = NOW - timedelta(hours=1)  # 1 h ago  — "expired"
T2 = NOW + timedelta(hours=1)  # 1 h from now — "future"


# ---------------------------------------------------------------------------
# 1. Default query excludes expired nodes
# ---------------------------------------------------------------------------


def test_default_query_excludes_expired_node(tmp_path: Path) -> None:
    """Node with valid_to in the past must NOT appear in default query results."""
    graph = make_graph(tmp_path)

    # Store a node that expired at T1 (in the past)
    result = graph.add_node(
        label="Expired fact",
        content="This fact has expired",
        node_type=NodeType.FACT,
        valid_from=T0,
        valid_to=T1,
    )
    expired_id = result.node.id

    # Default query — include_invalidated defaults to False
    subgraph = graph.query(query="expired fact", retrieval_mode="graph")
    returned_ids = {n.id for n in subgraph.nodes}
    assert expired_id not in returned_ids, "Expired node should be excluded from default query results"


def test_default_aggregate_excludes_expired_node(tmp_path: Path) -> None:
    """aggregate() must also exclude expired nodes by default."""
    graph = make_graph(tmp_path)

    result = graph.add_node(
        label="Expired preference",
        content="User preferred dark mode",
        node_type=NodeType.PREFERENCE,
        valid_from=T0,
        valid_to=T1,
    )
    expired_id = result.node.id

    subgraph = graph.aggregate(query="dark mode preference")
    returned_ids = {n.id for n in subgraph.nodes}
    assert expired_id not in returned_ids, "Expired node should be excluded from default aggregate results"


# ---------------------------------------------------------------------------
# 2. include_invalidated=True returns expired nodes
# ---------------------------------------------------------------------------


def test_include_invalidated_returns_expired_node(tmp_path: Path) -> None:
    """include_invalidated=True must return nodes whose valid_to has passed."""
    graph = make_graph(tmp_path)

    result = graph.add_node(
        label="Expired fact",
        content="This fact has expired",
        node_type=NodeType.FACT,
        valid_from=T0,
        valid_to=T1,
    )
    expired_id = result.node.id

    subgraph = graph.query(
        query="expired fact",
        retrieval_mode="graph",
        include_invalidated=True,
    )
    returned_ids = {n.id for n in subgraph.nodes}
    assert expired_id in returned_ids, "include_invalidated=True must return expired nodes"


def test_include_invalidated_aggregate_returns_expired_node(tmp_path: Path) -> None:
    """aggregate(include_invalidated=True) must return expired nodes."""
    graph = make_graph(tmp_path)

    result = graph.add_node(
        label="Expired preference",
        content="User preferred dark mode",
        node_type=NodeType.PREFERENCE,
        valid_from=T0,
        valid_to=T1,
    )
    expired_id = result.node.id

    subgraph = graph.aggregate(
        query="dark mode preference",
        include_invalidated=True,
    )
    returned_ids = {n.id for n in subgraph.nodes}
    assert expired_id in returned_ids, "aggregate(include_invalidated=True) must return expired nodes"


# ---------------------------------------------------------------------------
# 3. as_of returns nodes valid at that point in time
# ---------------------------------------------------------------------------


def test_as_of_returns_node_valid_at_that_time(tmp_path: Path) -> None:
    """as_of=T0+30min should return a node valid between T0 and T1."""
    graph = make_graph(tmp_path)

    result = graph.add_node(
        label="Expired fact",
        content="This fact has expired",
        node_type=NodeType.FACT,
        valid_from=T0,
        valid_to=T1,
    )
    expired_id = result.node.id

    # Query at a time when the node was still valid (halfway between T0 and T1)
    as_of_time = T0 + (T1 - T0) / 2

    subgraph = graph.query(
        query="expired fact",
        retrieval_mode="graph",
        as_of=as_of_time,
    )
    returned_ids = {n.id for n in subgraph.nodes}
    assert expired_id in returned_ids, "as_of within the validity window must return the node"


def test_as_of_excludes_node_not_yet_valid(tmp_path: Path) -> None:
    """as_of before valid_from should exclude the node."""
    graph = make_graph(tmp_path)

    result = graph.add_node(
        label="Future fact",
        content="This fact is not yet valid",
        node_type=NodeType.FACT,
        valid_from=T2,  # starts in the future
        valid_to=None,
    )
    future_id = result.node.id

    # Query at NOW — before valid_from
    subgraph = graph.query(
        query="future fact",
        retrieval_mode="graph",
        as_of=NOW,
    )
    returned_ids = {n.id for n in subgraph.nodes}
    assert future_id not in returned_ids, "as_of before valid_from must exclude the node"


def test_as_of_aggregate_returns_node_valid_at_that_time(tmp_path: Path) -> None:
    """aggregate(as_of=...) should return nodes valid at that point."""
    graph = make_graph(tmp_path)

    result = graph.add_node(
        label="Expired preference",
        content="User preferred dark mode",
        node_type=NodeType.PREFERENCE,
        valid_from=T0,
        valid_to=T1,
    )
    expired_id = result.node.id

    as_of_time = T0 + (T1 - T0) / 2

    subgraph = graph.aggregate(
        query="dark mode preference",
        as_of=as_of_time,
    )
    returned_ids = {n.id for n in subgraph.nodes}
    assert expired_id in returned_ids, "aggregate(as_of=...) within validity window must return the node"


# ---------------------------------------------------------------------------
# 4. resolve_conflict(winner=...) sets valid_to on the losing node only
# ---------------------------------------------------------------------------


def test_resolve_conflict_with_winner_sets_valid_to_on_loser(tmp_path: Path) -> None:
    """resolve_conflict(winner=A) must set valid_to on B, not on A."""
    graph = make_graph(tmp_path)

    node_a = graph.add_node(
        label="REST preference",
        content="User prefers REST APIs",
        node_type=NodeType.PREFERENCE,
    ).node

    node_b = graph.add_node(
        label="GraphQL preference",
        content="User prefers GraphQL APIs",
        node_type=NodeType.PREFERENCE,
    ).node

    # Find the auto-created CONTRADICTS edge
    conflicts = graph.list_conflicts()
    assert len(conflicts.conflicts) >= 1
    conflict_edge = conflicts.conflicts[0].edge

    before = datetime.now(UTC)
    graph.resolve_conflict(
        edge_id=conflict_edge.id,
        winner=node_a.id,
        resolution_note="REST wins",
    )
    after = datetime.now(UTC)

    # Reload both nodes
    refreshed_a = graph.get_node(node_a.id)
    refreshed_b = graph.get_node(node_b.id)

    # Winner (A) must NOT have valid_to set
    assert refreshed_a.valid_to is None, "Winning node must not have valid_to set"

    # Loser (B) must have valid_to set to approximately now
    assert refreshed_b.valid_to is not None, "Losing node must have valid_to set after resolve_conflict"
    assert before <= refreshed_b.valid_to <= after, "Losing node's valid_to must be set to the time of resolution"


def test_resolve_conflict_loser_excluded_from_default_query(tmp_path: Path) -> None:
    """After resolve_conflict(winner=A), B must not appear in default queries."""
    graph = make_graph(tmp_path)

    node_a = graph.add_node(
        label="REST preference",
        content="User prefers REST APIs",
        node_type=NodeType.PREFERENCE,
    ).node

    node_b = graph.add_node(
        label="GraphQL preference",
        content="User prefers GraphQL APIs",
        node_type=NodeType.PREFERENCE,
    ).node

    conflicts = graph.list_conflicts()
    conflict_edge = conflicts.conflicts[0].edge

    graph.resolve_conflict(edge_id=conflict_edge.id, winner=node_a.id)

    # Default query — B should be gone
    subgraph = graph.query(query="API preference", retrieval_mode="graph")
    returned_ids = {n.id for n in subgraph.nodes}
    assert node_b.id not in returned_ids, "Losing node must be excluded from default queries after resolve_conflict"
    # A should still be present
    assert node_a.id in returned_ids, "Winning node must still appear in default queries"


def test_resolve_conflict_loser_visible_with_include_invalidated(tmp_path: Path) -> None:
    """After resolve_conflict(winner=A), B must be visible with include_invalidated=True."""
    graph = make_graph(tmp_path)

    node_a = graph.add_node(
        label="REST preference",
        content="User prefers REST APIs",
        node_type=NodeType.PREFERENCE,
    ).node

    node_b = graph.add_node(
        label="GraphQL preference",
        content="User prefers GraphQL APIs",
        node_type=NodeType.PREFERENCE,
    ).node

    conflicts = graph.list_conflicts()
    conflict_edge = conflicts.conflicts[0].edge

    graph.resolve_conflict(edge_id=conflict_edge.id, winner=node_a.id)

    subgraph = graph.query(
        query="API preference",
        retrieval_mode="graph",
        include_invalidated=True,
    )
    returned_ids = {n.id for n in subgraph.nodes}
    assert node_b.id in returned_ids, "Losing node must be visible with include_invalidated=True"


# ---------------------------------------------------------------------------
# 5. resolve_conflict with invalid winner raises ValueError
# ---------------------------------------------------------------------------


def test_resolve_conflict_invalid_winner_raises(tmp_path: Path) -> None:
    """Passing a winner that is not an endpoint of the edge must raise ValueError."""
    graph = make_graph(tmp_path)

    graph.add_node(
        label="REST preference",
        content="User prefers REST APIs",
        node_type=NodeType.PREFERENCE,
    )
    graph.add_node(
        label="GraphQL preference",
        content="User prefers GraphQL APIs",
        node_type=NodeType.PREFERENCE,
    )

    unrelated = graph.add_node(
        label="Unrelated node",
        content="Something completely different",
        node_type=NodeType.FACT,
    ).node

    conflicts = graph.list_conflicts()
    conflict_edge = conflicts.conflicts[0].edge

    with pytest.raises(ValueError, match="not an endpoint"):
        graph.resolve_conflict(
            edge_id=conflict_edge.id,
            winner=unrelated.id,
        )


# ---------------------------------------------------------------------------
# 6. WAGGLE_ENFORCE_VALID_TO=false disables enforcement (legacy behaviour)
# ---------------------------------------------------------------------------


def test_enforcement_disabled_via_env_var_returns_expired_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With WAGGLE_ENFORCE_VALID_TO=false, expired nodes appear in default queries."""
    monkeypatch.setenv("WAGGLE_ENFORCE_VALID_TO", "false")

    graph = make_graph(tmp_path)

    result = graph.add_node(
        label="Expired fact",
        content="This fact has expired",
        node_type=NodeType.FACT,
        valid_from=T0,
        valid_to=T1,
    )
    expired_id = result.node.id

    subgraph = graph.query(query="expired fact", retrieval_mode="graph")
    returned_ids = {n.id for n in subgraph.nodes}
    assert expired_id in returned_ids, (
        "With WAGGLE_ENFORCE_VALID_TO=false, expired nodes must appear in default queries"
    )


def test_enforcement_disabled_aggregate_returns_expired_nodes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With WAGGLE_ENFORCE_VALID_TO=false, aggregate also returns expired nodes."""
    monkeypatch.setenv("WAGGLE_ENFORCE_VALID_TO", "false")

    graph = make_graph(tmp_path)

    result = graph.add_node(
        label="Expired preference",
        content="User preferred dark mode",
        node_type=NodeType.PREFERENCE,
        valid_from=T0,
        valid_to=T1,
    )
    expired_id = result.node.id

    subgraph = graph.aggregate(query="dark mode preference")
    returned_ids = {n.id for n in subgraph.nodes}
    assert expired_id in returned_ids, "With WAGGLE_ENFORCE_VALID_TO=false, aggregate must return expired nodes"


# ---------------------------------------------------------------------------
# 7. Nodes without valid_to are always returned (no false exclusions)
# ---------------------------------------------------------------------------


def test_node_without_valid_to_always_returned(tmp_path: Path) -> None:
    """Nodes with valid_to=None must always appear in default queries."""
    graph = make_graph(tmp_path)

    result = graph.add_node(
        label="Permanent fact",
        content="This fact has no expiry",
        node_type=NodeType.FACT,
    )
    permanent_id = result.node.id

    subgraph = graph.query(query="permanent fact", retrieval_mode="graph")
    returned_ids = {n.id for n in subgraph.nodes}
    assert permanent_id in returned_ids, "Nodes without valid_to must always appear in default queries"


def test_node_with_future_valid_to_returned_by_default(tmp_path: Path) -> None:
    """Nodes whose valid_to is in the future must appear in default queries."""
    graph = make_graph(tmp_path)

    result = graph.add_node(
        label="Active fact",
        content="This fact is still active",
        node_type=NodeType.FACT,
        valid_to=T2,  # expires in the future
    )
    active_id = result.node.id

    subgraph = graph.query(query="active fact", retrieval_mode="graph")
    returned_ids = {n.id for n in subgraph.nodes}
    assert active_id in returned_ids, "Nodes with valid_to in the future must appear in default queries"
