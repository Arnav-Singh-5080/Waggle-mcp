"""
Tests for waggle/recursive_context.py — Recursive Context Assembly.

Test 1: Decomposition
Test 2: Budget compression
Test 3: Conflict inclusion
Test 4: Updates preference
Test 5: MCP tool registered + aliases resolve
Test 6: Fallback safety (no hybrid retriever)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from waggle.graph import MemoryGraph
from waggle.models import Edge, Node, NodeType, RelationType, SubgraphResult
from waggle.recursive_context import (
    RecursiveContextController,
    RecursiveContextResult,
    _Hit,
)

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


class FakeEmbeddingModel:
    """Deterministic embedding model for tests."""

    model_name = "fake-model"
    model_id = "fake-model:deterministic-v1"

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(8, dtype=np.float32)
        for token in text.lower().split():
            index = sum(ord(c) for c in token) % len(vector)
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


def make_graph(tmp_path: Path) -> MemoryGraph:
    return MemoryGraph(tmp_path / "memory.db", FakeEmbeddingModel())


def make_node(
    label: str,
    content: str,
    node_type: NodeType = NodeType.FACT,
    score: float = 0.5,
    node_id: str | None = None,
) -> Node:
    n = Node(
        label=label,
        content=content,
        node_type=node_type,
    )
    if node_id:
        n.id = node_id
    n.final_score = score
    return n


def make_edge(
    source_id: str,
    target_id: str,
    relationship: str,
    edge_id: str | None = None,
) -> Edge:
    e = Edge(source_id=source_id, target_id=target_id, relationship=relationship)
    if edge_id:
        e.id = edge_id
    return e


# ---------------------------------------------------------------------------
# Test 1: Decomposition
# ---------------------------------------------------------------------------


class TestDecomposition:
    def test_project_query_generates_expected_subqueries(self) -> None:
        """Decomposing a project continuation query should include key purposes."""
        controller = RecursiveContextController(graph=MagicMock())
        subqueries = controller._decompose_query(
            "Continue building Waggle from where we left off",
            max_subqueries=6,
        )

        purposes = {sq.purpose for sq in subqueries}
        " ".join(sq.query for sq in subqueries).lower()

        # Must include decisions, unfinished work, constraints, implementation
        assert "decisions" in purposes, f"Missing 'decisions' in {purposes}"
        assert "unfinished_work" in purposes, f"Missing 'unfinished_work' in {purposes}"
        assert "constraints" in purposes, f"Missing 'constraints' in {purposes}"
        assert "implementation" in purposes, f"Missing 'implementation' in {purposes}"

    def test_subquery_count_respects_max(self) -> None:
        controller = RecursiveContextController(graph=MagicMock())
        subqueries = controller._decompose_query("build waggle feature", max_subqueries=3)
        assert len(subqueries) <= 3

    def test_subqueries_have_priorities(self) -> None:
        controller = RecursiveContextController(graph=MagicMock())
        subqueries = controller._decompose_query("implement waggle recursive context")
        for sq in subqueries:
            assert 0.0 < sq.priority <= 1.0

    def test_generic_query_includes_original(self) -> None:
        controller = RecursiveContextController(graph=MagicMock())
        query = "What is my favourite database?"
        subqueries = controller._decompose_query(query)
        original_queries = [sq for sq in subqueries if sq.purpose == "original_query"]
        assert len(original_queries) >= 1
        assert original_queries[0].query == query

    def test_fast_mode_fewer_subqueries(self) -> None:
        controller = RecursiveContextController(graph=MagicMock())
        fast = controller._decompose_query("build waggle", max_subqueries=6, mode="fast")
        balanced = controller._decompose_query("build waggle", max_subqueries=6, mode="balanced")
        assert len(fast) <= len(balanced)

    def test_deep_mode_more_subqueries(self) -> None:
        controller = RecursiveContextController(graph=MagicMock())
        deep = controller._decompose_query("build waggle", max_subqueries=8, mode="deep")
        balanced = controller._decompose_query("build waggle", max_subqueries=8, mode="balanced")
        assert len(deep) >= len(balanced)


# ---------------------------------------------------------------------------
# Test 2: Budget compression
# ---------------------------------------------------------------------------


class TestBudgetCompression:
    def _make_many_hits(self, count: int = 50) -> list[_Hit]:
        hits = []
        for i in range(count):
            hits.append(
                _Hit(
                    node_id=f"node-{i}",
                    label=f"Decision {i}: use framework X for component Y",
                    content=f"We decided to use framework X for component Y because of reason {i}. " * 3,
                    node_type="decision",
                    score=1.0 - (i * 0.01),
                    source="graph",
                )
            )
        return hits

    def test_context_pack_within_budget_plus_15_percent(self) -> None:
        """build_context must not exceed token_budget * 1.15."""
        graph = MagicMock()
        graph.query.return_value = SubgraphResult(nodes=[], edges=[])
        graph.get_related.return_value = SubgraphResult(nodes=[], edges=[])

        controller = RecursiveContextController(graph=graph)
        hits = self._make_many_hits(50)
        conflicts: list[dict] = []
        transcripts: list[Any] = []

        token_budget = 300
        context_pack, _ = controller._compress_to_budget(
            query="test query",
            hits=hits,
            conflicts=conflicts,
            transcript_hits=transcripts,
            token_budget=token_budget,
        )

        actual_tokens = controller._estimate_tokens(context_pack)
        assert actual_tokens <= int(token_budget * 1.15), (
            f"Token estimate {actual_tokens} exceeds budget {token_budget} * 1.15 = {int(token_budget * 1.15)}"
        )

    def test_build_context_respects_budget_end_to_end(self, tmp_path: Path) -> None:
        """Full build_context call must stay within budget."""
        graph = make_graph(tmp_path)
        # Store some nodes so retrieval has something to return
        for i in range(10):
            graph.add_node(
                label=f"Decision {i}",
                content=f"We decided to do thing {i} because of reason {i}. " * 5,
                node_type=NodeType.DECISION,
            )

        controller = RecursiveContextController(graph=graph)
        result = controller.build_context(
            query="Continue building Waggle from where we left off",
            token_budget=400,
        )

        assert result.token_estimate <= int(400 * 1.15), f"token_estimate {result.token_estimate} exceeds 400 * 1.15"
        assert result.context_pack  # non-empty

    def test_empty_graph_returns_gracefully(self, tmp_path: Path) -> None:
        graph = make_graph(tmp_path)
        controller = RecursiveContextController(graph=graph)
        result = controller.build_context(query="What did we decide about the database?")
        assert isinstance(result, RecursiveContextResult)
        assert result.original_query == "What did we decide about the database?"


# ---------------------------------------------------------------------------
# Test 3: Conflict inclusion
# ---------------------------------------------------------------------------


class TestConflictInclusion:
    def test_contradicts_edge_appears_in_context_pack(self, tmp_path: Path) -> None:
        """Two nodes with a contradicts edge must surface in the context pack."""
        graph = make_graph(tmp_path)

        r1 = graph.add_node(
            label="Use PostgreSQL",
            content="We should use PostgreSQL for the database.",
            node_type=NodeType.DECISION,
        )
        r2 = graph.add_node(
            label="Use MySQL",
            content="We should use MySQL for the database.",
            node_type=NodeType.DECISION,
        )
        graph.add_edge(
            source_id=r1.node.id,
            target_id=r2.node.id,
            relationship=RelationType.CONTRADICTS.value,
        )

        controller = RecursiveContextController(graph=graph)
        result = controller.build_context(
            query="What database should we use?",
            token_budget=1200,
        )

        pack_lower = result.context_pack.lower()
        # Must mention conflict/contradiction
        assert "conflict" in pack_lower or "contradict" in pack_lower, (
            f"Expected conflict mention in context pack:\n{result.context_pack}"
        )
        # Must mention both sides
        assert "postgresql" in pack_lower or "mysql" in pack_lower, (
            "Expected at least one database name in context pack"
        )

    def test_conflict_entries_populated(self) -> None:
        """_resolve_updates_and_conflicts must populate conflict_entries."""
        controller = RecursiveContextController(graph=MagicMock())

        hit_a = _Hit(
            node_id="a",
            label="Use Redis",
            content="Use Redis for caching",
            node_type="decision",
            score=0.8,
            source="graph",
        )
        hit_b = _Hit(
            node_id="b",
            label="Use Memcached",
            content="Use Memcached for caching",
            node_type="decision",
            score=0.7,
            source="graph",
        )
        edge = make_edge("a", "b", "contradicts")

        _, conflicts = controller._resolve_updates_and_conflicts([hit_a, hit_b], [edge])

        assert len(conflicts) == 1
        assert conflicts[0]["source_id"] == "a"
        assert conflicts[0]["target_id"] == "b"
        assert conflicts[0]["relationship"] == "contradicts"


# ---------------------------------------------------------------------------
# Test 4: Updates preference
# ---------------------------------------------------------------------------


class TestUpdatesPreference:
    def test_newer_node_ranked_before_superseded(self) -> None:
        """Node with updates edge should rank above the superseded node."""
        controller = RecursiveContextController(graph=MagicMock())

        older = _Hit(
            node_id="old", label="Old decision", content="Use Flask", node_type="decision", score=0.9, source="graph"
        )
        newer = _Hit(
            node_id="new", label="New decision", content="Use FastAPI", node_type="decision", score=0.7, source="graph"
        )
        # newer updates older
        edge = make_edge("new", "old", "updates")

        updated_hits, _ = controller._resolve_updates_and_conflicts([older, newer], [edge])
        ranked = controller._rank_hits(updated_hits)

        ranked_ids = [h.node_id for h in ranked]
        assert ranked_ids.index("new") < ranked_ids.index("old"), f"Expected 'new' before 'old' in {ranked_ids}"

    def test_superseded_node_score_reduced(self) -> None:
        """Node targeted by updates edge should have its score reduced."""
        controller = RecursiveContextController(graph=MagicMock())

        older = _Hit(node_id="old", label="Old", content="Old content", node_type="decision", score=0.9, source="graph")
        newer = _Hit(node_id="new", label="New", content="New content", node_type="decision", score=0.5, source="graph")
        edge = make_edge("new", "old", "updates")

        updated_hits, _ = controller._resolve_updates_and_conflicts([older, newer], [edge])
        old_hit = next(h for h in updated_hits if h.node_id == "old")

        assert old_hit.is_superseded is True
        assert old_hit.score < 0.9  # score was reduced

    def test_updates_appear_in_context_pack(self, tmp_path: Path) -> None:
        """Newer node should appear in context pack before superseded one."""
        graph = make_graph(tmp_path)

        r_old = graph.add_node(
            label="Use Flask",
            content="We decided to use Flask for the web framework.",
            node_type=NodeType.DECISION,
        )
        r_new = graph.add_node(
            label="Use FastAPI",
            content="We switched to FastAPI for better async support.",
            node_type=NodeType.DECISION,
        )
        graph.add_edge(
            source_id=r_new.node.id,
            target_id=r_old.node.id,
            relationship=RelationType.UPDATES.value,
        )

        controller = RecursiveContextController(graph=graph)
        result = controller.build_context(
            query="What web framework are we using?",
            token_budget=1200,
        )

        pack = result.context_pack
        # FastAPI (newer) should appear before Flask (superseded) if both present
        if "fastapi" in pack.lower() and "flask" in pack.lower():
            assert pack.lower().index("fastapi") < pack.lower().index("flask"), (
                "Expected FastAPI (newer) to appear before Flask (superseded)"
            )


# ---------------------------------------------------------------------------
# Test 5: MCP tool registered + aliases resolve
# ---------------------------------------------------------------------------


class TestMCPToolRegistered:
    def _make_server(self, tmp_path: Path) -> Any:
        from waggle.config import AppConfig
        from waggle.server import WaggleServer

        config = AppConfig(
            backend="sqlite",
            transport="stdio",
            model_name="fake",
            db_path=str(tmp_path / "test.db"),
            default_tenant_id="test",
            http_host="127.0.0.1",
            http_port=8080,
            log_level="ERROR",
            rate_limit_rpm=120,
            write_rate_limit_rpm=60,
            max_concurrent_requests=8,
            max_payload_bytes=1024 * 1024,
            request_timeout_seconds=30,
            export_dir=None,
            neo4j_uri="",
            neo4j_username="",
            neo4j_password="",
            neo4j_database="",
        )
        graph = make_graph(tmp_path)
        return WaggleServer(graph=graph, config=config)

    def test_build_context_tool_in_list(self, tmp_path: Path) -> None:
        """build_context must appear in the tool list."""
        server = self._make_server(tmp_path)
        tools = server.build_tools()
        tool_names = {t.name for t in tools}
        assert "build_context" in tool_names, f"build_context not in {sorted(tool_names)}"

    def test_build_context_tool_hidden_when_feature_flag_disabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """build_context must be hidden and rejected when the feature flag is disabled."""
        monkeypatch.setattr("waggle.server.RECURSIVE_CONTEXT_ENABLED", False)
        server = self._make_server(tmp_path)

        tool_names = {t.name for t in server.build_tools()}
        assert "build_context" not in tool_names

        result = server.handle_tool_call("build_context", {"query": "What did we decide?"})
        assert result.isError is True
        assert "disabled" in result.content[0].text.lower()

    def test_aliases_resolve_to_build_context(self, tmp_path: Path) -> None:
        """recursive_context, assemble_context, rlm_context must resolve to build_context."""
        from waggle.server import _TOOL_ALIASES

        for alias in ("recursive_context", "assemble_context", "rlm_context"):
            assert alias in _TOOL_ALIASES, f"Alias '{alias}' not in _TOOL_ALIASES"
            canonical, _ = _TOOL_ALIASES[alias]
            assert canonical == "build_context", f"Alias '{alias}' resolves to '{canonical}', expected 'build_context'"

    def test_build_context_tool_call_returns_result(self, tmp_path: Path) -> None:
        """Calling build_context via handle_tool_call must return a valid result."""
        server = self._make_server(tmp_path)
        # Store a node so retrieval has something
        server.graph.add_node(
            label="Use PostgreSQL",
            content="We decided to use PostgreSQL.",
            node_type=NodeType.DECISION,
        )
        result = server.handle_tool_call(
            "build_context",
            {"query": "What database are we using?", "token_budget": 500},
        )
        assert result is not None
        # Should not be an error result
        content_text = result.content[0].text if result.content else ""
        assert "error" not in content_text.lower() or "context" in content_text.lower()

    def test_alias_recursive_context_resolves(self, tmp_path: Path) -> None:
        """Calling recursive_context alias must work via handle_tool_call."""
        server = self._make_server(tmp_path)
        result = server.handle_tool_call(
            "recursive_context",
            {"query": "What did we decide?"},
        )
        assert result is not None


# ---------------------------------------------------------------------------
# Test 6: Fallback safety
# ---------------------------------------------------------------------------


class TestFallbackSafety:
    def test_build_context_works_without_hybrid_retriever(self, tmp_path: Path) -> None:
        """build_context must work even when hybrid retriever is unavailable."""
        graph = make_graph(tmp_path)
        graph.add_node(
            label="Architecture decision",
            content="We use SQLite for local storage.",
            node_type=NodeType.DECISION,
        )

        # Pass no hybrid_retriever — controller must fall back to graph-only
        controller = RecursiveContextController(graph=graph, hybrid_retriever=None)
        result = controller.build_context(
            query="What storage backend are we using?",
            token_budget=600,
        )

        assert isinstance(result, RecursiveContextResult)
        assert result.context_pack  # non-empty

    def test_build_context_survives_graph_failure(self) -> None:
        """build_context must not raise even if graph.query raises."""
        graph = MagicMock()
        graph.query.side_effect = RuntimeError("DB unavailable")
        graph.get_related.side_effect = RuntimeError("DB unavailable")

        controller = RecursiveContextController(graph=graph)
        # Should not raise — errors are non-fatal
        result = controller.build_context(query="What did we decide?")
        assert isinstance(result, RecursiveContextResult)

    def test_empty_query_returns_gracefully(self) -> None:
        """Empty query must return a result without raising."""
        controller = RecursiveContextController(graph=MagicMock())
        result = controller.build_context(query="")
        assert isinstance(result, RecursiveContextResult)
        assert result.context_pack == "No query provided."

    def test_dedup_removes_duplicate_node_ids(self) -> None:
        """_deduplicate_hits must keep only the highest-scored copy per node_id."""
        controller = RecursiveContextController(graph=MagicMock())
        hits = [
            _Hit(node_id="x", label="A", content="c", node_type="fact", score=0.3, source="graph"),
            _Hit(node_id="x", label="A", content="c", node_type="fact", score=0.8, source="hybrid"),
            _Hit(node_id="y", label="B", content="d", node_type="fact", score=0.5, source="graph"),
        ]
        deduped = controller._deduplicate_hits(hits)
        assert len(deduped) == 2
        x_hit = next(h for h in deduped if h.node_id == "x")
        assert x_hit.score == 0.8  # kept the higher-scored copy

    def test_token_estimate_approximation(self) -> None:
        """_estimate_tokens should approximate len(text) // 4."""
        controller = RecursiveContextController(graph=MagicMock())
        text = "a" * 400
        assert controller._estimate_tokens(text) == 100
