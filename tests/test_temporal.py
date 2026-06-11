"""Regression tests for topic-gated temporal ranking."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from waggle.retrieval.rankers.temporal import rank_temporal


@dataclass(frozen=True)
class Node:
    """Minimal graph node used by temporal ranker tests."""

    text: str
    ts: str
    tags: list[str]


@pytest.fixture
def minimal_graph_nodes() -> list[Node]:
    """Build the minimal temporal test graph."""
    return [
        Node(
            "Auth token expired at 10am",
            ts="2024-01-15T10:00",
            tags=["security-review"],
        ),
        Node(
            "Privacy export completed",
            ts="2024-01-15T11:00",
            tags=["privacy-export"],
        ),
        Node(
            "Auth request rejected by admin",
            ts="2024-01-15T09:00",
            tags=["security-review"],
        ),
        Node(
            "Model deployed to staging",
            ts="2024-01-15T12:00",
            tags=["model-ops"],
        ),
    ]


def test_latest_auth_query_returns_auth_node_not_newest_global_node(
    minimal_graph_nodes: list[Node],
) -> None:
    """Latest auth query must not return unrelated globally newest model node."""
    result = rank_temporal(
        "latest auth token",
        minimal_graph_nodes,
        direction="latest",
        top_k=1,
    )

    assert result[0].text == "Auth token expired at 10am"


def test_oldest_privacy_query_returns_earliest_privacy_node() -> None:
    """Oldest privacy query sorts within privacy-topic survivors."""
    nodes = [
        Node("Privacy export completed", ts="2024-01-15T11:00", tags=["privacy-export"]),
        Node("Privacy request opened", ts="2024-01-15T08:00", tags=["privacy-export"]),
        Node("Auth request rejected by admin", ts="2024-01-15T07:00", tags=["security-review"]),
    ]

    result = rank_temporal(
        "oldest privacy export",
        nodes,
        direction="oldest",
        top_k=1,
    )

    assert result[0].text == "Privacy request opened"


def test_fallback_when_no_candidates_pass_threshold(
    minimal_graph_nodes: list[Node],
) -> None:
    """If no candidates pass the topic gate, fallback uses top_k*2 by topic score."""
    result = rank_temporal(
        "unmatched billing invoice",
        minimal_graph_nodes,
        direction="latest",
        top_k=1,
        topic_threshold=0.99,
    )

    assert len(result) == 1
    assert result[0] in minimal_graph_nodes


def test_same_timestamp_tie_prefers_higher_topic_score() -> None:
    """When timestamps tie, temporal ranking prefers the stronger topic match."""
    nodes = [
        Node("Auth token expired at 10am", ts="2024-01-15T10:00", tags=["security-review"]),
        Node("Auth token expiry policy updated", ts="2024-01-15T10:00", tags=["security-review"]),
        Node("Model deployed to staging", ts="2024-01-15T12:00", tags=["model-ops"]),
    ]

    result = rank_temporal(
        "latest auth token expiry policy",
        nodes,
        direction="latest",
        top_k=1,
    )

    assert result[0].text == "Auth token expiry policy updated"
