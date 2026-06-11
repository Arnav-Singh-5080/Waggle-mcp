"""Regression tests for the four diff/merge fixes on feature/abhi-diff-merge-tool."""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any
from uuid import uuid4

import pytest

from waggle.abhi import (
    _apply_merge_strategy,
    diff_abhi_documents,
    validate_abhi_signature,
)
from waggle.errors import ValidationFailure
from waggle.models import (
    MergeConflictRecord,
    MergeStrategyConfig,
    MergeStrategyFieldOverride,
    MergeStrategyTypeOverride,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(**overrides: Any) -> dict[str, Any]:
    defaults = {
        "id": str(uuid4()),
        "label": "test-node",
        "content": "Some content",
        "node_type": "fact",
        "tags": [],
        "valid_from": None,
        "valid_to": None,
        "aliases": [],
        "metadata": {},
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    defaults.update(overrides)
    return defaults


def _make_edge(**overrides: Any) -> dict[str, Any]:
    defaults = {
        "id": str(uuid4()),
        "source_id": str(uuid4()),
        "target_id": str(uuid4()),
        "relationship": "relates_to",
        "weight": 1.0,
        "metadata": {},
        "created_at": "2026-01-01T00:00:00Z",
    }
    defaults.update(overrides)
    return defaults


def _make_document(nodes: list[dict], edges: list[dict] | None = None) -> dict[str, Any]:
    return {
        "manifest": {"schema_version": "2.0.0", "content_hash": ""},
        "nodes": nodes,
        "edges": edges or [],
        "transcripts": [],
        "context_windows": [],
    }


# ===========================================================================
# Fix #1: validate_abhi_signature warns when using bundled key
# ===========================================================================


class TestSignatureValidation:
    def test_unsigned_raises(self):
        document = {"manifest": {"signatures": {"present": False}}}
        with pytest.raises(ValidationFailure, match="not signed"):
            validate_abhi_signature(document)

    def test_bundled_key_emits_warning(self):
        """When no trusted key path is given, a UserWarning must be emitted."""
        # We need a signed document for the warning path to be reached.
        # Since we don't have real keys here, we'll catch the warning first.
        from cryptography.hazmat.primitives import serialization as _ser
        from cryptography.hazmat.primitives.asymmetric import ed25519 as _ed25519

        private_key = _ed25519.Ed25519PrivateKey.generate()
        public_pem = private_key.public_key().public_bytes(
            encoding=_ser.Encoding.PEM,
            format=_ser.PublicFormat.SubjectPublicKeyInfo,
        )
        payload = b"sha256:fakehash"
        signature = private_key.sign(payload)

        document = {
            "manifest": {
                "signatures": {"present": True},
                "content_hash": "sha256:fakehash",
            },
            "public_key_pem": public_pem,
            "signature": signature,
            "nodes": [],
            "edges": [],
            "transcripts": [],
            "context_windows": [],
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_abhi_signature(document)  # should pass but warn
            assert len(w) == 1
            assert "bundled inside the archive" in str(w[0].message)

    def test_trusted_key_path_missing_raises(self, tmp_path):
        document = {"manifest": {"signatures": {"present": True}}}
        missing = tmp_path / "nonexistent.pem"
        with pytest.raises(ValidationFailure, match="not found"):
            validate_abhi_signature(document, trusted_public_key_path=missing)

    def test_trusted_key_path_verifies(self, tmp_path):
        """Providing a trusted key path should not emit a warning."""
        from cryptography.hazmat.primitives import serialization as _ser
        from cryptography.hazmat.primitives.asymmetric import ed25519 as _ed25519

        private_key = _ed25519.Ed25519PrivateKey.generate()
        public_pem = private_key.public_key().public_bytes(
            encoding=_ser.Encoding.PEM,
            format=_ser.PublicFormat.SubjectPublicKeyInfo,
        )
        payload = b"sha256:fakehash"
        signature = private_key.sign(payload)

        key_file = tmp_path / "trusted.pem"
        key_file.write_bytes(public_pem)

        document = {
            "manifest": {
                "signatures": {"present": True},
                "content_hash": "sha256:fakehash",
            },
            "public_key_pem": b"SHOULD NOT BE USED",
            "signature": signature,
            "nodes": [],
            "edges": [],
            "transcripts": [],
            "context_windows": [],
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_abhi_signature(document, trusted_public_key_path=key_file)
            assert len(w) == 0, "No warning should be emitted when trusted key is provided"


# ===========================================================================
# Fix #2: contradict edges for edge conflicts reference nodes, not edge IDs
# ===========================================================================


class TestContradictEdgeNotDangling:
    def test_node_conflict_uses_node_id(self):
        """For node conflicts, the contradict edge should self-loop the node ID."""
        node_id = "node-abc"
        base = _make_node(id=node_id, content="original")
        left = _make_node(id=node_id, content="left-change")
        right = _make_node(id=node_id, content="right-change")

        conflicts: list[MergeConflictRecord] = []
        contradict_edges: list[dict] = []

        _apply_merge_strategy(
            node_id,
            "node",
            base,
            left,
            right,
            strategy="contradict",
            strategy_config=None,
            conflict_records=conflicts,
            contradict_edges=contradict_edges,
        )

        assert len(contradict_edges) >= 1
        for edge in contradict_edges:
            assert edge["source_id"] == node_id
            assert edge["target_id"] == node_id

    def test_edge_conflict_uses_node_ids_not_edge_id(self):
        """For edge conflicts, the contradict edge must reference the edge's
        source_id/target_id, NOT the edge's own ID."""
        edge_id = "edge-xyz"
        src, tgt = "node-1", "node-2"
        base = _make_edge(id=edge_id, source_id=src, target_id=tgt, weight=0.5)
        left = _make_edge(id=edge_id, source_id=src, target_id=tgt, weight=0.3)
        right = _make_edge(id=edge_id, source_id=src, target_id=tgt, weight=0.9)

        conflicts: list[MergeConflictRecord] = []
        contradict_edges: list[dict] = []

        _apply_merge_strategy(
            edge_id,
            "edge",
            base,
            left,
            right,
            strategy="contradict",
            strategy_config=None,
            conflict_records=conflicts,
            contradict_edges=contradict_edges,
        )

        assert len(contradict_edges) >= 1
        for ce in contradict_edges:
            # Must NOT reference the edge ID — that would create dangling edges
            assert ce["source_id"] != edge_id, "contradict edge must not use edge ID as source"
            assert ce["target_id"] != edge_id, "contradict edge must not use edge ID as target"
            # Must use the edge's actual node endpoints
            assert ce["source_id"] == src
            assert ce["target_id"] == tgt
            # Metadata should carry the conflicting edge ID for traceability
            assert ce["metadata"]["conflict_edge_id"] == edge_id


# ===========================================================================
# Fix #3: Three-way diff detects edge conflicts
# ===========================================================================


class TestThreeWayDiffEdgeConflicts:
    def test_two_way_diff_has_no_edge_conflicts(self):
        """Two-way diffs should never produce conflict_records (no base)."""
        nid = "n1"
        doc_a = _make_document([_make_node(id=nid)])
        doc_b = _make_document([_make_node(id=nid, content="changed")])
        result = diff_abhi_documents(doc_a, doc_b, input_path_a="a", input_path_b="b")
        assert result.diff_mode == "two_way"
        assert result.conflict_records == []

    def test_three_way_diff_detects_node_conflicts(self):
        """Sanity check: node conflicts should still work in three-way mode."""
        nid = "n1"
        base = _make_document([_make_node(id=nid, content="base")])
        left = _make_document([_make_node(id=nid, content="left-edit")])
        right = _make_document([_make_node(id=nid, content="right-edit")])
        result = diff_abhi_documents(left, right, input_path_a="a", input_path_b="b", base_document=base)
        assert result.diff_mode == "three_way"
        node_conflicts = [c for c in result.conflict_records if c.object_type == "node"]
        assert len(node_conflicts) >= 1

    def test_three_way_diff_detects_edge_conflicts(self):
        """This was the missing feature: edge conflicts must be detected."""
        nid_1, nid_2 = "n1", "n2"
        eid = "e1"
        nodes = [_make_node(id=nid_1), _make_node(id=nid_2)]

        base_edge = _make_edge(id=eid, source_id=nid_1, target_id=nid_2, weight=0.5)
        left_edge = _make_edge(id=eid, source_id=nid_1, target_id=nid_2, weight=0.3)
        right_edge = _make_edge(id=eid, source_id=nid_1, target_id=nid_2, weight=0.9)

        base = _make_document(deepcopy(nodes), [base_edge])
        left = _make_document(deepcopy(nodes), [left_edge])
        right = _make_document(deepcopy(nodes), [right_edge])

        result = diff_abhi_documents(left, right, input_path_a="a", input_path_b="b", base_document=base)
        assert result.diff_mode == "three_way"
        edge_conflicts = [c for c in result.conflict_records if c.object_type == "edge"]
        assert len(edge_conflicts) >= 1, "Edge conflicts should be detected in three-way diff"
        assert edge_conflicts[0].field == "weight"

    def test_three_way_diff_no_conflict_when_one_side_unchanged(self):
        """If only one side changed an edge, there's no conflict."""
        nid_1, nid_2 = "n1", "n2"
        eid = "e1"
        nodes = [_make_node(id=nid_1), _make_node(id=nid_2)]

        base_edge = _make_edge(id=eid, source_id=nid_1, target_id=nid_2, weight=0.5)
        left_edge = _make_edge(id=eid, source_id=nid_1, target_id=nid_2, weight=0.5)  # unchanged
        right_edge = _make_edge(id=eid, source_id=nid_1, target_id=nid_2, weight=0.9)  # changed

        base = _make_document(deepcopy(nodes), [base_edge])
        left = _make_document(deepcopy(nodes), [left_edge])
        right = _make_document(deepcopy(nodes), [right_edge])

        result = diff_abhi_documents(left, right, input_path_a="a", input_path_b="b", base_document=base)
        edge_conflicts = [c for c in result.conflict_records if c.object_type == "edge"]
        assert len(edge_conflicts) == 0, "No conflict when only one side changed"


# ===========================================================================
# Fix #4: field_overrides take precedence over type_overrides
# ===========================================================================


class TestStrategyOverridePrecedence:
    def test_field_override_wins_over_type_override(self):
        """A field-level override (more specific) must beat a type-level override (broad)."""
        node_id = "node-precedence"
        base = _make_node(id=node_id, content="base", node_type="fact", tags=["base-tag"])
        left = _make_node(id=node_id, content="left-content", node_type="fact", tags=["left-tag"])
        right = _make_node(id=node_id, content="right-content", node_type="fact", tags=["right-tag"])

        config = MergeStrategyConfig(
            default_strategy="prefer_right",
            type_overrides=[
                MergeStrategyTypeOverride(node_type="fact", strategy="prefer_right"),
            ],
            field_overrides=[
                MergeStrategyFieldOverride(field="content", strategy="prefer_left"),
            ],
        )

        conflicts: list[MergeConflictRecord] = []
        contradict_edges: list[dict] = []

        result = _apply_merge_strategy(
            node_id,
            "node",
            base,
            left,
            right,
            strategy="prefer_right",
            strategy_config=config,
            conflict_records=conflicts,
            contradict_edges=contradict_edges,
        )

        # The field override says content → prefer_left, even though
        # the type override says fact → prefer_right.
        assert result["content"] == "left-content", (
            "field_override for 'content' should win over type_override for 'fact'"
        )
        # Tags has no field-override, so the type_override (prefer_right) should apply.
        assert result["tags"] == ["right-tag"]

    def test_type_override_applies_when_no_field_override(self):
        """When there's no conflicting field override, type override takes effect."""
        node_id = "node-type-only"
        base = _make_node(id=node_id, content="base", node_type="decision")
        left = _make_node(id=node_id, content="left-decision", node_type="decision")
        right = _make_node(id=node_id, content="right-decision", node_type="decision")

        config = MergeStrategyConfig(
            default_strategy="prefer_right",
            type_overrides=[
                MergeStrategyTypeOverride(node_type="decision", strategy="prefer_left"),
            ],
            field_overrides=[],
        )

        conflicts: list[MergeConflictRecord] = []
        contradict_edges: list[dict] = []

        result = _apply_merge_strategy(
            node_id,
            "node",
            base,
            left,
            right,
            strategy="prefer_right",
            strategy_config=config,
            conflict_records=conflicts,
            contradict_edges=contradict_edges,
        )

        assert result["content"] == "left-decision"
