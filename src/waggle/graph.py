from __future__ import annotations

import heapq
import json
import sqlite3
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import networkx as nx
import numpy as np

from waggle.auth import generate_api_key, hash_api_key, verify_api_key
from waggle.context_bundle import build_context_bundle, build_query_summary, export_context_bundle_files
from waggle.embeddings import EmbeddingModel
from waggle.evidence import build_observation_evidence, merge_evidence_records, merge_validity_windows
from waggle.errors import AuthenticationError, ValidationFailure
from waggle.intelligence import (
    compatible_node_types,
    contains_conflicting_numbers,
    content_token_jaccard,
    detect_conflict_reason,
    extract_choice_entity,
    extract_conversation_candidates,
    infer_label,
    infer_node_type,
    infer_relationship,
    infer_temporal_hints,
    is_acronym_match,
    label_similarity,
    lexical_overlap,
    normalize_text,
    parse_since_value,
    score_node,
    split_atomic_items,
    summarize_topic,
    temporal_score_adjustment,
    tokenize_text,
    type_aware_dedup_threshold,
    within_time_window,
)
from waggle.markdown_vault import (
    evidence_from_lines,
    iter_vault_documents,
    render_node_document,
    slugify,
    vault_filename,
)
from waggle.models import (
    ApiKeyCreateResult,
    ApiKeyRecord,
    BackupResult,
    ConflictEntry,
    ConflictListResult,
    ConflictRecord,
    ConnectedNodeStat,
    ContextBundleExportResult,
    ContextScopeResult,
    ContextTimelineItem,
    Edge,
    EvidenceRecord,
    GraphDiffResult,
    GraphStats,
    ImportResult,
    MarkdownVaultExportResult,
    MarkdownVaultImportResult,
    Node,
    NodeHistoryResult,
    NodeStoreResult,
    NodeType,
    ObservationResult,
    PrimeContextResult,
    FusionHit,
    ReplayHit,
    RecentNodeStat,
    RelationType,
    SubgraphResult,
    TranscriptRecord,
    normalize_relationship,
    TenantRecord,
    TimelineResult,
    TopicCluster,
    TopicResult,
    utc_now,
)

SCHEMA_VERSION = 3


@dataclass(frozen=True)
class ExpansionMeta:
    via_relation: str
    from_node: str
    effective_priority: float


class _NeutralTemporalHints:
    """Neutral temporal hints for operations without query-driven time intent."""
    recency_mode: str = "none"
    time_window_start = None
    time_window_end = None


RELATION_SCORE_BOOST: dict[str, float] = {
    "contradicts": 0.15,
    "updates": 0.12,
    "depends_on": 0.08,
    "derived_from": 0.05,
    "part_of": 0.03,
    "relates_to": 0.00,
    "similar_to": -0.05,
    "seed": 0.00,
}

MUST_PAIR_RELATIONS: frozenset[str] = frozenset({
    "contradicts",
    "updates",
    "depends_on",
})


SCHEMA_VERSION = 3


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tenants (
    tenant_id TEXT PRIMARY KEY,
    name TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS api_keys (
    api_key_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    name TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    last_used_at TEXT DEFAULT NULL,
    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'local-default',
    agent_id TEXT DEFAULT '',
    project TEXT DEFAULT '',
    session_id TEXT DEFAULT '',
    label TEXT NOT NULL,
    content TEXT NOT NULL,
    node_type TEXT NOT NULL CHECK(
        node_type IN ('fact', 'entity', 'concept', 'preference', 'decision', 'question', 'note')
    ),
    tags TEXT DEFAULT '[]',
    embedding BLOB,
    source_prompt TEXT DEFAULT '',
    evidence_records TEXT DEFAULT '[]',
    valid_from TEXT DEFAULT NULL,
    valid_to TEXT DEFAULT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    access_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'local-default',
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relationship TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS transcript_records (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'local-default',
    agent_id TEXT DEFAULT '',
    project TEXT DEFAULT '',
    session_id TEXT DEFAULT '',
    observed_at TEXT NOT NULL,
    turn_index INTEGER NOT NULL DEFAULT 0,
    role TEXT NOT NULL DEFAULT '',
    transcript_text TEXT NOT NULL,
    embedding BLOB,
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_nodes_created ON nodes(created_at);
CREATE INDEX IF NOT EXISTS idx_nodes_tenant_type ON nodes(tenant_id, node_type);
CREATE INDEX IF NOT EXISTS idx_nodes_tenant_updated ON nodes(tenant_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_relationship ON edges(relationship);
CREATE INDEX IF NOT EXISTS idx_edges_tenant_relationship ON edges(tenant_id, relationship);
CREATE INDEX IF NOT EXISTS idx_transcripts_tenant_observed ON transcript_records(tenant_id, observed_at);
CREATE INDEX IF NOT EXISTS idx_transcripts_tenant_session_turn ON transcript_records(tenant_id, session_id, turn_index);
CREATE INDEX IF NOT EXISTS idx_api_keys_tenant ON api_keys(tenant_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
"""

RELATION_WEIGHTS: dict[str, float] = {
    "contradicts": 1.00,
    "updates": 0.95,
    "depends_on": 0.85,
    "derived_from": 0.75,
    "part_of": 0.70,
    "relates_to": 0.50,
    "similar_to": 0.30,
}


def _parse_datetime(raw: str) -> datetime:
    value = datetime.fromisoformat(raw)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _encode_evidence_records(records: list[EvidenceRecord]) -> str:
    return json.dumps([record.model_dump(mode="json") for record in records], sort_keys=True)


def _decode_evidence_records(raw: Any) -> list[EvidenceRecord]:
    if raw in (None, ""):
        return []
    if isinstance(raw, list):
        return [EvidenceRecord.model_validate(item) for item in raw]
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(decoded, list):
            return []
        return [EvidenceRecord.model_validate(item) for item in decoded]
    return []


def _encode_metadata(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, sort_keys=True)


def _decode_metadata(raw: Any) -> dict[str, Any]:
    if raw in (None, ""):
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _scope_matches(node: Node, *, agent_id: str = "", project: str = "", session_id: str = "") -> bool:
    normalized_agent = agent_id.strip().lower()
    normalized_project = project.strip().lower()
    normalized_session = session_id.strip().lower()
    if normalized_agent and node.agent_id.strip().lower() != normalized_agent:
        return False
    if normalized_session and node.session_id.strip().lower() != normalized_session:
        return False
    if normalized_project:
        project_tags = {str(tag).strip().lower() for tag in node.tags}
        if node.project.strip().lower() != normalized_project and normalized_project not in project_tags and f"project:{normalized_project}" not in project_tags:
            return False
    return True


def _retrieval_session_scope(*, agent_id: str = "", project: str = "", session_id: str = "") -> str:
    if session_id.strip() and (agent_id.strip() or project.strip()):
        return ""
    return session_id


def _merge_scope_value(existing: str, incoming: str) -> str:
    return existing.strip() or incoming.strip()


class MemoryGraph:
    """SQLite-backed graph memory with embedding-assisted retrieval."""

    def __init__(
        self,
        db_path: str | Path,
        embedding_model: EmbeddingModel,
        *,
        tenant_id: str = "local-default",
        dedup_similarity_threshold: float = 0.97,
        dedup_same_label_threshold: float = 0.9,
        export_dir: str | Path | None = None,
    ) -> None:
        self.db_path = Path(db_path).expanduser()
        self.embedding_model = embedding_model
        self.tenant_id = tenant_id.strip() or "local-default"
        self.dedup_similarity_threshold = dedup_similarity_threshold
        self.dedup_same_label_threshold = dedup_same_label_threshold
        self.export_dir = Path(export_dir).expanduser() if export_dir is not None else self.db_path.parent / "exports"
        self._lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_database(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(SCHEMA_SQL)
            self._migrate_legacy_schema(connection)
            created_at = utc_now().isoformat()
            connection.execute(
                """
                INSERT INTO tenants (tenant_id, name, status, created_at)
                VALUES (?, '', 'active', ?)
                ON CONFLICT(tenant_id) DO NOTHING
                """,
                (self.tenant_id, created_at),
            )

    def for_tenant(self, tenant_id: str) -> "MemoryGraph":
        clone = object.__new__(MemoryGraph)
        clone.db_path = self.db_path
        clone.embedding_model = self.embedding_model
        clone.tenant_id = tenant_id.strip() or "local-default"
        clone.dedup_similarity_threshold = self.dedup_similarity_threshold
        clone.dedup_same_label_threshold = self.dedup_same_label_threshold
        clone.export_dir = self.export_dir
        clone._lock = self._lock
        clone.ensure_tenant(clone.tenant_id)
        return clone

    def ensure_tenant(self, tenant_id: str, name: str = "") -> TenantRecord:
        normalized_tenant_id = tenant_id.strip()
        if not normalized_tenant_id:
            raise ValidationFailure("Tenant ID cannot be empty.")
        created_at = utc_now().isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO tenants (tenant_id, name, status, created_at)
                VALUES (?, ?, 'active', ?)
                ON CONFLICT(tenant_id) DO UPDATE SET name = CASE WHEN excluded.name != '' THEN excluded.name ELSE tenants.name END
                """,
                (normalized_tenant_id, name.strip(), created_at),
            )
            row = connection.execute(
                "SELECT tenant_id, name, status, created_at FROM tenants WHERE tenant_id = ?",
                (normalized_tenant_id,),
            ).fetchone()
        return TenantRecord(
            tenant_id=row["tenant_id"],
            name=row["name"] or "",
            status=row["status"],
            created_at=_parse_datetime(row["created_at"]),
        )

    def create_api_key(self, tenant_id: str, name: str = "") -> ApiKeyCreateResult:
        tenant = self.ensure_tenant(tenant_id)
        raw_api_key = generate_api_key()
        record = ApiKeyRecord(
            api_key_id=str(uuid4()),
            tenant_id=tenant.tenant_id,
            key_hash=hash_api_key(raw_api_key),
            name=name.strip(),
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO api_keys (api_key_id, tenant_id, key_hash, name, status, created_at, last_used_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.api_key_id,
                    record.tenant_id,
                    record.key_hash,
                    record.name,
                    record.status,
                    record.created_at.isoformat(),
                    None,
                ),
            )
        return ApiKeyCreateResult(record=record, raw_api_key=raw_api_key)

    def list_api_keys(self, tenant_id: str) -> list[ApiKeyRecord]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT api_key_id, tenant_id, key_hash, name, status, created_at, last_used_at
                FROM api_keys
                WHERE tenant_id = ?
                ORDER BY created_at DESC
                """,
                (tenant_id,),
            ).fetchall()
        return [
            ApiKeyRecord(
                api_key_id=row["api_key_id"],
                tenant_id=row["tenant_id"],
                key_hash=row["key_hash"],
                name=row["name"] or "",
                status=row["status"],
                created_at=_parse_datetime(row["created_at"]),
                last_used_at=_parse_datetime(row["last_used_at"]) if row["last_used_at"] else None,
            )
            for row in rows
        ]

    def revoke_api_key(self, api_key_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE api_keys SET status = 'revoked' WHERE api_key_id = ?",
                (api_key_id,),
            )

    def authenticate_api_key(self, raw_api_key: str) -> ApiKeyRecord:
        key_hash = hash_api_key(raw_api_key)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT api_key_id, tenant_id, key_hash, name, status, created_at, last_used_at
                FROM api_keys
                WHERE key_hash = ?
                LIMIT 1
                """,
                (key_hash,),
            ).fetchone()
            if row is None or not verify_api_key(raw_api_key, row["key_hash"]):
                raise AuthenticationError("Invalid API key.")
            connection.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE api_key_id = ?",
                (utc_now().isoformat(), row["api_key_id"]),
            )
        return ApiKeyRecord(
            api_key_id=row["api_key_id"],
            tenant_id=row["tenant_id"],
            key_hash=row["key_hash"],
            name=row["name"] or "",
            status=row["status"],
            created_at=_parse_datetime(row["created_at"]),
            last_used_at=utc_now(),
        )

    def _migrate_legacy_schema(self, connection: sqlite3.Connection) -> None:
        node_columns = {row["name"] for row in connection.execute("PRAGMA table_info(nodes)").fetchall()}
        edge_columns = {row["name"] for row in connection.execute("PRAGMA table_info(edges)").fetchall()}
        if "tenant_id" not in node_columns:
            connection.execute(
                f"ALTER TABLE nodes ADD COLUMN tenant_id TEXT NOT NULL DEFAULT '{self.tenant_id}'"
            )
            connection.execute("UPDATE nodes SET tenant_id = ? WHERE tenant_id = ''", (self.tenant_id,))
        if "evidence_records" not in node_columns:
            connection.execute("ALTER TABLE nodes ADD COLUMN evidence_records TEXT DEFAULT '[]'")
        if "valid_from" not in node_columns:
            connection.execute("ALTER TABLE nodes ADD COLUMN valid_from TEXT DEFAULT NULL")
        if "valid_to" not in node_columns:
            connection.execute("ALTER TABLE nodes ADD COLUMN valid_to TEXT DEFAULT NULL")
        if "agent_id" not in node_columns:
            connection.execute("ALTER TABLE nodes ADD COLUMN agent_id TEXT DEFAULT ''")
        if "project" not in node_columns:
            connection.execute("ALTER TABLE nodes ADD COLUMN project TEXT DEFAULT ''")
        if "session_id" not in node_columns:
            connection.execute("ALTER TABLE nodes ADD COLUMN session_id TEXT DEFAULT ''")
        if "tenant_id" not in edge_columns:
            connection.execute(
                f"ALTER TABLE edges ADD COLUMN tenant_id TEXT NOT NULL DEFAULT '{self.tenant_id}'"
            )
            connection.execute("UPDATE edges SET tenant_id = ? WHERE tenant_id = ''", (self.tenant_id,))
        connection.execute(
            """
            INSERT OR IGNORE INTO schema_migrations (version, applied_at)
            VALUES (?, ?)
            """,
            (SCHEMA_VERSION, utc_now().isoformat()),
        )

    def add_node(
        self,
        *,
        label: str,
        content: str,
        node_type: NodeType,
        tags: list[str] | None = None,
        source_prompt: str = "",
        agent_id: str = "",
        project: str = "",
        session_id: str = "",
        evidence_records: list[EvidenceRecord] | None = None,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
    ) -> NodeStoreResult:
        node = Node(
            tenant_id=self.tenant_id,
            agent_id=agent_id,
            project=project,
            session_id=session_id,
            label=label,
            content=content,
            node_type=node_type,
            tags=tags or [],
            source_prompt=source_prompt,
            evidence_records=evidence_records or [],
            valid_from=valid_from,
            valid_to=valid_to,
        )
        embedding = self.embedding_model.embed(node.content)

        with self._lock, self._connect() as connection:
            duplicate = self._find_duplicate_node(connection, node=node, embedding=embedding)
            if duplicate is not None:
                existing_node, dedup_reason, similarity = duplicate
                merged_node = self._merge_duplicate_node(
                    connection,
                    existing_node=existing_node,
                    incoming_node=node,
                )
                return NodeStoreResult(
                    node=merged_node,
                    created=False,
                    dedup_reason=dedup_reason,
                    similarity=similarity,
                )

            connection.execute(
                """
                INSERT INTO nodes (
                    id, tenant_id, agent_id, project, session_id, label, content, node_type, tags, embedding,
                    source_prompt, evidence_records, valid_from, valid_to,
                    created_at, updated_at, access_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node.id,
                    node.tenant_id,
                    node.agent_id,
                    node.project,
                    node.session_id,
                    node.label,
                    node.content,
                    node.node_type.value,
                    json.dumps(node.tags),
                    self.embedding_model.to_bytes(embedding),
                    node.source_prompt,
                    _encode_evidence_records(node.evidence_records),
                    node.valid_from.isoformat() if node.valid_from is not None else None,
                    node.valid_to.isoformat() if node.valid_to is not None else None,
                    node.created_at.isoformat(),
                    node.updated_at.isoformat(),
                    node.access_count,
                ),
            )
            conflicts = self._register_conflicts(connection, node)
        return NodeStoreResult(node=node, created=True, conflicts=conflicts)

    def add_edge(
        self,
        *,
        source_id: str,
        target_id: str,
        relationship: str | RelationType,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
        ) -> Edge:
        edge = Edge(
            tenant_id=self.tenant_id,
            source_id=source_id,
            target_id=target_id,
            relationship=relationship,
            weight=weight,
            metadata=metadata or {},
        )

        with self._lock, self._connect() as connection:
            self._require_node(connection, edge.source_id)
            self._require_node(connection, edge.target_id)
            existing_edge = self._find_existing_edge(
                connection,
                source_id=edge.source_id,
                target_id=edge.target_id,
                relationship=edge.relationship,
            )
            if existing_edge is not None:
                return existing_edge
            connection.execute(
                """
                INSERT INTO edges (
                    id, tenant_id, source_id, target_id, relationship, weight, metadata, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    edge.id,
                    edge.tenant_id,
                    edge.source_id,
                    edge.target_id,
                    edge.relationship,
                    edge.weight,
                    json.dumps(edge.metadata),
                    edge.created_at.isoformat(),
                ),
            )
        return edge

    def get_node(self, node_id: str) -> Node:
        with self._lock, self._connect() as connection:
            row = self._fetch_node_row(connection, node_id)
            if row is None:
                raise ValueError(f"Node not found: {node_id}")
            return self._row_to_node(row)

    def get_node_history(self, *, node_id: str, max_depth: int = 2) -> NodeHistoryResult:
        node = self.get_node(node_id)
        related = self.get_related(node_id=node_id, max_depth=max_depth)
        related_nodes = [item for item in related.nodes if item.id != node_id]
        return NodeHistoryResult(node=node, related_nodes=related_nodes, edges=related.edges)

    def timeline(
        self,
        *,
        node_id: str = "",
        query: str = "",
        limit: int = 25,
        max_depth: int = 2,
        include_evidence: bool = True,
    ) -> TimelineResult:
        if limit < 1:
            raise ValueError("limit must be at least 1.")
        if max_depth < 0:
            raise ValueError("max_depth cannot be negative.")
        if node_id.strip() and query.strip():
            raise ValueError("Provide either node_id or query, not both.")

        if node_id.strip():
            related = self.get_related(node_id=node_id, max_depth=max_depth)
            nodes = related.nodes
            edges = related.edges
            scope = f"node:{node_id.strip()}"
        elif query.strip():
            subgraph = self.query(query=query, max_nodes=max(limit, 10), max_depth=max_depth)
            nodes = subgraph.nodes
            edges = subgraph.edges
            scope = f"query:{query.strip()}"
        else:
            with self._lock, self._connect() as connection:
                nodes = self.list_recent_nodes(limit=max(limit, 10))
                edges = self._fetch_edges_for_nodes(connection, [node.id for node in nodes])
            scope = "tenant"

        items = self._build_timeline_items(
            nodes=nodes,
            edges=edges,
            include_evidence=include_evidence,
            limit=limit,
        )
        return TimelineResult(scope=scope, items=items)

    def list_conflicts(
        self,
        *,
        include_resolved: bool = False,
        limit: int = 25,
    ) -> ConflictListResult:
        if limit < 1:
            raise ValueError("limit must be at least 1.")

        with self._lock, self._connect() as connection:
            edge_rows = connection.execute(
                """
                SELECT id, source_id, target_id, relationship, weight, metadata, created_at, tenant_id
                FROM edges
                WHERE tenant_id = ?
                  AND relationship IN (?, ?)
                ORDER BY created_at DESC
                """,
                (self.tenant_id, RelationType.CONTRADICTS.value, RelationType.UPDATES.value),
            ).fetchall()
            edges = [self._row_to_edge(row) for row in edge_rows]
            entries = self._build_conflict_entries(
                connection,
                edges=edges,
                include_resolved=include_resolved,
                limit=limit,
            )
        return ConflictListResult(conflicts=entries, include_resolved=include_resolved)

    def resolve_conflict(
        self,
        *,
        edge_id: str,
        resolution_note: str = "",
    ) -> ConflictEntry:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, source_id, target_id, relationship, weight, metadata, created_at, tenant_id
                FROM edges
                WHERE tenant_id = ? AND id = ?
                LIMIT 1
                """,
                (self.tenant_id, edge_id),
            ).fetchone()
            if row is None:
                raise ValueError(f"Conflict edge not found: {edge_id}")
            edge = self._row_to_edge(row)
            if edge.relationship not in {RelationType.CONTRADICTS.value, RelationType.UPDATES.value}:
                raise ValueError("Only contradicts or updates edges can be resolved.")

            metadata = dict(edge.metadata)
            metadata["resolved"] = True
            metadata["resolved_at"] = utc_now().isoformat()
            if resolution_note.strip():
                metadata["resolution_note"] = resolution_note.strip()

            connection.execute(
                """
                UPDATE edges
                SET metadata = ?
                WHERE tenant_id = ? AND id = ?
                """,
                (json.dumps(metadata, sort_keys=True), self.tenant_id, edge_id),
            )
            updated_edge = Edge(
                id=edge.id,
                tenant_id=edge.tenant_id,
                source_id=edge.source_id,
                target_id=edge.target_id,
                relationship=edge.relationship,
                weight=edge.weight,
                metadata=metadata,
                created_at=edge.created_at,
            )
            entry = self._build_conflict_entries(
                connection,
                edges=[updated_edge],
                include_resolved=True,
                limit=1,
            )
        if not entry:
            raise ValueError(f"Resolved conflict could not be loaded: {edge_id}")
        return entry[0]

    def query(
        self,
        *,
        query: str,
        max_nodes: int = 20,
        max_depth: int = 2,
        agent_id: str = "",
        project: str = "",
        session_id: str = "",
        retrieval_mode: str = "graph",
    ) -> SubgraphResult:
        query_text = query.strip()
        if not query_text:
            raise ValueError("Query cannot be empty.")
        if max_nodes < 1:
            raise ValueError("max_nodes must be at least 1.")
        if max_depth < 0:
            raise ValueError("max_depth cannot be negative.")
        normalized_mode = retrieval_mode.strip().lower()
        if normalized_mode not in {"graph", "replay", "fusion"}:
            raise ValueError("retrieval_mode must be one of: graph, replay, fusion.")

        graph_result = (
            self._query_graph_only(
                query=query_text,
                max_nodes=max_nodes,
                max_depth=max_depth,
                agent_id=agent_id,
                project=project,
                session_id=session_id,
            )
            if normalized_mode in {"graph", "fusion"}
            else None
        )
        replay_hits = (
            self._query_replay_hits(
                query=query_text,
                max_hits=max_nodes,
                agent_id=agent_id,
                project=project,
                session_id=session_id,
            )
            if normalized_mode in {"replay", "fusion"}
            else []
        )
        if normalized_mode == "graph":
            graph_result.retrieval_mode = "graph"
            return graph_result
        if normalized_mode == "replay":
            return SubgraphResult(
                replay_hits=replay_hits,
                retrieval_mode="replay",
                query=query_text,
                total_nodes_in_graph=graph_result.total_nodes_in_graph if graph_result is not None else 0,
            )
        fusion_hits = self._build_fusion_hits(graph_result or SubgraphResult(query=query_text), replay_hits)
        return SubgraphResult(
            nodes=graph_result.nodes if graph_result is not None else [],
            edges=graph_result.edges if graph_result is not None else [],
            replay_hits=replay_hits,
            fusion_hits=fusion_hits[:max_nodes],
            retrieval_mode="fusion",
            query=query_text,
            total_nodes_in_graph=graph_result.total_nodes_in_graph if graph_result is not None else 0,
        )

    def _query_graph_only(
        self,
        *,
        query: str,
        max_nodes: int,
        max_depth: int,
        agent_id: str,
        project: str,
        session_id: str,
    ) -> SubgraphResult:
        with self._lock, self._connect() as connection:
            temporal_hints = infer_temporal_hints(query)
            node_rows = connection.execute(
                """
                SELECT id, agent_id, project, session_id, label, content, node_type, tags, source_prompt,
                       evidence_records, valid_from, valid_to, created_at, updated_at, access_count, embedding, tenant_id
                FROM nodes
                WHERE tenant_id = ? AND embedding IS NOT NULL
                """,
                (self.tenant_id,),
            ).fetchall()
            total_nodes = len(node_rows)
            if total_nodes == 0:
                return SubgraphResult(query=query, total_nodes_in_graph=0)

            def collect_scoped_nodes(active_session_id: str) -> tuple[dict[str, Node], dict[str, np.ndarray]]:
                scoped_nodes: dict[str, Node] = {}
                scoped_embeddings: dict[str, np.ndarray] = {}
                for row in node_rows:
                    node = self._row_to_node(row)
                    if not _scope_matches(node, agent_id=agent_id, project=project, session_id=active_session_id):
                        continue
                    scoped_nodes[node.id] = node
                    scoped_embeddings[node.id] = self.embedding_model.from_bytes(row["embedding"])
                return scoped_nodes, scoped_embeddings

            active_session_id = _retrieval_session_scope(
                agent_id=agent_id,
                project=project,
                session_id=session_id,
            )
            nodes_by_id, embeddings_by_id = collect_scoped_nodes(active_session_id)

            if not nodes_by_id:
                return SubgraphResult(query=query, total_nodes_in_graph=total_nodes)

            query_embedding = self.embedding_model.embed(query)
            similarity_by_id = {
                node_id: max(self.embedding_model.cosine_similarity(query_embedding, embedding), 0.0)
                for node_id, embedding in embeddings_by_id.items()
            }
            lexical_by_id = {
                node_id: lexical_overlap(query, node.label, node.content)
                for node_id, node in nodes_by_id.items()
            }

            seed_count = min(total_nodes, max(1, max_nodes // 2))
            seed_candidates = [
                (
                    node_id,
                    (0.72 * similarity_by_id.get(node_id, 0.0))
                    + (0.28 * lexical_by_id.get(node_id, 0.0)),
                    self._seed_temporal_order(nodes_by_id[node_id], temporal_hints),
                )
                for node_id in nodes_by_id
            ]
            if temporal_hints.recency_mode in {"latest", "oldest"}:
                ranked_seed_ids = [
                    item[0]
                    for item in sorted(
                        seed_candidates,
                        key=lambda item: (item[2], -item[1], nodes_by_id[item[0]].label.lower()),
                    )[:seed_count]
                ]
            else:
                ranked_seed_ids = [
                    item[0]
                    for item in sorted(
                        seed_candidates,
                        key=lambda item: (-item[1], item[2], nodes_by_id[item[0]].label.lower()),
                    )[:seed_count]
                ]

            graph = self._load_graph(connection, node_ids=nodes_by_id.keys())
            expanded_depths, expansion_metadata = self._expand_node_depths_with_context(
                graph, ranked_seed_ids, max_depth
            )
            candidate_nodes = [nodes_by_id[node_id] for node_id in expanded_depths]
            temporal_candidates = [node for node in candidate_nodes if within_time_window(node, temporal_hints)]
            if temporal_candidates:
                candidate_nodes = temporal_candidates

            max_access = max((node.access_count for node in candidate_nodes), default=0)
            degree_by_id = dict(graph.degree(expanded_depths.keys()))
            max_degree = max(degree_by_id.values(), default=0)
            scored_nodes = self._sort_scored_nodes(
                candidate_nodes,
                temporal_hints=temporal_hints,
                similarity_by_id=similarity_by_id,
                lexical_by_id=lexical_by_id,
                degree_by_id=degree_by_id,
                max_access=max_access,
                max_degree=max_degree,
                max_depth=max_depth,
                expanded_depths=expanded_depths,
                expansion_metadata=expansion_metadata,
            )
            selected_nodes = scored_nodes[:max_nodes]
            candidate_pool = {node.id: node for node in candidate_nodes}
            selected_nodes = self._ensure_support_coverage(selected_nodes, candidate_pool, graph, max_nodes)
            selected_ids = [node.id for node in selected_nodes]

            edges = self._fetch_edges_for_nodes(connection, selected_ids)
            self._increment_access_counts(connection, selected_ids)
            for node in selected_nodes:
                node.access_count += 1

            return SubgraphResult(
                nodes=selected_nodes,
                edges=edges,
                retrieval_mode="graph",
                query=query,
                total_nodes_in_graph=total_nodes,
            )

    def _query_replay_hits(
        self,
        *,
        query: str,
        max_hits: int,
        agent_id: str,
        project: str,
        session_id: str,
    ) -> list[ReplayHit]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, tenant_id, agent_id, project, session_id, observed_at, turn_index, role, transcript_text, embedding, metadata
                FROM transcript_records
                WHERE tenant_id = ? AND embedding IS NOT NULL
                ORDER BY observed_at DESC, turn_index DESC
                """,
                (self.tenant_id,),
            ).fetchall()
        if not rows:
            return []
        query_embedding = self.embedding_model.embed(query)
        temporal_hints = infer_temporal_hints(query)
        timestamps = np.asarray([_parse_datetime(row["observed_at"]).timestamp() for row in rows], dtype=np.float64)
        max_timestamp = float(np.max(timestamps))
        min_timestamp = float(np.min(timestamps))
        span = max(max_timestamp - min_timestamp, 1.0)

        def build_hits(active_session_id: str) -> list[tuple[float, ReplayHit]]:
            hits: list[tuple[float, ReplayHit]] = []
            for row, raw_timestamp in zip(rows, timestamps, strict=True):
                record = self._row_to_transcript_record(row)
                if not self._transcript_scope_matches(record, agent_id=agent_id, project=project, session_id=active_session_id):
                    continue
                embedding = self.embedding_model.from_bytes(row["embedding"])
                semantic_score = max(self.embedding_model.cosine_similarity(query_embedding, embedding), 0.0)
                lexical_score = lexical_overlap(query, record.role, record.transcript_text)
                temporal_score = 0.0
                if temporal_hints.recency_mode == "latest":
                    temporal_score = float((raw_timestamp - min_timestamp) / span)
                elif temporal_hints.recency_mode == "oldest":
                    temporal_score = float((max_timestamp - raw_timestamp) / span)
                role_score = 1.0 if record.role == "user" else 0.8
                score = (0.6 * semantic_score) + (0.2 * lexical_score) + (0.1 * temporal_score) + (0.1 * role_score)
                hits.append(
                    (
                        score,
                        ReplayHit(
                            score=score,
                            session_id=record.session_id,
                            turn_index=record.turn_index,
                            role=record.role,
                            transcript_text=record.transcript_text,
                            transcript_snippet=record.transcript_text[:280],
                            observed_at=record.observed_at,
                        ),
                    )
                )
            return hits

        active_session_id = _retrieval_session_scope(
            agent_id=agent_id,
            project=project,
            session_id=session_id,
        )
        hits = build_hits(active_session_id)
        return [item[1] for item in sorted(hits, key=lambda item: (-item[0], -item[1].observed_at.timestamp(), item[1].turn_index))[:max_hits]]

    def _build_fusion_hits(self, graph_result: SubgraphResult, replay_hits: list[ReplayHit]) -> list[FusionHit]:
        rrf_k = 60.0
        replay_by_session = {hit.session_id for hit in replay_hits if hit.session_id}
        graph_edge_map: dict[str, list[dict[str, Any]]] = {}
        graph_nodes_by_session = {node.session_id: node for node in graph_result.nodes if node.session_id}
        combined: dict[str, FusionHit] = {}

        for edge in graph_result.edges:
            payload = {
                "id": edge.id,
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "relationship": edge.relationship,
                "weight": edge.weight,
            }
            graph_edge_map.setdefault(edge.source_id, []).append(payload)
            graph_edge_map.setdefault(edge.target_id, []).append(payload)

        for index, node in enumerate(graph_result.nodes, start=1):
            source_lane = "both" if node.session_id and node.session_id in replay_by_session else "graph"
            combined[f"graph:{node.id}"] = FusionHit(
                content=node.content,
                score=1.0 / (rrf_k + index),
                source_lane=source_lane,
                graph_rank=index,
                replay_rank=None,
                fused_rank=0,
                node_id=node.id,
                node_type=node.node_type.value,
                edges=graph_edge_map.get(node.id, []),
                session_id=node.session_id or None,
            )

        for index, hit in enumerate(replay_hits, start=1):
            contribution = 1.0 / (rrf_k + index)
            matching_graph = graph_nodes_by_session.get(hit.session_id) if hit.session_id else None
            if matching_graph is not None:
                existing = combined.get(f"graph:{matching_graph.id}")
                if existing is not None:
                    existing.score += contribution
                    existing.source_lane = "both"
                    existing.replay_rank = index
                    existing.session_id = hit.session_id or None
                    continue
                key = f"both:{matching_graph.id}:{hit.session_id}:{hit.turn_index}"
                source_lane = "both"
            else:
                key = f"replay:{hit.session_id}:{hit.turn_index}:{index}"
                source_lane = "replay"
            combined[key] = FusionHit(
                content=hit.transcript_text,
                score=contribution,
                source_lane=source_lane,
                graph_rank=None,
                replay_rank=index,
                fused_rank=0,
                session_id=hit.session_id or None,
                transcript_snippet=hit.transcript_snippet,
                turn_index=hit.turn_index,
            )

        ordered = sorted(
            combined.values(),
            key=lambda item: (-item.score, 0 if item.source_lane in {"both", "graph"} else 1, item.content.lower()),
        )
        for index, item in enumerate(ordered, start=1):
            item.fused_rank = index
        return ordered

    def get_related(self, *, node_id: str, max_depth: int = 2) -> SubgraphResult:
        if max_depth < 0:
            raise ValueError("max_depth cannot be negative.")

        with self._lock, self._connect() as connection:
            self._require_node(connection, node_id)
            node_rows = connection.execute(
                """
                SELECT id, agent_id, project, session_id, label, content, node_type, tags, source_prompt, evidence_records, valid_from, valid_to,
                       created_at, updated_at, access_count, tenant_id
                FROM nodes
                WHERE tenant_id = ?
                """
            , (self.tenant_id,)).fetchall()
            nodes_by_id = {row["id"]: self._row_to_node(row) for row in node_rows}
            graph = self._load_graph(connection, node_ids=nodes_by_id.keys())
            related_ids = list(self._expand_node_depths(graph, [node_id], max_depth))

            ordered_nodes: list[Node] = []
            seen: set[str] = set()
            for related_id in [node_id, *related_ids]:
                if related_id in seen:
                    continue
                seen.add(related_id)
                ordered_nodes.append(nodes_by_id[related_id])

            edges = self._fetch_edges_for_nodes(connection, [node.id for node in ordered_nodes])
            self._increment_access_counts(connection, [node.id for node in ordered_nodes])
            for node in ordered_nodes:
                node.access_count += 1

            return SubgraphResult(
                nodes=ordered_nodes,
                edges=edges,
                query=f"related:{node_id}",
                total_nodes_in_graph=len(nodes_by_id),
            )

    def update_node(
        self,
        *,
        node_id: str,
        content: str | None = None,
        label: str | None = None,
        tags: list[str] | None = None,
        agent_id: str | None = None,
        project: str | None = None,
        session_id: str | None = None,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        evidence_records: list[EvidenceRecord] | None = None,
    ) -> Node:
        if (
            content is None
            and label is None
            and tags is None
            and agent_id is None
            and project is None
            and session_id is None
            and valid_from is None
            and valid_to is None
            and evidence_records is None
        ):
            raise ValueError("At least one field must be provided for update.")

        with self._lock, self._connect() as connection:
            row = self._fetch_node_row(connection, node_id)
            if row is None:
                raise ValueError(f"Node not found: {node_id}")

            node = self._row_to_node(row)
            updated_label = label if label is not None else node.label
            updated_content = content if content is not None else node.content
            updated_tags = tags if tags is not None else node.tags
            updated_at = utc_now()
            embedding_bytes = row["embedding"]
            if content is not None:
                embedding_bytes = self.embedding_model.to_bytes(self.embedding_model.embed(updated_content))

            updated_node = Node(
                id=node.id,
                tenant_id=node.tenant_id,
                agent_id=agent_id if agent_id is not None else node.agent_id,
                project=project if project is not None else node.project,
                session_id=session_id if session_id is not None else node.session_id,
                label=updated_label,
                content=updated_content,
                node_type=node.node_type,
                tags=updated_tags,
                source_prompt=node.source_prompt,
                evidence_records=evidence_records if evidence_records is not None else node.evidence_records,
                valid_from=valid_from if valid_from is not None else node.valid_from,
                valid_to=valid_to if valid_to is not None else node.valid_to,
                created_at=node.created_at,
                updated_at=updated_at,
                access_count=node.access_count,
            )

            connection.execute(
                """
                UPDATE nodes
                SET label = ?, content = ?, tags = ?, embedding = ?, updated_at = ?,
                    agent_id = ?, project = ?, session_id = ?,
                    evidence_records = ?, valid_from = ?, valid_to = ?
                WHERE id = ? AND tenant_id = ?
                """,
                (
                    updated_node.label,
                    updated_node.content,
                    json.dumps(updated_node.tags),
                    embedding_bytes,
                    updated_node.updated_at.isoformat(),
                    updated_node.agent_id,
                    updated_node.project,
                    updated_node.session_id,
                    _encode_evidence_records(updated_node.evidence_records),
                    updated_node.valid_from.isoformat() if updated_node.valid_from is not None else None,
                    updated_node.valid_to.isoformat() if updated_node.valid_to is not None else None,
                    updated_node.id,
                    self.tenant_id,
                ),
            )
            return updated_node

    def delete_node(self, *, node_id: str) -> Node:
        with self._lock, self._connect() as connection:
            row = self._fetch_node_row(connection, node_id)
            if row is None:
                raise ValueError(f"Node not found: {node_id}")
            node = self._row_to_node(row)
            connection.execute("DELETE FROM nodes WHERE id = ? AND tenant_id = ?", (node_id, self.tenant_id))
            return node

    def list_recent_nodes(
        self,
        limit: int = 10,
        *,
        agent_id: str = "",
        project: str = "",
        session_id: str = "",
    ) -> list[Node]:
        limit = max(1, limit)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, agent_id, project, session_id, label, content, node_type, tags, source_prompt, evidence_records, valid_from, valid_to,
                       created_at, updated_at, access_count, tenant_id
                FROM nodes
                WHERE tenant_id = ?
                ORDER BY updated_at DESC, created_at DESC
                """,
                (self.tenant_id,),
            ).fetchall()
            selected: list[Node] = []
            for row in rows:
                node = self._row_to_node(row)
                if not _scope_matches(node, agent_id=agent_id, project=project, session_id=session_id):
                    continue
                selected.append(node)
                if len(selected) >= limit:
                    break
            return selected

    def list_context_scopes(self) -> ContextScopeResult:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT agent_id, project, session_id
                FROM nodes
                WHERE tenant_id = ?
                """,
                (self.tenant_id,),
            ).fetchall()
        agent_ids = sorted({str(row["agent_id"]).strip() for row in rows if str(row["agent_id"]).strip()})
        projects = sorted({str(row["project"]).strip() for row in rows if str(row["project"]).strip()})
        session_ids = sorted({str(row["session_id"]).strip() for row in rows if str(row["session_id"]).strip()})
        return ContextScopeResult(agent_ids=agent_ids, projects=projects, session_ids=session_ids)

    def get_stats(self) -> GraphStats:
        with self._lock, self._connect() as connection:
            total_nodes = int(
                connection.execute("SELECT COUNT(*) FROM nodes WHERE tenant_id = ?", (self.tenant_id,)).fetchone()[0]
            )
            total_edges = int(
                connection.execute("SELECT COUNT(*) FROM edges WHERE tenant_id = ?", (self.tenant_id,)).fetchone()[0]
            )

            counts = {
                node_type.value: 0
                for node_type in NodeType
            }
            for row in connection.execute(
                "SELECT node_type, COUNT(*) AS count FROM nodes WHERE tenant_id = ? GROUP BY node_type",
                (self.tenant_id,),
            ).fetchall():
                counts[str(row["node_type"])] = int(row["count"])

            most_connected_rows = connection.execute(
                """
                SELECT n.id, n.label, n.node_type,
                       COUNT(e.id) AS connection_count
                FROM nodes AS n
                LEFT JOIN edges AS e
                    ON (n.id = e.source_id OR n.id = e.target_id) AND e.tenant_id = ?
                WHERE n.tenant_id = ?
                GROUP BY n.id
                ORDER BY connection_count DESC, n.updated_at DESC
                LIMIT 5
                """
            , (self.tenant_id, self.tenant_id)).fetchall()

            most_recent_rows = connection.execute(
                """
                SELECT id, label, node_type, updated_at
                FROM nodes
                WHERE tenant_id = ?
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 5
                """
            , (self.tenant_id,)).fetchall()

            return GraphStats(
                total_nodes=total_nodes,
                total_edges=total_edges,
                node_type_breakdown=counts,
                most_connected_nodes=[
                    ConnectedNodeStat(
                        id=row["id"],
                        label=row["label"],
                        node_type=NodeType(row["node_type"]),
                        connection_count=int(row["connection_count"]),
                    )
                    for row in most_connected_rows
                ],
                most_recent_nodes=[
                    RecentNodeStat(
                        id=row["id"],
                        label=row["label"],
                        node_type=NodeType(row["node_type"]),
                        updated_at=_parse_datetime(row["updated_at"]),
                    )
                    for row in most_recent_rows
                ],
            )

    def export_graph_html(
        self,
        *,
        output_path: str | Path | None = None,
        include_physics: bool = True,
    ) -> Path:
        try:
            from pyvis.network import Network
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pyvis is not installed. Install the project dependencies again.") from exc

        with self._lock, self._connect() as connection:
            node_rows = connection.execute(
                """
                SELECT id, label, content, node_type, tags, source_prompt,
                       created_at, updated_at, access_count
                FROM nodes
                WHERE tenant_id = ?
                ORDER BY updated_at DESC, created_at DESC
                """
            , (self.tenant_id,)).fetchall()
            edge_rows = connection.execute(
                """
                SELECT id, source_id, target_id, relationship, weight, metadata, created_at
                FROM edges
                WHERE tenant_id = ?
                ORDER BY created_at ASC
                """
            , (self.tenant_id,)).fetchall()

        if output_path is None:
            self.export_dir.mkdir(parents=True, exist_ok=True)
            timestamp = utc_now().strftime("%Y%m%d-%H%M%S")
            destination = self.export_dir / f"waggle-{timestamp}.html"
        else:
            destination = Path(output_path).expanduser()
            destination.parent.mkdir(parents=True, exist_ok=True)

        network = Network(
            height="800px",
            width="100%",
            directed=True,
            bgcolor="#0f172a",
            font_color="#e2e8f0",
        )
        network.barnes_hut()
        if not include_physics:
            network.toggle_physics(False)

        palette = {
            NodeType.FACT: "#38bdf8",
            NodeType.ENTITY: "#34d399",
            NodeType.CONCEPT: "#fbbf24",
            NodeType.PREFERENCE: "#fb7185",
            NodeType.DECISION: "#c084fc",
            NodeType.QUESTION: "#f97316",
            NodeType.NOTE: "#94a3b8",
        }

        nodes = [self._row_to_node(row) for row in node_rows]
        edges = [self._row_to_edge(row) for row in edge_rows]

        for node in nodes:
            title_lines = [
                f"<b>{node.label}</b>",
                f"Type: {node.node_type.value}",
                f"Created: {node.created_at.isoformat()}",
                f"Updated: {node.updated_at.isoformat()}",
                f"Access Count: {node.access_count}",
                "",
                node.content,
            ]
            if node.tags:
                title_lines.insert(4, f"Tags: {', '.join(node.tags)}")

            network.add_node(
                node.id,
                label=node.label,
                title="<br>".join(title_lines),
                color=palette[node.node_type],
                shape="dot",
                size=18 + min(node.access_count, 8) * 2,
            )

        for edge in edges:
            network.add_edge(
                edge.source_id,
                edge.target_id,
                label=edge.relationship,
                title=f"weight={edge.weight}",
                value=max(edge.weight, 0.1),
                arrows="to",
            )

        destination.write_text(network.generate_html(notebook=False), encoding="utf-8")
        return destination

    def export_graph_backup(self, *, output_path: str | Path | None = None) -> BackupResult:
        with self._lock, self._connect() as connection:
            snapshot = self._build_backup_snapshot(connection)

        if output_path is None:
            self.export_dir.mkdir(parents=True, exist_ok=True)
            timestamp = utc_now().strftime("%Y%m%d-%H%M%S")
            destination = self.export_dir / f"waggle-backup-{timestamp}.json"
        else:
            destination = Path(output_path).expanduser()
            destination.parent.mkdir(parents=True, exist_ok=True)

        destination.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        return BackupResult(
            output_path=str(destination),
            tenant_id=self.tenant_id,
            schema_version=SCHEMA_VERSION,
            node_count=len(snapshot["nodes"]),
            edge_count=len(snapshot["edges"]),
        )

    def export_context_bundle(
        self,
        *,
        mode: str = "prime",
        query: str = "",
        project: str = "",
        agent_id: str = "",
        session_id: str = "",
        max_nodes: int = 25,
        max_depth: int = 2,
        retrieval_mode: str = "graph",
        format: str = "both",
        output_path: str | Path | None = None,
        include_edges: bool = True,
        include_timestamps: bool = True,
        include_source_prompt: bool = False,
        audience: str = "llm",
    ) -> ContextBundleExportResult:
        normalized_mode = mode.strip().lower()
        normalized_format = format.strip().lower()
        normalized_audience = audience.strip().lower()
        normalized_retrieval_mode = retrieval_mode.strip().lower()
        if normalized_mode not in {"prime", "query", "graph"}:
            raise ValidationFailure("mode must be one of: prime, query, graph.")
        if normalized_format not in {"markdown", "json", "both"}:
            raise ValidationFailure("format must be one of: markdown, json, both.")
        if normalized_audience not in {"llm", "human"}:
            raise ValidationFailure("audience must be one of: llm, human.")
        if normalized_retrieval_mode not in {"graph", "replay", "fusion"}:
            raise ValidationFailure("retrieval_mode must be one of: graph, replay, fusion.")
        if normalized_mode == "query" and not query.strip():
            raise ValidationFailure("query is required when mode='query'.")
        if normalized_mode != "query" and normalized_retrieval_mode != "graph":
            raise ValidationFailure("retrieval_mode is only supported when mode='query'.")

        replay_hits: list[ReplayHit] = []
        if normalized_mode == "prime":
            selected = self.prime_context(project=project, agent_id=agent_id, session_id=session_id, max_nodes=max_nodes)
            selected_nodes = selected.nodes
            selected_edges = selected.edges if include_edges else []
            summary = selected.summary
        elif normalized_mode == "query":
            selected = self.query(
                query=query,
                max_nodes=max_nodes,
                max_depth=max_depth,
                agent_id=agent_id,
                project=project,
                session_id=session_id,
                retrieval_mode=normalized_retrieval_mode,
            )
            selected_nodes = selected.nodes
            selected_edges = selected.edges if include_edges else []
            replay_hits = selected.replay_hits
            summary = build_query_summary(
                query=query,
                nodes=selected_nodes,
                edges=selected_edges,
                replay_hits=replay_hits,
                retrieval_mode=normalized_retrieval_mode,
            )
        else:
            with self._lock, self._connect() as connection:
                node_rows = connection.execute(
                    """
                    SELECT id, agent_id, project, session_id, label, content, node_type, tags, source_prompt,
                           evidence_records, valid_from, valid_to, created_at, updated_at, access_count, tenant_id
                    FROM nodes
                    WHERE tenant_id = ?
                    ORDER BY updated_at DESC, created_at DESC
                    """,
                    (self.tenant_id,),
                ).fetchall()
                edge_rows = connection.execute(
                    """
                    SELECT id, source_id, target_id, relationship, weight, metadata, created_at
                    FROM edges
                    WHERE tenant_id = ?
                    ORDER BY created_at ASC
                    """,
                    (self.tenant_id,),
                ).fetchall()
            selected_nodes = [
                node
                for row in node_rows
                for node in [self._row_to_node(row)]
                if _scope_matches(node, agent_id=agent_id, project=project, session_id=session_id)
            ]
            selected_edges = [self._row_to_edge(row) for row in edge_rows] if include_edges else []
            if include_edges:
                selected_ids = {node.id for node in selected_nodes}
                selected_edges = [
                    edge for edge in selected_edges if edge.source_id in selected_ids and edge.target_id in selected_ids
                ]
            summary = (
                f"Full graph export for tenant '{self.tenant_id}' with {len(selected_nodes)} nodes and "
                f"{len(selected_edges)} edges."
            )

        bundle = build_context_bundle(
            tenant_id=self.tenant_id,
            project=project,
            mode=normalized_mode,
            retrieval_mode=normalized_retrieval_mode if normalized_mode == "query" else "graph",
            audience=normalized_audience,
            query=query,
            summary=summary,
            nodes=selected_nodes,
            edges=selected_edges,
            replay_hits=replay_hits,
            stats=self.get_stats(),
        )
        return export_context_bundle_files(
            bundle,
            output_path=output_path,
            export_dir=self.export_dir,
            format=normalized_format,
            include_edges=include_edges,
            include_timestamps=include_timestamps,
            include_source_prompt=include_source_prompt,
        )

    def export_markdown_vault(
        self,
        *,
        root_path: str | Path,
        project: str = "",
        agent_id: str = "",
        session_id: str = "",
    ) -> MarkdownVaultExportResult:
        root = Path(root_path).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as connection:
            node_rows = connection.execute(
                """
                SELECT id, agent_id, project, session_id, label, content, node_type, tags, source_prompt,
                       evidence_records, valid_from, valid_to, created_at, updated_at, access_count, tenant_id
                FROM nodes
                WHERE tenant_id = ?
                ORDER BY updated_at DESC, created_at DESC
                """,
                (self.tenant_id,),
            ).fetchall()
            edge_rows = connection.execute(
                """
                SELECT id, source_id, target_id, relationship, weight, metadata, created_at, tenant_id
                FROM edges
                WHERE tenant_id = ?
                ORDER BY created_at ASC
                """,
                (self.tenant_id,),
            ).fetchall()
        selected_nodes = [
            node
            for row in node_rows
            for node in [self._row_to_node(row)]
            if _scope_matches(node, agent_id=agent_id, project=project, session_id=session_id)
        ]
        selected_ids = {node.id for node in selected_nodes}
        selected_edges = [
            self._row_to_edge(row)
            for row in edge_rows
            if row["source_id"] in selected_ids and row["target_id"] in selected_ids
        ]
        node_by_id = {node.id: node for node in selected_nodes}
        files_written: list[str] = []
        for node in selected_nodes:
            project_dir = slugify(node.project or project or "default")
            node_type_dir = slugify(node.node_type.value)
            destination = root / project_dir / node_type_dir / vault_filename(node)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                render_node_document(node, selected_edges, node_by_id),
                encoding="utf-8",
            )
            files_written.append(str(destination.relative_to(root)))
        return MarkdownVaultExportResult(
            root_path=str(root),
            tenant_id=self.tenant_id,
            project=project,
            node_count=len(selected_nodes),
            edge_count=len(selected_edges),
            files_written=files_written,
        )

    def import_markdown_vault(
        self,
        *,
        root_path: str | Path,
    ) -> MarkdownVaultImportResult:
        root = Path(root_path).expanduser()
        documents = iter_vault_documents(root)
        result = MarkdownVaultImportResult(root_path=str(root), tenant_id=self.tenant_id)
        if not documents:
            return result

        with self._lock, self._connect() as connection:
            nodes_by_id = {
                node.id: node
                for node in self._fetch_nodes_by_ids(
                    connection,
                    [str(document.frontmatter["node_id"]) for document in documents],
                )
            }
            label_index: dict[str, Node] = {}
            all_rows = connection.execute(
                """
                SELECT id, agent_id, project, session_id, label, content, node_type, tags, source_prompt,
                       evidence_records, valid_from, valid_to, created_at, updated_at, access_count, tenant_id
                FROM nodes
                WHERE tenant_id = ?
                """,
                (self.tenant_id,),
            ).fetchall()
            for row in all_rows:
                node = self._row_to_node(row)
                label_index.setdefault(node.label.strip().lower(), node)
                nodes_by_id.setdefault(node.id, node)

            imported_id_map: dict[str, str] = {}
            for document in documents:
                original_node_id = str(document.frontmatter.get("node_id", "")).strip()
                node, created = self._upsert_vault_document(connection, document)
                nodes_by_id[node.id] = node
                if original_node_id:
                    imported_id_map[original_node_id] = node.id
                    nodes_by_id[original_node_id] = node
                label_index[node.label.strip().lower()] = node
                if created:
                    result.nodes_created += 1
                else:
                    result.nodes_updated += 1

            for document in documents:
                source_node_id = str(document.frontmatter.get("node_id", "")).strip()
                source_node = nodes_by_id.get(imported_id_map.get(source_node_id, source_node_id))
                if source_node is None:
                    result.conflicts.append(f"Missing source node for document {document.path}.")
                    continue
                for relation in document.relations:
                    target_lookup_id = imported_id_map.get(relation.target_node_id, relation.target_node_id)
                    target_node = nodes_by_id.get(target_lookup_id) if target_lookup_id else None
                    if target_node is None and relation.target_label:
                        target_node = label_index.get(relation.target_label.strip().lower())
                    if target_node is None and relation.target_label:
                        target_node = self._insert_vault_stub_node(
                            connection,
                            label=relation.target_label,
                            project=source_node.project,
                            agent_id=source_node.agent_id,
                            session_id=source_node.session_id,
                        )
                        nodes_by_id[target_node.id] = target_node
                        label_index[target_node.label.strip().lower()] = target_node
                        result.stub_nodes_created += 1
                    if target_node is None:
                        result.conflicts.append(
                            f"Could not resolve relation target '{relation.target_label}' in {document.path.name}."
                        )
                        continue
                    if relation.deleted:
                        if self._delete_edge_record(
                            connection,
                            source_id=source_node.id,
                            target_id=target_node.id,
                            relationship=relation.relationship,
                        ):
                            result.edges_deleted += 1
                        continue
                    if self._find_existing_edge(
                        connection,
                        source_id=source_node.id,
                        target_id=target_node.id,
                        relationship=relation.relationship,
                    ) is None:
                        self._insert_edge_record(
                            connection,
                            source_id=source_node.id,
                            target_id=target_node.id,
                            relationship=relation.relationship,
                        )
                        result.edges_created += 1
        return result

    def import_graph_backup(self, *, input_path: str | Path) -> ImportResult:
        source = Path(input_path).expanduser()
        snapshot = json.loads(source.read_text(encoding="utf-8"))

        with self._lock, self._connect() as connection:
            snapshot_tenant = str(snapshot.get("tenant_id") or self.tenant_id)
            result = ImportResult(
                input_path=str(source),
                tenant_id=self.tenant_id,
                schema_version=int(snapshot.get("schema_version", 1)),
            )
            for raw_node in snapshot.get("nodes", []):
                raw_node = {**raw_node, "tenant_id": raw_node.get("tenant_id") or snapshot_tenant}
                if raw_node["tenant_id"] != self.tenant_id:
                    raw_node["tenant_id"] = self.tenant_id
                if self._fetch_node_row(connection, raw_node["id"]) is None:
                    self._insert_snapshot_node(connection, raw_node)
                    result.nodes_created += 1
                else:
                    self._update_snapshot_node(connection, raw_node)
                    result.nodes_updated += 1

            for raw_edge in snapshot.get("edges", []):
                raw_edge = {**raw_edge, "tenant_id": raw_edge.get("tenant_id") or snapshot_tenant}
                if raw_edge["tenant_id"] != self.tenant_id:
                    raw_edge["tenant_id"] = self.tenant_id
                if self._fetch_edge_row(connection, raw_edge["id"]) is None:
                    self._insert_snapshot_edge(connection, raw_edge)
                    result.edges_created += 1
                else:
                    self._update_snapshot_edge(connection, raw_edge)
                    result.edges_updated += 1
        return result

    def decompose_and_store(self, *, content: str, context: str = "") -> SubgraphResult:
        trimmed_content = content.strip()
        if not trimmed_content:
            raise ValueError("Content cannot be empty.")

        created_nodes: list[Node] = []
        created_ids: set[str] = set()
        context_node: Node | None = None
        if context.strip():
            context_result = self.add_node(
                label=infer_label(context),
                content=context.strip(),
                node_type=NodeType.CONCEPT,
                tags=["decomposition-context"],
                source_prompt=trimmed_content,
            )
            context_node = context_result.node
            created_nodes.append(context_node)
            created_ids.add(context_node.id)

        atomic_items = split_atomic_items(trimmed_content)
        item_nodes: list[Node] = []
        for item in atomic_items:
            store_result = self.add_node(
                label=infer_label(item),
                content=item,
                node_type=infer_node_type(item),
                tags=["decomposed"],
                source_prompt=context.strip() or trimmed_content,
            )
            node = store_result.node
            item_nodes.append(node)
            if node.id not in created_ids:
                created_nodes.append(node)
                created_ids.add(node.id)
            if context_node is not None:
                self.add_edge(
                    source_id=node.id,
                    target_id=context_node.id,
                    relationship=RelationType.PART_OF,
                    metadata={"origin": "decomposition"},
                )

        for index, node in enumerate(item_nodes):
            if index == 0:
                continue
            previous = item_nodes[index - 1]
            shared_tokens = tokenize_text(previous.content) & tokenize_text(node.content)
            if shared_tokens or previous.node_type == node.node_type:
                self.add_edge(
                    source_id=previous.id,
                    target_id=node.id,
                    relationship=infer_relationship(previous, node, shared_tokens=shared_tokens),
                    metadata={"origin": "decomposition"},
                )

        node_ids = [node.id for node in created_nodes]
        with self._lock, self._connect() as connection:
            edges = self._fetch_edges_for_nodes(connection, node_ids)
        return SubgraphResult(
            nodes=created_nodes,
            edges=edges,
            query=f"decomposition:{context.strip() or infer_label(trimmed_content)}",
            total_nodes_in_graph=self.get_stats().total_nodes,
        )

    def observe_conversation(
        self,
        *,
        user_message: str,
        assistant_response: str,
        agent_id: str = "",
        project: str = "",
        session_id: str = "",
    ) -> ObservationResult:
        transcript = f"user: {user_message.strip()}\nassistant: {assistant_response.strip()}".strip()
        observed_at = utc_now()
        candidates = extract_conversation_candidates(
            user_message=user_message,
            assistant_response=assistant_response,
        )

        result = ObservationResult()
        stored_candidate_records: list[tuple[Node, list[str]]] = []
        with self._lock, self._connect() as connection:
            next_turn_index = self._next_transcript_turn_index(connection, session_id=session_id)
            turns = [
                ("user", user_message.strip(), next_turn_index),
                ("assistant", assistant_response.strip(), next_turn_index + 1),
            ]
            for role, text, turn_index in turns:
                if not text:
                    continue
                self._store_transcript_record(
                    connection,
                    agent_id=agent_id,
                    project=project,
                    session_id=session_id,
                    observed_at=observed_at,
                    turn_index=turn_index,
                    role=role,
                    transcript_text=text,
                )
        for candidate in candidates:
            candidate_tags = list(candidate.get("tags", []))
            speaker_tag = next((tag for tag in candidate_tags if str(tag).startswith("speaker:")), "")
            speaker = speaker_tag.split(":", 1)[1] if ":" in speaker_tag else "user"
            turn_index = next_turn_index if speaker == "user" else next_turn_index + 1
            evidence = build_observation_evidence(
                transcript=transcript,
                source_text=str(candidate["content"]),
                speaker=speaker,
                turn_index=turn_index,
                observed_at=observed_at,
                session_id=session_id,
            )
            store_result = self.add_node(
                label=str(candidate["label"]),
                content=str(candidate["content"]),
                node_type=candidate["node_type"],
                tags=candidate_tags,
                source_prompt=transcript,
                agent_id=agent_id,
                project=project,
                session_id=session_id,
                evidence_records=[evidence],
                valid_from=observed_at,
            )
            result.stored_nodes.append(store_result.node)
            stored_candidate_records.append((store_result.node, candidate_tags))
            if store_result.created:
                result.created_count += 1
            else:
                result.reused_count += 1
            for conflict in store_result.conflicts:
                if conflict.other_node_id not in {item.other_node_id for item in result.conflicts}:
                    result.conflicts.append(conflict)

        decision_nodes = [
            (node, tags)
            for node, tags in stored_candidate_records
            if node.node_type == NodeType.DECISION
        ]
        rationale_nodes = [
            (node, tags)
            for node, tags in stored_candidate_records
            if "decision-rationale" in tags and node.node_type == NodeType.FACT
        ]
        for decision_node, decision_tags in decision_nodes:
            decision_categories = {tag for tag in decision_tags if tag in {"database", "backend-framework", "frontend-framework", "auth-mechanism", "api-style"}}
            for rationale_node, rationale_tags in rationale_nodes:
                rationale_categories = {tag for tag in rationale_tags if tag in {"database", "backend-framework", "frontend-framework", "auth-mechanism", "api-style"}}
                if rationale_categories and decision_categories and not (rationale_categories & decision_categories):
                    continue
                self.add_edge(
                    source_id=decision_node.id,
                    target_id=rationale_node.id,
                    relationship=RelationType.DEPENDS_ON,
                    metadata={"origin": "observe_conversation"},
                )
        return result

    def graph_diff(self, *, since: str = "24h") -> GraphDiffResult:
        cutoff = parse_since_value(since)
        with self._lock, self._connect() as connection:
            added_nodes = [
                self._row_to_node(row)
                for row in connection.execute(
                """
                SELECT id, agent_id, project, session_id, label, content, node_type, tags, source_prompt, evidence_records, valid_from, valid_to,
                       created_at, updated_at, access_count, tenant_id
                FROM nodes
                WHERE tenant_id = ? AND created_at >= ?
                    ORDER BY created_at DESC
                    """,
                    (self.tenant_id, cutoff.isoformat()),
                ).fetchall()
            ]
            updated_nodes = [
                self._row_to_node(row)
                for row in connection.execute(
                """
                    SELECT id, agent_id, project, session_id, label, content, node_type, tags, source_prompt, evidence_records, valid_from, valid_to,
                           created_at, updated_at, access_count, tenant_id
                    FROM nodes
                    WHERE tenant_id = ?
                      AND updated_at >= ?
                      AND created_at < ?
                    ORDER BY updated_at DESC
                    """,
                    (self.tenant_id, cutoff.isoformat(), cutoff.isoformat()),
                ).fetchall()
            ]
            created_edges = [
                self._row_to_edge(row)
                for row in connection.execute(
                    """
                    SELECT id, source_id, target_id, relationship, weight, metadata, created_at
                    FROM edges
                    WHERE tenant_id = ? AND created_at >= ?
                    ORDER BY created_at DESC
                    """,
                    (self.tenant_id, cutoff.isoformat()),
                ).fetchall()
            ]
            contradiction_edges = [
                edge for edge in created_edges if edge.relationship == RelationType.CONTRADICTS.value
            ]
        return GraphDiffResult(
            since=since,
            added_nodes=added_nodes,
            updated_nodes=updated_nodes,
            created_edges=created_edges,
            contradiction_edges=contradiction_edges,
        )

    def prime_context(
        self,
        *,
        project: str = "",
        agent_id: str = "",
        session_id: str = "",
        max_nodes: int = 25,
    ) -> PrimeContextResult:
        with self._lock, self._connect() as connection:
            total_nodes = int(
                connection.execute("SELECT COUNT(*) FROM nodes WHERE tenant_id = ?", (self.tenant_id,)).fetchone()[0]
            )
            if total_nodes == 0:
                return PrimeContextResult(project=project, summary="No stored memory is available yet.")

            active_session_id = _retrieval_session_scope(
                agent_id=agent_id,
                project=project,
                session_id=session_id,
            )
            # Collect seed anchors from multiple sources
            seed_ids: list[str] = []
            seed_ids.extend(
                self._most_connected_node_ids(
                    connection,
                    limit=5,
                    agent_id=agent_id,
                    project=project,
                    session_id=active_session_id,
                )
            )
            seed_ids.extend(
                node.id
                for node in self.list_recent_nodes(
                    limit=5,
                    agent_id=agent_id,
                    project=project,
                    session_id=active_session_id,
                )
            )
            if project.strip():
                seed_ids.extend(
                    self._find_project_node_ids(
                        connection,
                        project=project,
                        agent_id=agent_id,
                        session_id=active_session_id,
                        limit=8,
                    )
                )
            seed_ids = list(dict.fromkeys(seed_ids))  # Deduplicate

            if not seed_ids:
                return PrimeContextResult(project=project, summary="No seed nodes found for priming.")

            # Load all embeddable nodes and build graph
            node_rows = connection.execute(
                """
                SELECT id, agent_id, project, session_id, label, content, node_type, tags, source_prompt, evidence_records, valid_from, valid_to,
                       created_at, updated_at, access_count, embedding, tenant_id
                FROM nodes
                WHERE tenant_id = ? AND embedding IS NOT NULL
                """,
                (self.tenant_id,),
            ).fetchall()

            if not node_rows:
                return PrimeContextResult(project=project, summary="No embeddable nodes available for expansion.")

            nodes_by_id: dict[str, Node] = {}
            for row in node_rows:
                node = self._row_to_node(row)
                if not _scope_matches(node, agent_id=agent_id, project=project, session_id=active_session_id):
                    continue
                nodes_by_id[node.id] = node

            graph = self._load_graph(connection, node_ids=nodes_by_id.keys())

            if not nodes_by_id:
                return PrimeContextResult(project=project, summary="No scoped nodes found for priming.")

            # Expand from seeds using relation-aware traversal
            max_depth = 2
            expanded_depths, expansion_metadata = self._expand_node_depths_with_context(
                graph, seed_ids, max_depth
            )

            # Build candidate nodes from expansion
            candidate_nodes = [
                nodes_by_id[nid]
                for nid in expanded_depths
                if nid in nodes_by_id
            ]
            if not candidate_nodes:
                return PrimeContextResult(project=project, summary="Expansion produced no candidate nodes.")

            # Score with relation-aware ranking (no natural language query)
            similarity_by_id = {nid: 0.0 for nid in expanded_depths}
            lexical_by_id = {nid: 0.0 for nid in expanded_depths}
            # Boost seed IDs synthetically
            for seed_id in seed_ids:
                if seed_id in similarity_by_id:
                    similarity_by_id[seed_id] = 0.5

            degree_by_id = dict(graph.degree(expanded_depths.keys()))
            max_access = max((node.access_count for node in candidate_nodes), default=0)
            max_degree = max(degree_by_id.values(), default=0)

            temporal_hints = _NeutralTemporalHints()
            scored_nodes = self._sort_scored_nodes(
                candidate_nodes,
                temporal_hints=temporal_hints,
                similarity_by_id=similarity_by_id,
                lexical_by_id=lexical_by_id,
                degree_by_id=degree_by_id,
                max_access=max_access,
                max_degree=max_degree,
                max_depth=max_depth,
                expanded_depths=expanded_depths,
                expansion_metadata=expansion_metadata,
            )

            # Apply support coverage
            selected_nodes = scored_nodes[:max_nodes]
            candidate_pool = {node.id: node for node in candidate_nodes}
            selected_nodes = self._ensure_support_coverage(
                selected_nodes, candidate_pool, graph, max_nodes
            )

            selected_ids = [node.id for node in selected_nodes]
            edges = self._fetch_edges_for_nodes(connection, selected_ids)

        # Build structured summary
        summary = self._build_prime_summary(
            selected_nodes=selected_nodes,
            edges=edges,
            total_nodes_in_graph=total_nodes,
            project=project,
        )

        return PrimeContextResult(
            project=project,
            summary=summary,
            nodes=selected_nodes,
            edges=edges,
            total_nodes_in_graph=total_nodes,
        )

    def get_topics(self) -> TopicResult:
        with self._lock, self._connect() as connection:
            node_rows = connection.execute(
                """
                SELECT id, agent_id, project, session_id, label, content, node_type, tags, source_prompt,
                       evidence_records, valid_from, valid_to, created_at, updated_at, access_count, tenant_id
                FROM nodes
                WHERE tenant_id = ?
                """
            , (self.tenant_id,)).fetchall()
            if not node_rows:
                return TopicResult(clusters=[], total_clusters=0)
            nodes = [self._row_to_node(row) for row in node_rows]
            graph = self._load_graph(connection, node_ids=[node.id for node in nodes]).to_undirected()
            partition = self._build_topic_partition(graph, nodes)

        nodes_by_id = {node.id: node for node in nodes}
        clusters_by_id: dict[int, list[Node]] = {}
        for node_id, cluster_id in partition.items():
            clusters_by_id.setdefault(int(cluster_id), []).append(nodes_by_id[node_id])

        clusters: list[TopicCluster] = []
        for cluster_id, cluster_nodes in sorted(
            clusters_by_id.items(),
            key=lambda item: (-len(item[1]), item[0]),
        ):
            label, top_tags = summarize_topic(cluster_nodes)
            ordered_nodes = sorted(
                cluster_nodes,
                key=lambda node: (-node.access_count, -node.updated_at.timestamp(), node.label.lower()),
            )
            clusters.append(
                TopicCluster(
                    cluster_id=cluster_id,
                    label=label,
                    node_count=len(cluster_nodes),
                    top_tags=top_tags,
                    nodes=ordered_nodes,
                )
            )
        return TopicResult(clusters=clusters, total_clusters=len(clusters))

    def _build_prime_summary(
        self,
        *,
        selected_nodes: list[Node],
        edges: list[Edge],
        total_nodes_in_graph: int,
        project: str = "",
    ) -> str:
        """Build a structured summary of prime context with type and relationship counts."""
        # Count node types
        type_counts: dict[str, int] = {}
        for node in selected_nodes:
            type_counts[node.node_type.value] = type_counts.get(node.node_type.value, 0) + 1

        # Count edge relationships
        relationship_counts: dict[str, int] = {}
        for edge in edges:
            rel = edge.relationship
            relationship_counts[rel] = relationship_counts.get(rel, 0) + 1

        # Build type breakdown
        type_breakdown = ", ".join(
            f"{count} {ttype}" for ttype, count in sorted(type_counts.items())
        ) if type_counts else "no nodes"

        # Build relationship breakdown
        relationship_breakdown = ", ".join(
            f"{count} {rel}" for rel, count in sorted(relationship_counts.items())
        ) if relationship_counts else "no edges"

        # Check for contradictions
        has_contradictions = "contradicts" in relationship_counts
        contradiction_warning = " [⚠ Contradictions present]" if has_contradictions else ""

        # Check for questions
        has_questions = "question" in type_counts
        question_warning = " [?]" if has_questions else ""

        # Build base summary
        if project.strip():
            base = f"Prime context for project '{project}': {len(selected_nodes)} nodes ({type_breakdown}) with {len(edges)} edges ({relationship_breakdown})"
        else:
            base = f"Prime context: {len(selected_nodes)} nodes ({type_breakdown}) with {len(edges)} edges ({relationship_breakdown})"

        base += f" from {total_nodes_in_graph} total nodes"
        base += contradiction_warning + question_warning

        return base

    def _require_node(self, connection: sqlite3.Connection, node_id: str) -> None:
        if self._fetch_node_row(connection, node_id) is None:
            raise ValueError(f"Node not found: {node_id}")

    def _find_duplicate_node(
        self,
        connection: sqlite3.Connection,
        *,
        node: Node,
        embedding: np.ndarray,
    ) -> tuple[Node, str, float | None] | None:
        rows = connection.execute(
            """
            SELECT id, agent_id, project, session_id, label, content, node_type, tags, source_prompt, evidence_records,
                   valid_from, valid_to, created_at, updated_at, access_count, embedding, tenant_id
            FROM nodes
            WHERE tenant_id = ? AND embedding IS NOT NULL
            """,
            (self.tenant_id,),
        ).fetchall()

        normalized_label = normalize_text(node.label)
        normalized_content = normalize_text(node.content)
        # Type-aware cosine threshold — decisions merge at 0.82, facts at 0.92, etc.
        type_threshold = type_aware_dedup_threshold(node.node_type,
                                                    default=self.dedup_similarity_threshold)
        best_match: tuple[Node, float] | None = None

        for row in rows:
            existing_node = self._row_to_node(row)
            if not _scope_matches(
                existing_node,
                agent_id=node.agent_id,
                project=node.project,
                session_id=node.session_id,
            ):
                continue
            if not compatible_node_types(node.node_type, existing_node.node_type):
                continue
            existing_label = normalize_text(existing_node.label)
            existing_content = normalize_text(existing_node.content)

            # ── Layer 0: entity-key hard block ────────────────────────
            # If both nodes name a specific technology AND those technologies
            # are different (but in the same category), block the merge.
            # e.g. "use PostgreSQL" vs "use MySQL" — similar sentence, different choice.
            node_entity = extract_choice_entity(node.content)
            existing_entity = extract_choice_entity(existing_node.content)
            if (
                node_entity is not None
                and existing_entity is not None
                and node_entity[1] == existing_entity[1]   # same category
                and node_entity[0] != existing_entity[0]   # different entity
            ):
                continue  # never merge "postgres" node with "mysql" node

            # ── Layer 0b: numeric-conflict guard ───────────────────────
            # Same entity BUT different critical number (e.g. JWT 15min vs 1hr).
            # Conflicting numbers signal distinct facts, not duplicates.
            # Also applies to non-entity facts that have conflicting numbers.
            if contains_conflicting_numbers(node.content, existing_node.content) and (
                node_entity is None
                or existing_entity is None
                or node_entity[0] == existing_entity[0]
            ):
                continue

            if normalized_content == existing_content:
                return existing_node, "exact_content", 1.0

            # ── Layer 2: substring containment (cheap, catches rephrased subsets)
            if len(normalized_content) >= 10 and len(existing_content) >= 10:
                if normalized_content in existing_content or existing_content in normalized_content:
                    return existing_node, "content_substring", 0.98

            # ── Layer 3: semantic similarity (expensive — compute embedding once) ─
            existing_embedding = self.embedding_model.from_bytes(row["embedding"])
            similarity = self.embedding_model.cosine_similarity(embedding, existing_embedding)
            label_score = label_similarity(node.label, existing_node.label)
            acronym_match = is_acronym_match(node.label, existing_node.label)

            if normalized_label == existing_label and similarity >= self.dedup_same_label_threshold:
                return existing_node, "same_label_high_similarity", similarity
            if acronym_match and similarity >= max(self.dedup_same_label_threshold - 0.25, 0.55):
                return existing_node, "acronym_entity_match", similarity
            if label_score >= 0.92 and similarity >= max(self.dedup_same_label_threshold - 0.2, 0.6):
                return existing_node, "label_entity_match", similarity

            # ── Layer 3b: same-entity aggressive merge ──────────────────
            # If both nodes reference the SAME named entity, lower the cosine
            # threshold significantly — "fastapi was chosen" and "we chose fastapi
            # because async" should merge even at cosine ~0.65.
            # The numeric-conflict guard (Layer 0b) already blocked cases where
            # the same entity appears with different critical numbers.
            if (
                node_entity is not None
                and existing_entity is not None
                and node_entity[0] == existing_entity[0]  # identical entity token
                and similarity >= 0.60
            ):
                return existing_node, "same_entity_merge", similarity

            # ── Layer 3c: Jaccard-boosted merge (type-aware lower threshold) ──
            # If content words overlap significantly AND cosine is high for the
            # node type, treat as duplicate — catches paraphrase true-dups.
            jaccard = content_token_jaccard(node.content, existing_node.content)
            boosted_threshold = max(type_threshold - 0.05, 0.70)
            if jaccard >= 0.35 and similarity >= boosted_threshold:
                return existing_node, "jaccard_boosted_similarity", similarity

            # ── Layer 3c: pure cosine fallback (conservative global threshold) ─
            if similarity >= self.dedup_similarity_threshold:
                if best_match is None or similarity > best_match[1]:
                    best_match = (existing_node, similarity)

        if best_match is None:
            return None

        return best_match[0], "high_similarity", best_match[1]

    def _merge_duplicate_node(
        self,
        connection: sqlite3.Connection,
        *,
        existing_node: Node,
        incoming_node: Node,
    ) -> Node:
        merged_tags = list(dict.fromkeys([*existing_node.tags, *incoming_node.tags]))
        updated_source_prompt = existing_node.source_prompt or incoming_node.source_prompt
        merged_evidence = merge_evidence_records(existing_node.evidence_records, incoming_node.evidence_records)
        merged_valid_from, merged_valid_to = merge_validity_windows(
            existing_node.valid_from,
            incoming_node.valid_from,
            existing_node.valid_to,
            incoming_node.valid_to,
        )
        updated_at = utc_now()
        connection.execute(
            """
            UPDATE nodes
            SET agent_id = ?, project = ?, session_id = ?, tags = ?, source_prompt = ?, evidence_records = ?, valid_from = ?, valid_to = ?, updated_at = ?
            WHERE id = ? AND tenant_id = ?
            """,
            (
                _merge_scope_value(existing_node.agent_id, incoming_node.agent_id),
                _merge_scope_value(existing_node.project, incoming_node.project),
                _merge_scope_value(existing_node.session_id, incoming_node.session_id),
                json.dumps(merged_tags),
                updated_source_prompt,
                _encode_evidence_records(merged_evidence),
                merged_valid_from.isoformat() if merged_valid_from is not None else None,
                merged_valid_to.isoformat() if merged_valid_to is not None else None,
                updated_at.isoformat(),
                existing_node.id,
                self.tenant_id,
            ),
        )
        return Node(
            id=existing_node.id,
            tenant_id=existing_node.tenant_id,
            agent_id=_merge_scope_value(existing_node.agent_id, incoming_node.agent_id),
            project=_merge_scope_value(existing_node.project, incoming_node.project),
            session_id=_merge_scope_value(existing_node.session_id, incoming_node.session_id),
            label=existing_node.label,
            content=existing_node.content,
            node_type=existing_node.node_type,
            tags=merged_tags,
            source_prompt=updated_source_prompt,
            evidence_records=merged_evidence,
            valid_from=merged_valid_from,
            valid_to=merged_valid_to,
            created_at=existing_node.created_at,
            updated_at=updated_at,
            access_count=existing_node.access_count,
        )

    def _register_conflicts(
        self,
        connection: sqlite3.Connection,
        node: Node,
    ) -> list[ConflictRecord]:
        if node.node_type not in {NodeType.PREFERENCE, NodeType.DECISION}:
            return []

        rows = connection.execute(
            """
            SELECT id, agent_id, project, session_id, label, content, node_type, tags, source_prompt,
                   evidence_records, valid_from, valid_to, created_at, updated_at, access_count, embedding, tenant_id
            FROM nodes
            WHERE tenant_id = ? AND id != ?
            """,
            (self.tenant_id, node.id),
        ).fetchall()
        conflicts: list[ConflictRecord] = []
        for row in rows:
            existing_node = self._row_to_node(row)
            if not _scope_matches(
                existing_node,
                agent_id=node.agent_id,
                project=node.project,
                session_id=node.session_id,
            ):
                continue
            reason = detect_conflict_reason(existing_node, node)
            if reason is None:
                continue
            existing_edge = self._find_existing_edge(
                connection,
                source_id=node.id,
                target_id=existing_node.id,
                relationship=RelationType.CONTRADICTS,
            )
            if existing_edge is None:
                edge = Edge(
                    tenant_id=self.tenant_id,
                    source_id=node.id,
                    target_id=existing_node.id,
                    relationship=RelationType.CONTRADICTS,
                    metadata={"origin": "auto-conflict", "reason": reason},
                )
                connection.execute(
                    """
                    INSERT INTO edges (
                        id, tenant_id, source_id, target_id, relationship, weight, metadata, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        edge.id,
                        edge.tenant_id,
                        edge.source_id,
                        edge.target_id,
                        edge.relationship,
                        edge.weight,
                        json.dumps(edge.metadata),
                        edge.created_at.isoformat(),
                    ),
                )
            conflicts.append(
                ConflictRecord(
                    other_node_id=existing_node.id,
                    other_node_label=existing_node.label,
                    reason=reason,
                )
            )
        return conflicts

    def _fetch_node_row(self, connection: sqlite3.Connection, node_id: str) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT id, agent_id, project, session_id, label, content, node_type, tags, source_prompt, evidence_records, valid_from, valid_to,
                   created_at, updated_at, access_count, embedding, tenant_id
            FROM nodes
            WHERE id = ? AND tenant_id = ?
            """,
            (node_id, self.tenant_id),
        ).fetchone()

    def _fetch_nodes_by_ids(
        self,
        connection: sqlite3.Connection,
        node_ids: list[str],
    ) -> list[Node]:
        if not node_ids:
            return []
        placeholders = ", ".join("?" for _ in node_ids)
        rows = connection.execute(
            f"""
            SELECT id, agent_id, project, session_id, label, content, node_type, tags, source_prompt, evidence_records, valid_from, valid_to,
                   created_at, updated_at, access_count, tenant_id
            FROM nodes
            WHERE tenant_id = ? AND id IN ({placeholders})
            """,
            (self.tenant_id, *node_ids),
        ).fetchall()
        rows_by_id = {row["id"]: row for row in rows}
        return [self._row_to_node(rows_by_id[node_id]) for node_id in node_ids if node_id in rows_by_id]

    def _build_timeline_items(
        self,
        *,
        nodes: list[Node],
        edges: list[Edge],
        include_evidence: bool,
        limit: int,
    ) -> list[ContextTimelineItem]:
        items: list[ContextTimelineItem] = []
        for node in nodes:
            items.append(
                ContextTimelineItem(
                    kind="node_created",
                    timestamp=node.created_at,
                    label=node.label,
                    summary=node.content,
                    node_id=node.id,
                )
            )
            if node.updated_at != node.created_at:
                items.append(
                    ContextTimelineItem(
                        kind="node_updated",
                        timestamp=node.updated_at,
                        label=node.label,
                        summary=node.content,
                        node_id=node.id,
                    )
                )
            if include_evidence:
                for record in node.evidence_records:
                    items.append(
                        ContextTimelineItem(
                            kind="evidence",
                            timestamp=record.observed_at,
                            label=node.label,
                            summary=f"{record.source_role or 'unknown'} turn {record.turn_index}: {record.source_text or node.content}",
                            node_id=node.id,
                        )
                    )
        node_by_id = {node.id: node for node in nodes}
        for edge in edges:
            source_label = node_by_id.get(edge.source_id).label if edge.source_id in node_by_id else edge.source_id[:8]
            target_label = node_by_id.get(edge.target_id).label if edge.target_id in node_by_id else edge.target_id[:8]
            items.append(
                ContextTimelineItem(
                    kind=f"edge_{edge.relationship}",
                    timestamp=edge.created_at,
                    label=f"{source_label} -> {target_label}",
                    summary=edge.relationship,
                    edge_id=edge.id,
                )
            )
        return sorted(
            items,
            key=lambda item: (item.timestamp, item.kind, item.label),
            reverse=True,
        )[:limit]

    def _build_conflict_entries(
        self,
        connection: sqlite3.Connection,
        *,
        edges: list[Edge],
        include_resolved: bool,
        limit: int,
    ) -> list[ConflictEntry]:
        node_ids = list(dict.fromkeys([edge.source_id for edge in edges] + [edge.target_id for edge in edges]))
        nodes_by_id = {node.id: node for node in self._fetch_nodes_by_ids(connection, node_ids)}
        entries: list[ConflictEntry] = []
        for edge in edges:
            resolved, resolution_note, resolved_at = self._conflict_resolution_state(edge)
            if resolved and not include_resolved:
                continue
            source_node = nodes_by_id.get(edge.source_id)
            target_node = nodes_by_id.get(edge.target_id)
            if source_node is None or target_node is None:
                continue
            entries.append(
                ConflictEntry(
                    edge=edge,
                    source_node=source_node,
                    target_node=target_node,
                    resolved=resolved,
                    resolution_note=resolution_note,
                    resolved_at=resolved_at,
                )
            )
            if len(entries) >= limit:
                break
        return entries

    def _conflict_resolution_state(self, edge: Edge) -> tuple[bool, str, datetime | None]:
        metadata = edge.metadata or {}
        resolved = bool(metadata.get("resolved"))
        resolution_note = str(metadata.get("resolution_note", "") or "")
        resolved_at_raw = metadata.get("resolved_at")
        resolved_at = _parse_datetime(resolved_at_raw) if resolved_at_raw else None
        return resolved, resolution_note, resolved_at

    def _row_to_node(self, row: sqlite3.Row) -> Node:
        row_keys = set(row.keys())
        return Node(
            id=row["id"],
            tenant_id=row["tenant_id"] if "tenant_id" in row.keys() else self.tenant_id,
            agent_id=row["agent_id"] if "agent_id" in row_keys else "",
            project=row["project"] if "project" in row_keys else "",
            session_id=row["session_id"] if "session_id" in row_keys else "",
            label=row["label"],
            content=row["content"],
            node_type=NodeType(row["node_type"]),
            tags=json.loads(row["tags"] or "[]"),
            source_prompt=row["source_prompt"] or "",
            evidence_records=_decode_evidence_records(row["evidence_records"]) if "evidence_records" in row_keys else [],
            valid_from=_parse_datetime(row["valid_from"]) if "valid_from" in row_keys and row["valid_from"] else None,
            valid_to=_parse_datetime(row["valid_to"]) if "valid_to" in row_keys and row["valid_to"] else None,
            created_at=_parse_datetime(row["created_at"]),
            updated_at=_parse_datetime(row["updated_at"]),
            access_count=int(row["access_count"] or 0),
        )

    def _row_to_edge(self, row: sqlite3.Row) -> Edge:
        return Edge(
            id=row["id"],
            tenant_id=row["tenant_id"] if "tenant_id" in row.keys() else self.tenant_id,
            source_id=row["source_id"],
            target_id=row["target_id"],
            relationship=row["relationship"],
            weight=float(row["weight"]),
            metadata=json.loads(row["metadata"] or "{}"),
            created_at=_parse_datetime(row["created_at"]),
        )

    def _row_to_transcript_record(self, row: sqlite3.Row) -> TranscriptRecord:
        return TranscriptRecord(
            id=row["id"],
            tenant_id=row["tenant_id"] if "tenant_id" in row.keys() else self.tenant_id,
            agent_id=row["agent_id"] if "agent_id" in row.keys() else "",
            project=row["project"] if "project" in row.keys() else "",
            session_id=row["session_id"] if "session_id" in row.keys() else "",
            observed_at=_parse_datetime(row["observed_at"]),
            turn_index=int(row["turn_index"] or 0),
            role=row["role"] or "",
            transcript_text=row["transcript_text"],
            metadata=_decode_metadata(row["metadata"]) if "metadata" in row.keys() else {},
        )

    def _transcript_scope_matches(
        self,
        record: TranscriptRecord,
        *,
        agent_id: str = "",
        project: str = "",
        session_id: str = "",
    ) -> bool:
        normalized_agent = agent_id.strip().lower()
        normalized_project = project.strip().lower()
        normalized_session = session_id.strip().lower()
        if normalized_agent and record.agent_id.strip().lower() != normalized_agent:
            return False
        if normalized_project and record.project.strip().lower() != normalized_project:
            return False
        if normalized_session and record.session_id.strip().lower() != normalized_session:
            return False
        return True

    def _next_transcript_turn_index(self, connection: sqlite3.Connection, *, session_id: str) -> int:
        row = connection.execute(
            """
            SELECT COALESCE(MAX(turn_index), -1) AS max_turn_index
            FROM transcript_records
            WHERE tenant_id = ? AND session_id = ?
            """,
            (self.tenant_id, session_id),
        ).fetchone()
        return int(row["max_turn_index"] or -1) + 1

    def _store_transcript_record(
        self,
        connection: sqlite3.Connection,
        *,
        agent_id: str,
        project: str,
        session_id: str,
        observed_at: datetime,
        turn_index: int,
        role: str,
        transcript_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> TranscriptRecord:
        record = TranscriptRecord(
            tenant_id=self.tenant_id,
            agent_id=agent_id,
            project=project,
            session_id=session_id,
            observed_at=observed_at,
            turn_index=turn_index,
            role=role,
            transcript_text=transcript_text,
            metadata=metadata or {},
        )
        embedding = self.embedding_model.embed(record.transcript_text)
        connection.execute(
            """
            INSERT INTO transcript_records (
                id, tenant_id, agent_id, project, session_id, observed_at, turn_index, role, transcript_text, embedding, metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.tenant_id,
                record.agent_id,
                record.project,
                record.session_id,
                record.observed_at.isoformat(),
                record.turn_index,
                record.role,
                record.transcript_text,
                self.embedding_model.to_bytes(embedding),
                _encode_metadata(record.metadata),
            ),
        )
        return record

    def _upsert_vault_document(
        self,
        connection: sqlite3.Connection,
        document: Any,
    ) -> tuple[Node, bool]:
        node_id = str(document.frontmatter.get("node_id", "")).strip()
        row = self._fetch_node_row(connection, node_id)
        raw_type = str(document.frontmatter.get("node_type", "note") or "note")
        try:
            node_type = NodeType(raw_type)
        except ValueError:
            node_type = NodeType.NOTE
        tags = [str(tag) for tag in document.frontmatter.get("tags", []) or []]
        agent_id = str(document.frontmatter.get("agent_id", "") or "")
        project = str(document.frontmatter.get("project", "") or "")
        session_id = str(document.frontmatter.get("session_id", "") or "")
        valid_from = self._parse_optional_datetime(document.frontmatter.get("valid_from"))
        valid_to = self._parse_optional_datetime(document.frontmatter.get("valid_to"))
        evidence_records = evidence_from_lines(document.evidence_lines)
        content = document.content.strip() or document.label
        embedding_bytes = self.embedding_model.to_bytes(self.embedding_model.embed(content))
        if row is None:
            created_at = self._parse_optional_datetime(document.frontmatter.get("created_at")) or utc_now()
            updated_at = utc_now()
            node = Node(
                id=node_id,
                tenant_id=self.tenant_id,
                agent_id=agent_id,
                project=project,
                session_id=session_id,
                label=document.label,
                content=content,
                node_type=node_type,
                tags=tags,
                evidence_records=evidence_records,
                valid_from=valid_from,
                valid_to=valid_to,
                created_at=created_at,
                updated_at=updated_at,
            )
            connection.execute(
                """
                INSERT INTO nodes (
                    id, tenant_id, agent_id, project, session_id, label, content, node_type, tags, embedding,
                    source_prompt, evidence_records, valid_from, valid_to, created_at, updated_at, access_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node.id,
                    node.tenant_id,
                    node.agent_id,
                    node.project,
                    node.session_id,
                    node.label,
                    node.content,
                    node.node_type.value,
                    json.dumps(node.tags),
                    embedding_bytes,
                    "",
                    _encode_evidence_records(node.evidence_records),
                    node.valid_from.isoformat() if node.valid_from is not None else None,
                    node.valid_to.isoformat() if node.valid_to is not None else None,
                    node.created_at.isoformat(),
                    node.updated_at.isoformat(),
                    node.access_count,
                ),
            )
            return node, True

        existing = self._row_to_node(row)
        updated_at = utc_now()
        node = Node(
            id=existing.id,
            tenant_id=existing.tenant_id,
            agent_id=agent_id,
            project=project,
            session_id=session_id,
            label=document.label,
            content=content,
            node_type=node_type,
            tags=tags,
            source_prompt=existing.source_prompt,
            evidence_records=evidence_records or existing.evidence_records,
            valid_from=valid_from,
            valid_to=valid_to,
            created_at=existing.created_at,
            updated_at=updated_at,
            access_count=existing.access_count,
        )
        connection.execute(
            """
            UPDATE nodes
            SET agent_id = ?, project = ?, session_id = ?, label = ?, content = ?, node_type = ?, tags = ?,
                embedding = ?, evidence_records = ?, valid_from = ?, valid_to = ?, updated_at = ?
            WHERE id = ? AND tenant_id = ?
            """,
            (
                node.agent_id,
                node.project,
                node.session_id,
                node.label,
                node.content,
                node.node_type.value,
                json.dumps(node.tags),
                embedding_bytes,
                _encode_evidence_records(node.evidence_records),
                node.valid_from.isoformat() if node.valid_from is not None else None,
                node.valid_to.isoformat() if node.valid_to is not None else None,
                node.updated_at.isoformat(),
                node.id,
                self.tenant_id,
            ),
        )
        return node, False

    def _insert_vault_stub_node(
        self,
        connection: sqlite3.Connection,
        *,
        label: str,
        project: str,
        agent_id: str,
        session_id: str,
    ) -> Node:
        node = Node(
            tenant_id=self.tenant_id,
            agent_id=agent_id,
            project=project,
            session_id=session_id,
            label=label,
            content=f"Stub node imported from vault for {label}.",
            node_type=NodeType.NOTE,
            tags=["stub", "vault-import"],
        )
        connection.execute(
            """
            INSERT INTO nodes (
                id, tenant_id, agent_id, project, session_id, label, content, node_type, tags, embedding,
                source_prompt, evidence_records, valid_from, valid_to, created_at, updated_at, access_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node.id,
                node.tenant_id,
                node.agent_id,
                node.project,
                node.session_id,
                node.label,
                node.content,
                node.node_type.value,
                json.dumps(node.tags),
                self.embedding_model.to_bytes(self.embedding_model.embed(node.content)),
                "",
                _encode_evidence_records([]),
                None,
                None,
                node.created_at.isoformat(),
                node.updated_at.isoformat(),
                node.access_count,
            ),
        )
        return node

    def _insert_edge_record(
        self,
        connection: sqlite3.Connection,
        *,
        source_id: str,
        target_id: str,
        relationship: str,
    ) -> Edge:
        edge = Edge(
            tenant_id=self.tenant_id,
            source_id=source_id,
            target_id=target_id,
            relationship=relationship,
        )
        connection.execute(
            """
            INSERT INTO edges (
                id, tenant_id, source_id, target_id, relationship, weight, metadata, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                edge.id,
                edge.tenant_id,
                edge.source_id,
                edge.target_id,
                edge.relationship,
                edge.weight,
                _encode_metadata(edge.metadata),
                edge.created_at.isoformat(),
            ),
        )
        return edge

    def _delete_edge_record(
        self,
        connection: sqlite3.Connection,
        *,
        source_id: str,
        target_id: str,
        relationship: str,
    ) -> bool:
        cursor = connection.execute(
            """
            DELETE FROM edges
            WHERE tenant_id = ? AND source_id = ? AND target_id = ? AND relationship = ?
            """,
            (self.tenant_id, source_id, target_id, normalize_relationship(relationship)),
        )
        return int(cursor.rowcount or 0) > 0

    def _parse_optional_datetime(self, raw: Any) -> datetime | None:
        if raw in (None, ""):
            return None
        if isinstance(raw, datetime):
            return raw if raw.tzinfo is not None else raw.replace(tzinfo=timezone.utc)
        try:
            return _parse_datetime(str(raw))
        except ValueError:
            return None

    def _load_graph(
        self,
        connection: sqlite3.Connection,
        *,
        node_ids: Iterable[str],
    ) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(node_ids)
        rows = connection.execute(
            """
            SELECT source_id, target_id, relationship, weight, metadata, created_at
            FROM edges
            WHERE tenant_id = ?
            """,
            (self.tenant_id,),
        ).fetchall()

        for row in rows:
            try:
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            except (json.JSONDecodeError, TypeError):
                metadata = {}

            graph.add_edge(
                row["source_id"],
                row["target_id"],
                relationship=row["relationship"] or "relates_to",
                weight=float(row["weight"]) if row["weight"] is not None else 1.0,
                metadata=metadata,
                created_at=row["created_at"],
            )
        return graph

    def _relation_priority(self, relationship: str) -> float:
        return RELATION_WEIGHTS.get(relationship, 0.50)

    def _temporal_sort_value(self, node: Node, hints: Any) -> float:
        if hints.recency_mode == "latest":
            return -node.updated_at.timestamp()
        if hints.recency_mode == "oldest":
            return node.created_at.timestamp()
        return -node.updated_at.timestamp()

    def _seed_temporal_order(self, node: Node, hints: Any) -> float:
        if hints.recency_mode == "latest":
            return -node.updated_at.timestamp()
        if hints.recency_mode == "oldest":
            return node.created_at.timestamp()
        return 0.0

    def _sort_scored_nodes(
        self,
        candidate_nodes: list[Node],
        *,
        temporal_hints: Any,
        similarity_by_id: dict[str, float],
        lexical_by_id: dict[str, float],
        degree_by_id: dict[str, int],
        max_access: int,
        max_degree: int,
        max_depth: int,
        expanded_depths: dict[str, int],
        expansion_metadata: dict[str, ExpansionMeta] | None = None,
    ) -> list[Node]:
        def combined_score(node: Node) -> float:
            base = score_node(
                node=node,
                semantic_similarity=similarity_by_id.get(node.id, 0.0),
                lexical_score=lexical_by_id.get(node.id, 0.0),
                max_access=max_access,
                degree_score=(degree_by_id.get(node.id, 0) / max_degree if max_degree > 0 else 0.0),
                depth=expanded_depths.get(node.id, max_depth + 1),
            ) + temporal_score_adjustment(node, temporal_hints)
            
            if expansion_metadata is not None and node.id in expansion_metadata:
                meta = expansion_metadata[node.id]
                base += RELATION_SCORE_BOOST.get(meta.via_relation, 0.0)
            
            return base

        if temporal_hints.recency_mode == "latest":
            return sorted(
                candidate_nodes,
                key=lambda node: (-node.updated_at.timestamp(), -combined_score(node), node.label.lower()),
            )
        if temporal_hints.recency_mode == "oldest":
            return sorted(
                candidate_nodes,
                key=lambda node: (node.created_at.timestamp(), -combined_score(node), node.label.lower()),
            )
        return sorted(
            candidate_nodes,
            key=lambda node: (-combined_score(node), -node.updated_at.timestamp(), node.label.lower()),
        )

    def _expand_node_depths_with_context(
        self,
        graph: nx.DiGraph,
        seed_ids: list[str],
        max_depth: int,
        *,
        min_priority: float = 0.20,
        decay: float = 0.70,
    ) -> tuple[dict[str, int], dict[str, ExpansionMeta]]:
        ordered: dict[str, int] = {}
        metadata: dict[str, ExpansionMeta] = {}
        seen: set[str] = set()

        # Heap entries: (neg_priority, tiebreaker, node_id, depth, via_relation, from_node, effective_priority)
        _counter = 0
        heap: list[tuple[float, int, str, int, str, str, float]] = []

        for seed_id in seed_ids:
            heapq.heappush(heap, (0.0, _counter, seed_id, 0, "seed", "", 0.0))
            _counter += 1

        while heap:
            neg_pri, _, node_id, depth, via_relation, from_node, effective_priority = heapq.heappop(heap)

            if node_id in seen:
                continue
            seen.add(node_id)
            ordered[node_id] = depth
            if via_relation != "seed":
                metadata[node_id] = ExpansionMeta(
                    via_relation=via_relation,
                    from_node=from_node,
                    effective_priority=effective_priority,
                )

            if depth >= max_depth:
                continue

            neighbors_with_data: list[tuple[str, dict]] = []

            if graph.has_node(node_id):
                for _, neighbor, data in graph.edges(node_id, data=True):
                    if neighbor not in seen:
                        neighbors_with_data.append((neighbor, data))

                for predecessor, _, data in graph.in_edges(node_id, data=True):
                    if predecessor not in seen:
                        neighbors_with_data.append((predecessor, data))

            for neighbor, data in neighbors_with_data:
                relationship = data.get("relationship", "relates_to")
                weight = float(data.get("weight", 1.0))

                effective = (
                    self._relation_priority(relationship)
                    * weight
                    * (decay ** depth)
                )

                if effective < min_priority:
                    continue

                heapq.heappush(
                    heap,
                    (-effective, _counter, neighbor, depth + 1, relationship, node_id, effective),
                )
                _counter += 1

        return ordered, metadata

    def _expand_node_depths(
        self,
        graph: nx.DiGraph,
        seed_ids: list[str],
        max_depth: int,
        *,
        min_priority: float = 0.20,
        decay: float = 0.70,
    ) -> dict[str, int]:
        ordered, _ = self._expand_node_depths_with_context(
            graph, seed_ids, max_depth, min_priority=min_priority, decay=decay
        )
        return ordered

    def _ensure_support_coverage(
        self,
        selected_nodes: list[Node],
        candidate_pool: dict[str, Node],
        graph: nx.DiGraph,
        max_nodes: int,
    ) -> list[Node]:
        """Augment selected nodes with supporting context for contradictions, updates, and dependencies."""
        if len(selected_nodes) >= max_nodes:
            return selected_nodes

        coverage_nodes: list[Node] = []
        seen = {node.id for node in selected_nodes}

        for node in selected_nodes:
            if len(selected_nodes) + len(coverage_nodes) >= max_nodes:
                break

            if graph.has_node(node.id):
                # Find supporting edges via MUST_PAIR_RELATIONS
                for _, neighbor, data in graph.edges(node.id, data=True):
                    if neighbor in seen or neighbor not in candidate_pool:
                        continue
                    relationship = data.get("relationship", "relates_to")
                    if relationship in MUST_PAIR_RELATIONS:
                        support_node = candidate_pool[neighbor]
                        if support_node not in coverage_nodes:
                            coverage_nodes.append(support_node)
                            seen.add(neighbor)
                        if len(selected_nodes) + len(coverage_nodes) >= max_nodes:
                            break

                # Find incoming edges (predecessors) with strong relationships
                for predecessor, _, data in graph.in_edges(node.id, data=True):
                    if predecessor in seen or predecessor not in candidate_pool:
                        continue
                    relationship = data.get("relationship", "relates_to")
                    if relationship in MUST_PAIR_RELATIONS:
                        support_node = candidate_pool[predecessor]
                        if support_node not in coverage_nodes:
                            coverage_nodes.append(support_node)
                            seen.add(predecessor)
                        if len(selected_nodes) + len(coverage_nodes) >= max_nodes:
                            break

        return selected_nodes + coverage_nodes[: max_nodes - len(selected_nodes)]

    def _build_topic_partition(self, graph: nx.Graph, nodes: list[Node]) -> dict[str, int]:
        if graph.number_of_edges() == 0:
            return {node.id: index for index, node in enumerate(nodes)}
        try:
            import community  # type: ignore[import-not-found]

            return community.best_partition(graph)
        except ImportError:  # pragma: no cover
            communities = nx.algorithms.community.greedy_modularity_communities(graph)
            partition: dict[str, int] = {}
            for cluster_id, members in enumerate(communities):
                for member in members:
                    partition[str(member)] = cluster_id
            return partition

    

    def _fetch_edges_for_nodes(
        self,
        connection: sqlite3.Connection,
        node_ids: list[str],
    ) -> list[Edge]:
        if not node_ids:
            return []
        placeholders = ", ".join("?" for _ in node_ids)
        rows = connection.execute(
            f"""
            SELECT id, source_id, target_id, relationship, weight, metadata, created_at, tenant_id
            FROM edges
            WHERE tenant_id = ?
              AND source_id IN ({placeholders})
              AND target_id IN ({placeholders})
            ORDER BY created_at ASC
            """,
            (self.tenant_id, *node_ids, *node_ids),
        ).fetchall()
        return [self._row_to_edge(row) for row in rows]

    def _increment_access_counts(self, connection: sqlite3.Connection, node_ids: list[str]) -> None:
        if not node_ids:
            return
        placeholders = ", ".join("?" for _ in node_ids)
        connection.execute(
            f"""
            UPDATE nodes
            SET access_count = access_count + 1
            WHERE tenant_id = ? AND id IN ({placeholders})
            """,
            (self.tenant_id, *node_ids),
        )

    def _find_existing_edge(
        self,
        connection: sqlite3.Connection,
        *,
        source_id: str,
        target_id: str,
        relationship: str | RelationType,
    ) -> Edge | None:
        row = connection.execute(
            """
            SELECT id, source_id, target_id, relationship, weight, metadata, created_at, tenant_id
            FROM edges
            WHERE tenant_id = ? AND source_id = ? AND target_id = ? AND relationship = ?
            LIMIT 1
            """,
            (self.tenant_id, source_id, target_id, normalize_relationship(relationship)),
        ).fetchone()
        return self._row_to_edge(row) if row is not None else None

    def _most_connected_node_ids(
        self,
        connection: sqlite3.Connection,
        *,
        limit: int,
        agent_id: str = "",
        project: str = "",
        session_id: str = "",
    ) -> list[str]:
        rows = connection.execute(
            """
            SELECT n.id, n.agent_id, n.project, n.session_id, n.label, n.content, n.node_type, n.tags, n.source_prompt,
                   n.evidence_records, n.valid_from, n.valid_to, n.created_at, n.updated_at, n.access_count, n.tenant_id,
                   COUNT(e.id) AS connection_count
            FROM nodes AS n
            LEFT JOIN edges AS e ON (n.id = e.source_id OR n.id = e.target_id) AND e.tenant_id = ?
            WHERE n.tenant_id = ?
            GROUP BY n.id
            ORDER BY connection_count DESC, n.updated_at DESC
            """,
            (self.tenant_id, self.tenant_id),
        ).fetchall()
        selected: list[str] = []
        for row in rows:
            node = self._row_to_node(row)
            if not _scope_matches(node, agent_id=agent_id, project=project, session_id=session_id):
                continue
            selected.append(str(row["id"]))
            if len(selected) >= limit:
                break
        return selected

    def _find_project_node_ids(
        self,
        connection: sqlite3.Connection,
        *,
        project: str,
        agent_id: str = "",
        session_id: str = "",
        limit: int,
    ) -> list[str]:
        project_lower = project.strip().lower()
        rows = connection.execute(
            """
            SELECT id, agent_id, project, session_id, label, content, node_type, tags, source_prompt,
                   evidence_records, valid_from, valid_to, created_at, updated_at, access_count, tenant_id
            FROM nodes
            WHERE tenant_id = ?
            ORDER BY updated_at DESC
            """
        , (self.tenant_id,)).fetchall()
        scored: list[tuple[str, float, str]] = []
        for row in rows:
            node = self._row_to_node(row)
            if not _scope_matches(node, agent_id=agent_id, project=project, session_id=session_id):
                continue

            tags = json.loads(row["tags"] or "[]")
            tag_match = 1.0 if any(project_lower in {str(tag).lower(), f"project:{str(tag).lower()}"} for tag in tags) else 0.0
            explicit_match = 1.0 if str(row["project"] or "").strip().lower() == project_lower else 0.0
            lexical = lexical_overlap(project, row["label"], row["content"])
            score = max(explicit_match, tag_match, lexical)
            if score <= 0.0:
                continue
            scored.append((row["id"], score, row["updated_at"]))
        scored.sort(key=lambda item: (-item[1], item[2]), reverse=False)
        return [node_id for node_id, _, _ in scored[:limit]]

    def _fetch_edge_row(self, connection: sqlite3.Connection, edge_id: str) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT id, source_id, target_id, relationship, weight, metadata, created_at, tenant_id
            FROM edges
            WHERE id = ? AND tenant_id = ?
            """,
            (edge_id, self.tenant_id),
        ).fetchone()

    def _build_backup_snapshot(self, connection: sqlite3.Connection) -> dict[str, Any]:
        node_rows = connection.execute(
            """
            SELECT id, tenant_id, agent_id, project, session_id, label, content, node_type, tags, source_prompt,
                   evidence_records, valid_from, valid_to, created_at, updated_at, access_count
            FROM nodes
            WHERE tenant_id = ?
            ORDER BY created_at ASC
            """
        , (self.tenant_id,)).fetchall()
        edge_rows = connection.execute(
            """
            SELECT id, tenant_id, source_id, target_id, relationship, weight, metadata, created_at
            FROM edges
            WHERE tenant_id = ?
            ORDER BY created_at ASC
            """
        , (self.tenant_id,)).fetchall()
        return {
            "schema_version": SCHEMA_VERSION,
            "tenant_id": self.tenant_id,
            "nodes": [
                {
                    "id": row["id"],
                    "tenant_id": row["tenant_id"],
                    "agent_id": row["agent_id"] or "",
                    "project": row["project"] or "",
                    "session_id": row["session_id"] or "",
                    "label": row["label"],
                    "content": row["content"],
                    "node_type": row["node_type"],
                    "tags": json.loads(row["tags"] or "[]"),
                    "source_prompt": row["source_prompt"] or "",
                    "evidence_records": [record.model_dump(mode="json") for record in _decode_evidence_records(row["evidence_records"])],
                    "valid_from": row["valid_from"],
                    "valid_to": row["valid_to"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "access_count": int(row["access_count"] or 0),
                }
                for row in node_rows
            ],
            "edges": [
                {
                    "id": row["id"],
                    "tenant_id": row["tenant_id"],
                    "source_id": row["source_id"],
                    "target_id": row["target_id"],
                    "relationship": row["relationship"],
                    "weight": float(row["weight"]),
                    "metadata": json.loads(row["metadata"] or "{}"),
                    "created_at": row["created_at"],
                }
                for row in edge_rows
            ],
        }

    def _insert_snapshot_node(self, connection: sqlite3.Connection, raw_node: dict[str, Any]) -> None:
        embedding = self.embedding_model.to_bytes(self.embedding_model.embed(raw_node["content"]))
        connection.execute(
            """
            INSERT INTO nodes (
                id, tenant_id, agent_id, project, session_id, label, content, node_type, tags, embedding,
                source_prompt, evidence_records, valid_from, valid_to, created_at, updated_at, access_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_node["id"],
                raw_node.get("tenant_id", self.tenant_id),
                raw_node.get("agent_id", ""),
                raw_node.get("project", ""),
                raw_node.get("session_id", ""),
                raw_node["label"],
                raw_node["content"],
                raw_node["node_type"],
                json.dumps(raw_node.get("tags", [])),
                embedding,
                raw_node.get("source_prompt", ""),
                _encode_evidence_records([EvidenceRecord.model_validate(item) for item in raw_node.get("evidence_records", [])]),
                raw_node.get("valid_from"),
                raw_node.get("valid_to"),
                raw_node["created_at"],
                raw_node["updated_at"],
                int(raw_node.get("access_count", 0)),
            ),
        )

    def _update_snapshot_node(self, connection: sqlite3.Connection, raw_node: dict[str, Any]) -> None:
        embedding = self.embedding_model.to_bytes(self.embedding_model.embed(raw_node["content"]))
        connection.execute(
            """
            UPDATE nodes
            SET tenant_id = ?, agent_id = ?, project = ?, session_id = ?, label = ?, content = ?, node_type = ?, tags = ?, embedding = ?,
                source_prompt = ?, evidence_records = ?, valid_from = ?, valid_to = ?,
                created_at = ?, updated_at = ?, access_count = ?
            WHERE id = ? AND tenant_id = ?
            """,
            (
                raw_node.get("tenant_id", self.tenant_id),
                raw_node.get("agent_id", ""),
                raw_node.get("project", ""),
                raw_node.get("session_id", ""),
                raw_node["label"],
                raw_node["content"],
                raw_node["node_type"],
                json.dumps(raw_node.get("tags", [])),
                embedding,
                raw_node.get("source_prompt", ""),
                _encode_evidence_records([EvidenceRecord.model_validate(item) for item in raw_node.get("evidence_records", [])]),
                raw_node.get("valid_from"),
                raw_node.get("valid_to"),
                raw_node["created_at"],
                raw_node["updated_at"],
                int(raw_node.get("access_count", 0)),
                raw_node["id"],
                self.tenant_id,
            ),
        )

    def _insert_snapshot_edge(self, connection: sqlite3.Connection, raw_edge: dict[str, Any]) -> None:
        self._require_node(connection, raw_edge["source_id"])
        self._require_node(connection, raw_edge["target_id"])
        connection.execute(
            """
            INSERT INTO edges (id, tenant_id, source_id, target_id, relationship, weight, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_edge["id"],
                raw_edge.get("tenant_id", self.tenant_id),
                raw_edge["source_id"],
                raw_edge["target_id"],
                raw_edge["relationship"],
                float(raw_edge.get("weight", 1.0)),
                json.dumps(raw_edge.get("metadata", {})),
                raw_edge["created_at"],
            ),
        )

    def _update_snapshot_edge(self, connection: sqlite3.Connection, raw_edge: dict[str, Any]) -> None:
        self._require_node(connection, raw_edge["source_id"])
        self._require_node(connection, raw_edge["target_id"])
        connection.execute(
            """
            UPDATE edges
            SET tenant_id = ?, source_id = ?, target_id = ?, relationship = ?, weight = ?, metadata = ?, created_at = ?
            WHERE id = ? AND tenant_id = ?
            """,
            (
                raw_edge.get("tenant_id", self.tenant_id),
                raw_edge["source_id"],
                raw_edge["target_id"],
                raw_edge["relationship"],
                float(raw_edge.get("weight", 1.0)),
                json.dumps(raw_edge.get("metadata", {})),
                raw_edge["created_at"],
                raw_edge["id"],
                self.tenant_id,
            ),
        )
