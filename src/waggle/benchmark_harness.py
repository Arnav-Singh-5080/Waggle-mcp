from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import statistics
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from waggle.embeddings import EmbeddingModel
from waggle.graph import MemoryGraph
from waggle.intelligence import extract_conversation_candidates, infer_temporal_hints
from waggle.models import NodeType, RelationType

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURES_DIR = ROOT / "benchmarks" / "fixtures"
DEFAULT_DEDUP_THRESHOLDS = [0.82, 0.85, 0.88, 0.9, 0.92, 0.95, 0.97]


class BenchmarkRuntimeError(RuntimeError):
    """Raised when a requested benchmark cannot be executed honestly."""


@dataclass
class MetricSummary:
    metric: str
    backend: str
    passed: int
    total: int
    accuracy: float
    case_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparativeCaseResult:
    query_id: str
    system: str
    task_family: str
    hit_at_k: bool
    exact_support: bool
    context_tokens: int
    retrieved_ids: list[str]
    gold_support_ids: list[str]
    failure_label: str = ""
    retrieval_mode: str = ""
    max_depth: int = 0


@dataclass
class BenchmarkReport:
    fixtures: dict[str, Any]
    metrics: list[MetricSummary]
    errors: list[str] = field(default_factory=list)
    threshold_sweep: list[MetricSummary] = field(default_factory=list)
    comparative: dict[str, Any] = field(default_factory=dict)
    stress_eval: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixtures": self.fixtures,
            "metrics": [asdict(metric) for metric in self.metrics],
            "errors": list(self.errors),
            "threshold_sweep": [asdict(metric) for metric in self.threshold_sweep],
            "comparative": self.comparative,
            "stress_eval": self.stress_eval,
        }


@dataclass
class RagChunk:
    chunk_id: str
    scenario_id: str
    session_id: str
    text: str
    timestamp: str
    support_fact_ids: list[str]


COMPARATIVE_TASK_QUERY_POLICY: dict[str, dict[str, Any]] = {
    "factual_recall": {"retrieval_mode": "flat", "max_depth": 0},
    "temporal_latest": {"retrieval_mode": "flat", "max_depth": 0},
    "temporal_original": {"retrieval_mode": "flat", "max_depth": 0},
    "multi_session_change": {"retrieval_mode": "graph", "max_depth": 2},
    "cross_scenario_synthesis": {"retrieval_mode": "graph", "max_depth": 2},
    "decision_delta": {"retrieval_mode": "graph", "max_depth": 2},
    "adversarial_paraphrase": {"retrieval_mode": "graph", "max_depth": 1},
}

COMPARATIVE_TRANSCRIPT_RE = re.compile(
    r"^\s*User:\s*(?P<user>.*?)\s*(?:Agent|Assistant):\s*(?P<assistant>.*?)\s*$",
    re.IGNORECASE | re.DOTALL,
)


def load_benchmark_fixtures(fixtures_dir: Path | str = DEFAULT_FIXTURES_DIR) -> dict[str, Any]:
    base = Path(fixtures_dir)
    extraction_cases = json.loads((base / "extraction_cases.json").read_text(encoding="utf-8"))
    retrieval_cases = json.loads((base / "retrieval_cases.json").read_text(encoding="utf-8"))
    dedup_cases = json.loads((base / "dedup_cases.json").read_text(encoding="utf-8"))
    comparative_eval = json.loads((base / "comparative_eval.json").read_text(encoding="utf-8"))
    query_stress_cases = json.loads((base / "query_stress_cases.json").read_text(encoding="utf-8"))

    # Compute repo-relative path
    try:
        relative_dir = os.path.relpath(base, ROOT)
    except ValueError:
        # If relpath fails (e.g., different drives on Windows), fall back to absolute
        relative_dir = str(base)

    return {
        "base_dir": relative_dir,
        "extraction_cases": extraction_cases,
        "retrieval_cases": retrieval_cases,
        "dedup_cases": dedup_cases,
        "comparative_eval": comparative_eval,
        "query_stress_cases": query_stress_cases,
    }


def _embedding_benchmark_error(exc: Exception, embedding_model: Any) -> BenchmarkRuntimeError:
    model_name = getattr(embedding_model, "model_name", "all-MiniLM-L6-v2")
    return BenchmarkRuntimeError(
        "Embedding-backed benchmarks require a locally available sentence-transformer model "
        f"('{model_name}'). Pre-cache the model before running retrieval/dedup benchmarks. "
        f"Original error: {exc}"
    )


def _graph(
    embedding_model: Any,
    *,
    dedup_similarity_threshold: float = 0.97,
    dedup_same_label_threshold: float = 0.9,
) -> MemoryGraph:
    tmpdir = tempfile.TemporaryDirectory()
    graph = MemoryGraph(
        Path(tmpdir.name) / "benchmark.db",
        embedding_model,
        dedup_similarity_threshold=dedup_similarity_threshold,
        dedup_same_label_threshold=dedup_same_label_threshold,
    )
    setattr(graph, "_benchmark_tmpdir", tmpdir)
    return graph


def _normalize_node_type(value: Any) -> str:
    if isinstance(value, NodeType):
        return value.value
    return str(value)


def _score_extraction_case(case: dict[str, Any], found_types: set[str]) -> bool:
    expected_types = set(case.get("expected_node_types", []))
    min_type_matches = int(case.get("min_type_matches", len(expected_types)))
    forbidden_types = set(case.get("forbidden_node_types", []))

    if expected_types:
        if len(found_types & expected_types) < min_type_matches:
            return False
    elif found_types:
        return False

    return not bool(found_types & forbidden_types)


def _estimate_tokens(text: str) -> int:
    normalized = text.strip()
    if not normalized:
        return 0
    return max(1, math.ceil(len(normalized) / 4))


def _comparative_query_config(task_family: str) -> dict[str, Any]:
    return dict(COMPARATIVE_TASK_QUERY_POLICY.get(task_family, {"retrieval_mode": "flat", "max_depth": 0}))


def _policy_summary() -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for task_family, config in COMPARATIVE_TASK_QUERY_POLICY.items():
        entry = grouped.setdefault(
            str(config["retrieval_mode"]),
            {"task_families": [], "max_depths": []},
        )
        entry["task_families"].append(task_family)
        entry["max_depths"].append(int(config["max_depth"]))
    for entry in grouped.values():
        entry["task_families"].sort()
        entry["max_depths"] = sorted(set(entry["max_depths"]))
    return grouped


def _parse_comparative_transcript(transcript: str) -> tuple[str, str]:
    match = COMPARATIVE_TRANSCRIPT_RE.match(transcript.strip())
    if match is None:
        raise BenchmarkRuntimeError(f"Comparative fixture transcript is malformed: {transcript!r}")
    return match.group("user").strip(), match.group("assistant").strip()


def _percentile(values: list[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return (ordered[lower] * (1 - weight)) + (ordered[upper] * weight)


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _set_node_timestamp(graph: MemoryGraph, node_id: str, timestamp: str) -> None:
    with graph._lock, graph._connect() as connection:  # noqa: SLF001 - benchmark helper
        connection.execute(
            """
            UPDATE nodes
            SET valid_from = COALESCE(valid_from, ?), created_at = ?, updated_at = ?
            WHERE id = ? AND tenant_id = ?
            """,
            (timestamp, timestamp, timestamp, node_id, graph.tenant_id),
        )


def _extract_fact_id(tags: list[str]) -> str | None:
    for tag in tags:
        if tag.startswith("fact_id:"):
            return tag.split(":", 1)[1]
    return None


def _set_session_timestamp(graph: MemoryGraph, session_id: str, timestamp: str) -> None:
    with graph._lock, graph._connect() as connection:  # noqa: SLF001 - benchmark helper
        connection.execute(
            """
            UPDATE transcript_records
            SET observed_at = ?
            WHERE tenant_id = ? AND session_id = ?
            """,
            (timestamp, graph.tenant_id, session_id),
        )


def run_extraction_benchmark(
    cases: list[dict[str, Any]],
    *,
    backend: Literal["regex"],
) -> MetricSummary:
    passed = 0
    for case in cases:
        candidates = extract_conversation_candidates(
            user_message=case["user_message"],
            assistant_response=case["assistant_response"],
        )

        found_types = {_normalize_node_type(candidate["node_type"]) for candidate in candidates}
        if _score_extraction_case(case, found_types):
            passed += 1

    total = len(cases)
    return MetricSummary(
        metric="extraction",
        backend=backend,
        passed=passed,
        total=total,
        accuracy=passed / total if total else 0.0,
        case_count=total,
        metadata={"runtime": "deterministic-regex"},
    )


def run_retrieval_benchmark(
    retrieval_fixtures: dict[str, Any],
    *,
    embedding_model: Any,
) -> MetricSummary:
    try:
        graph = _graph(embedding_model)
        for node in retrieval_fixtures["nodes"]:
            graph.add_node(
                label=node["label"],
                content=node["content"],
                node_type=NodeType(node["node_type"]),
                tags=["benchmark"],
            )

        passed = 0
        queries = retrieval_fixtures["queries"]
        for case in queries:
            result = graph.query(query=case["query"], max_nodes=5, max_depth=0)
            labels = [node.label for node in result.nodes]
            if any(case["expected_label_contains"].lower() in label.lower() for label in labels):
                passed += 1
    except Exception as exc:
        raise _embedding_benchmark_error(exc, embedding_model) from exc

    total = len(queries)
    return MetricSummary(
        metric="retrieval",
        backend="semantic-query",
        passed=passed,
        total=total,
        accuracy=passed / total if total else 0.0,
        case_count=total,
        metadata={"corpus_nodes": len(retrieval_fixtures["nodes"]), "top_k": 5},
    )


def run_dedup_benchmark(
    cases: list[dict[str, Any]],
    *,
    embedding_model: Any,
    dedup_threshold: float,
) -> MetricSummary:
    try:
        passed = 0
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        for case in cases:
            graph = _graph(
                embedding_model,
                dedup_similarity_threshold=dedup_threshold,
                dedup_same_label_threshold=dedup_threshold,
            )
            first = case["first"]
            second = case["second"]
            node_type = NodeType(case["node_type"])
            graph.add_node(label=first["label"], content=first["content"], node_type=node_type)
            second_result = graph.add_node(label=second["label"], content=second["content"], node_type=node_type)
            did_dedup = not second_result.created
            expected = bool(case["should_dedup"])

            if did_dedup == expected:
                passed += 1
                if expected:
                    true_positives += 1
                else:
                    true_negatives += 1
            elif expected:
                false_negatives += 1
            else:
                false_positives += 1
    except Exception as exc:
        raise _embedding_benchmark_error(exc, embedding_model) from exc

    total = len(cases)
    return MetricSummary(
        metric="deduplication",
        backend="semantic-dedup",
        passed=passed,
        total=total,
        accuracy=passed / total if total else 0.0,
        case_count=total,
        metadata={
            "threshold": dedup_threshold,
            "positive_cases": sum(1 for case in cases if case["should_dedup"]),
            "negative_cases": sum(1 for case in cases if not case["should_dedup"]),
            "true_positives": true_positives,
            "true_negatives": true_negatives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        },
    )


def choose_best_dedup_threshold(
    cases: list[dict[str, Any]],
    *,
    embedding_model: Any,
    thresholds: list[float] | None = None,
) -> tuple[MetricSummary, list[MetricSummary]]:
    candidates = thresholds or DEFAULT_DEDUP_THRESHOLDS
    sweep = [
        run_dedup_benchmark(cases, embedding_model=embedding_model, dedup_threshold=threshold)
        for threshold in candidates
    ]
    best = max(
        sweep,
        key=lambda summary: (
            summary.accuracy,
            summary.metadata["true_negatives"],
            summary.metadata["true_positives"],
            summary.metadata["threshold"],
        ),
    )
    return best, sweep


def _build_rag_chunks(
    comparative_eval: dict[str, Any],
    *,
    chunk_size_words: int,
    overlap_words: int,
) -> list[RagChunk]:
    chunks: list[RagChunk] = []
    for scenario in comparative_eval["scenarios"]:
        for session in scenario["sessions"]:
            words = session["transcript"].split()
            step = max(1, chunk_size_words - overlap_words)
            for index, start in enumerate(range(0, len(words), step)):
                end = start + chunk_size_words
                chunk_words = words[start:end]
                if not chunk_words:
                    continue
                chunks.append(
                    RagChunk(
                        chunk_id=f"{session['id']}::chunk{index}",
                        scenario_id=scenario["id"],
                        session_id=session["id"],
                        text=" ".join(chunk_words),
                        timestamp=session["timestamp"],
                        support_fact_ids=list(session.get("support_fact_ids", [])),
                    )
                )
                if end >= len(words):
                    break
    return chunks


def _score_rag_chunk(
    query: str,
    chunk: RagChunk,
    query_embedding: Any,
    chunk_embedding: Any,
    embedding_model: Any,
    *,
    lexical_weight: float,
    temporal_weight: float,
) -> float:
    similarity = max(embedding_model.cosine_similarity(query_embedding, chunk_embedding), 0.0)
    lexical_tokens = set(query.lower().split())
    chunk_tokens = set(chunk.text.lower().split())
    lexical_overlap = len(lexical_tokens & chunk_tokens) / max(len(lexical_tokens), 1)
    temporal_hints = infer_temporal_hints(query)
    timestamp = _parse_iso(chunk.timestamp)
    temporal_score = 0.0
    if temporal_hints.recency_mode == "latest":
        temporal_score = timestamp.timestamp()
    elif temporal_hints.recency_mode == "oldest":
        temporal_score = -timestamp.timestamp()
    return (0.72 * similarity) + (lexical_weight * lexical_overlap) + (temporal_weight * temporal_score)


def _rank_rag_chunks(
    query: str,
    chunks: list[RagChunk],
    embedding_model: Any,
    *,
    top_k: int,
    lexical_weight: float,
    temporal_weight: float,
) -> list[RagChunk]:
    chunk_embeddings = [embedding_model.embed(chunk.text) for chunk in chunks]
    query_embedding = embedding_model.embed(query)
    scored = [
        (
            _score_rag_chunk(
                query,
                chunk,
                query_embedding,
                chunk_embedding,
                embedding_model,
                lexical_weight=lexical_weight,
                temporal_weight=temporal_weight,
            ),
            chunk.timestamp,
            chunk.chunk_id,
            chunk,
        )
        for chunk, chunk_embedding in zip(chunks, chunk_embeddings, strict=True)
    ]
    return [
        item[3]
        for item in sorted(scored, key=lambda item: (-item[0], item[1], item[2]))[:top_k]
    ]


def _aggregate_case_results(results: list[ComparativeCaseResult]) -> dict[str, Any]:
    hit_scores = [1 if result.hit_at_k else 0 for result in results]
    exact_scores = [1 if result.exact_support else 0 for result in results]
    token_costs = [result.context_tokens for result in results]
    by_family: dict[str, dict[str, Any]] = {}
    by_failure_label: dict[str, dict[str, Any]] = {}

    for family in sorted({result.task_family for result in results}):
        family_results = [result for result in results if result.task_family == family]
        by_family[family] = {
            "case_count": len(family_results),
            "hit_at_k": sum(1 if result.hit_at_k else 0 for result in family_results) / len(family_results),
            "exact_support": sum(1 if result.exact_support else 0 for result in family_results) / len(family_results),
        }

    labeled_results = [result for result in results if result.failure_label]
    for label in sorted({result.failure_label for result in labeled_results}):
        failure_results = [result for result in labeled_results if result.failure_label == label]
        by_failure_label[label] = {
            "case_count": len(failure_results),
            "hit_at_k": sum(1 if result.hit_at_k else 0 for result in failure_results) / len(failure_results),
            "exact_support": sum(1 if result.exact_support else 0 for result in failure_results) / len(failure_results),
        }

    retrieval_modes = sorted({result.retrieval_mode for result in results if result.retrieval_mode})
    by_retrieval_mode: dict[str, dict[str, Any]] = {}
    for retrieval_mode in retrieval_modes:
        mode_results = [result for result in results if result.retrieval_mode == retrieval_mode]
        by_retrieval_mode[retrieval_mode] = {
            "case_count": len(mode_results),
            "hit_at_k": sum(1 if result.hit_at_k else 0 for result in mode_results) / len(mode_results),
            "exact_support": sum(1 if result.exact_support else 0 for result in mode_results) / len(mode_results),
            "max_depths": sorted({result.max_depth for result in mode_results}),
            "task_families": sorted({result.task_family for result in mode_results}),
        }

    summary = {
        "case_count": len(results),
        "hit_at_k": sum(hit_scores) / len(hit_scores) if hit_scores else 0.0,
        "exact_support": sum(exact_scores) / len(exact_scores) if exact_scores else 0.0,
        "context_tokens": {
            "mean": statistics.mean(token_costs) if token_costs else 0.0,
            "median": statistics.median(token_costs) if token_costs else 0.0,
            "p95": _percentile(token_costs, 0.95),
        },
        "by_task_family": by_family,
        "by_failure_label": by_failure_label,
    }
    if by_retrieval_mode:
        summary["by_retrieval_mode"] = by_retrieval_mode
    return summary


def _build_comparative_graph(
    comparative_eval: dict[str, Any],
    *,
    embedding_model: Any,
) -> MemoryGraph:
    graph = _graph(embedding_model)
    for scenario in comparative_eval["scenarios"]:
        fact_node_ids: dict[str, str] = {}
        for fact in scenario["facts"]:
            result = graph.add_node(
                label=fact["label"],
                content=fact["content"],
                node_type=NodeType(fact["node_type"]),
                tags=["benchmark", f"fact_id:{fact['id']}", f"scenario:{scenario['id']}"],
                project="comparative-benchmark",
            )
            fact_node_ids[fact["id"]] = result.node.id
            _set_node_timestamp(graph, result.node.id, fact["timestamp"])
        _add_declared_comparative_relations(graph, scenario, fact_node_ids)
    return graph


def _iter_declared_comparative_relations(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    for relation in scenario.get("relations", []):
        if not isinstance(relation, dict):
            continue
        source_fact_id = str(relation.get("source_fact_id") or relation.get("source") or "").strip()
        target_fact_id = str(relation.get("target_fact_id") or relation.get("target") or "").strip()
        relationship = str(relation.get("relationship") or "").strip().lower()
        if not source_fact_id or not target_fact_id or not relationship:
            continue
        relations.append(
            {
                "source_fact_id": source_fact_id,
                "target_fact_id": target_fact_id,
                "relationship": relationship,
                "metadata": dict(relation.get("metadata") or {}),
            }
        )
    for supersede in scenario.get("supersedes", []):
        if not isinstance(supersede, dict):
            continue
        source_fact_id = str(supersede.get("newer_fact_id") or supersede.get("source_fact_id") or "").strip()
        target_fact_id = str(supersede.get("older_fact_id") or supersede.get("target_fact_id") or "").strip()
        if not source_fact_id or not target_fact_id:
            continue
        relations.append(
            {
                "source_fact_id": source_fact_id,
                "target_fact_id": target_fact_id,
                "relationship": RelationType.UPDATES.value,
                "metadata": {"origin": "comparative-supersedes", "scenario": scenario["id"]},
            }
        )
        if supersede.get("contradicts", True):
            relations.append(
                {
                    "source_fact_id": source_fact_id,
                    "target_fact_id": target_fact_id,
                    "relationship": RelationType.CONTRADICTS.value,
                    "metadata": {
                        "origin": "comparative-supersedes",
                        "scenario": scenario["id"],
                        "kind": "superseded-state",
                    },
                }
            )
    return relations


def _add_declared_comparative_relations(
    graph: MemoryGraph,
    scenario: dict[str, Any],
    fact_node_ids: dict[str, str],
) -> None:
    for relation in _iter_declared_comparative_relations(scenario):
        source_id = fact_node_ids.get(relation["source_fact_id"])
        target_id = fact_node_ids.get(relation["target_fact_id"])
        if source_id is None or target_id is None or source_id == target_id:
            continue
        metadata = {"scenario": scenario["id"], **relation["metadata"]}
        graph.add_edge(
            source_id=source_id,
            target_id=target_id,
            relationship=relation["relationship"],
            metadata=metadata,
        )


def _build_comparative_graph_with_sessions(
    comparative_eval: dict[str, Any],
    *,
    embedding_model: Any,
) -> MemoryGraph:
    graph = _graph(embedding_model)
    fact_node_ids: dict[str, str] = {}

    for scenario in comparative_eval["scenarios"]:
        ordered_facts = sorted(scenario["facts"], key=lambda item: item["timestamp"])
        for fact in ordered_facts:
            result = graph.add_node(
                label=fact["label"],
                content=fact["content"],
                node_type=NodeType(fact["node_type"]),
                tags=["benchmark", f"fact_id:{fact['id']}", f"scenario:{scenario['id']}"],
                project="comparative-benchmark",
            )
            fact_node_ids[fact["id"]] = result.node.id
            _set_node_timestamp(graph, result.node.id, fact["timestamp"])
        _add_declared_comparative_relations(graph, scenario, fact_node_ids)

        for session in scenario["sessions"]:
            user_message, assistant_response = _parse_comparative_transcript(session["transcript"])
            observation = graph.observe_conversation(
                user_message=user_message,
                assistant_response=assistant_response,
                project="comparative-benchmark",
                session_id=session["id"],
            )
            _set_session_timestamp(graph, session["id"], session["timestamp"])
            for stored_node in observation.stored_nodes:
                _set_node_timestamp(graph, stored_node.id, session["timestamp"])
            for fact_id in session.get("support_fact_ids", []):
                target_id = fact_node_ids.get(fact_id)
                if target_id is None:
                    continue
                for stored_node in observation.stored_nodes:
                    if target_id == stored_node.id:
                        continue
                    graph.add_edge(
                        source_id=target_id,
                        target_id=stored_node.id,
                        relationship=RelationType.DEPENDS_ON,
                        metadata={"origin": "comparative-support", "session_id": session["id"]},
                    )

    return graph


def _run_waggle_system(
    comparative_eval: dict[str, Any],
    *,
    embedding_model: Any,
    top_k: int,
) -> tuple[dict[str, Any], list[ComparativeCaseResult]]:
    graph = _build_comparative_graph_with_sessions(comparative_eval, embedding_model=embedding_model)

    results: list[ComparativeCaseResult] = []
    for case in comparative_eval["queries"]:
        query_config = _comparative_query_config(case["task_family"])
        subgraph = graph.query(
            query=case["query"],
            max_nodes=top_k,
            max_depth=int(query_config["max_depth"]),
        )
        retrieved_fact_ids = [
            fact_id
            for node in subgraph.nodes
            for fact_id in [_extract_fact_id(node.tags)]
            if fact_id is not None
        ]
        union_ids = set(retrieved_fact_ids)
        gold_ids = set(case["gold_support_ids"])
        context_text = "\n".join(f"{node.label}: {node.content}" for node in subgraph.nodes)
        results.append(
            ComparativeCaseResult(
                query_id=case["id"],
                system="waggle",
                task_family=case["task_family"],
                failure_label=case.get("failure_label", ""),
                hit_at_k=bool(union_ids & gold_ids),
                exact_support=gold_ids.issubset(union_ids),
                context_tokens=_estimate_tokens(context_text),
                retrieved_ids=retrieved_fact_ids,
                gold_support_ids=list(case["gold_support_ids"]),
                retrieval_mode=str(query_config["retrieval_mode"]),
                max_depth=int(query_config["max_depth"]),
            )
        )

    return {
        "system": "waggle",
        "parameters": {"top_k": top_k, "task_family_max_depth": {key: value["max_depth"] for key, value in COMPARATIVE_TASK_QUERY_POLICY.items()}},
        "query_policy": _policy_summary(),
    }, results


def _run_rag_system(
    comparative_eval: dict[str, Any],
    *,
    embedding_model: Any,
    system: Literal["rag_naive", "rag_tuned"],
) -> tuple[dict[str, Any], list[ComparativeCaseResult]]:
    if system == "rag_naive":
        config = {"chunk_size_words": 120, "overlap_words": 20, "top_k": 5, "lexical_weight": 0.0, "temporal_weight": 0.0}
    else:
        config = {"chunk_size_words": 80, "overlap_words": 10, "top_k": 8, "lexical_weight": 0.18, "temporal_weight": 0.000000001}

    chunks = _build_rag_chunks(
        comparative_eval,
        chunk_size_words=config["chunk_size_words"],
        overlap_words=config["overlap_words"],
    )
    results: list[ComparativeCaseResult] = []
    for case in comparative_eval["queries"]:
        ranked = _rank_rag_chunks(
            case["query"],
            chunks,
            embedding_model,
            top_k=config["top_k"],
            lexical_weight=config["lexical_weight"],
            temporal_weight=config["temporal_weight"],
        )
        retrieved_ids = [fact_id for chunk in ranked for fact_id in chunk.support_fact_ids]
        union_ids = set(retrieved_ids)
        gold_ids = set(case["gold_support_ids"])
        context_text = "\n".join(chunk.text for chunk in ranked)
        results.append(
            ComparativeCaseResult(
                query_id=case["id"],
                system=system,
                task_family=case["task_family"],
                failure_label=case.get("failure_label", ""),
                hit_at_k=bool(union_ids & gold_ids),
                exact_support=gold_ids.issubset(union_ids),
                context_tokens=_estimate_tokens(context_text),
                retrieved_ids=retrieved_ids,
                gold_support_ids=list(case["gold_support_ids"]),
            )
        )
    return {"system": system, "parameters": config}, results


def run_comparative_evaluation(
    comparative_eval: dict[str, Any],
    *,
    embedding_model: Any,
    systems: list[str],
) -> dict[str, Any]:
    try:
        system_summaries: dict[str, Any] = {}
        all_case_results: list[dict[str, Any]] = []
        for system in systems:
            if system == "waggle":
                info, results = _run_waggle_system(comparative_eval, embedding_model=embedding_model, top_k=5)
            elif system in {"rag_naive", "rag_tuned"}:
                info, results = _run_rag_system(
                    comparative_eval,
                    embedding_model=embedding_model,
                    system=system,
                )
            else:
                raise BenchmarkRuntimeError(f"Unsupported comparative system: {system}")

            summary = _aggregate_case_results(results)
            summary["parameters"] = info["parameters"]
            if "query_policy" in info:
                summary["query_policy"] = info["query_policy"]
            system_summaries[system] = summary
            all_case_results.extend(asdict(result) for result in results)

        return {
            "corpus": {
                "scenario_count": len(comparative_eval["scenarios"]),
                "query_count": len(comparative_eval["queries"]),
                "task_families": sorted({case["task_family"] for case in comparative_eval["queries"]}),
            },
            "systems": system_summaries,
            "per_case": all_case_results,
            "failure_protocol": list(comparative_eval.get("failure_protocol", [])),
        }
    except BenchmarkRuntimeError:
        raise
    except Exception as exc:
        raise _embedding_benchmark_error(exc, embedding_model) from exc


def run_query_stress_evaluation(
    comparative_eval: dict[str, Any],
    query_stress_cases: dict[str, Any],
    *,
    embedding_model: Any,
) -> dict[str, Any]:
    try:
        graph = _build_comparative_graph(comparative_eval, embedding_model=embedding_model)
        system_configs = {
            "graph_raw": {"top_k": 5, "max_depth": 0},
            "graph_hybrid": {"top_k": 5, "max_depth": 2},
        }
        system_summaries: dict[str, Any] = {}
        all_case_results: list[dict[str, Any]] = []

        for system_name, config in system_configs.items():
            results: list[ComparativeCaseResult] = []
            for case in query_stress_cases["queries"]:
                subgraph = graph.query(
                    query=case["query"],
                    max_nodes=config["top_k"],
                    max_depth=config["max_depth"],
                )
                retrieved_fact_ids = [
                    fact_id
                    for node in subgraph.nodes
                    for fact_id in [_extract_fact_id(node.tags)]
                    if fact_id is not None
                ]
                union_ids = set(retrieved_fact_ids)
                gold_ids = set(case["gold_support_ids"])
                context_text = "\n".join(f"{node.label}: {node.content}" for node in subgraph.nodes)
                results.append(
                    ComparativeCaseResult(
                        query_id=case["id"],
                        system=system_name,
                        task_family=case["task_family"],
                        failure_label=case.get("failure_label", ""),
                        hit_at_k=bool(union_ids & gold_ids),
                        exact_support=gold_ids.issubset(union_ids),
                        context_tokens=_estimate_tokens(context_text),
                        retrieved_ids=retrieved_fact_ids,
                        gold_support_ids=list(case["gold_support_ids"]),
                    )
                )
            summary = _aggregate_case_results(results)
            summary["parameters"] = config
            system_summaries[system_name] = summary
            all_case_results.extend(asdict(result) for result in results)

        return {
            "case_count": len(query_stress_cases["queries"]),
            "task_families": sorted({case["task_family"] for case in query_stress_cases["queries"]}),
            "systems": system_summaries,
            "per_case": all_case_results,
        }
    except BenchmarkRuntimeError:
        raise
    except Exception as exc:
        raise _embedding_benchmark_error(exc, embedding_model) from exc


def _format_metric(metric: MetricSummary) -> str:
    extras = []
    if metric.metric == "extraction":
        extras.append(f"backend={metric.backend}")
        if metric.metadata.get("model"):
            extras.append(f"model={metric.metadata['model']}")
        if metric.metadata.get("timeout_seconds") is not None:
            extras.append(f"timeout={metric.metadata['timeout_seconds']}s")
    elif metric.metric == "retrieval":
        extras.append(f"backend={metric.backend}")
        extras.append(f"corpus_nodes={metric.metadata['corpus_nodes']}")
    elif metric.metric == "deduplication":
        extras.append(f"backend={metric.backend}")
        extras.append(f"threshold={metric.metadata['threshold']:.2f}")
        extras.append(
            f"positives={metric.metadata['positive_cases']}, negatives={metric.metadata['negative_cases']}"
        )
    extras.append(f"cases={metric.case_count}")
    return (
        f"{metric.metric:<14} {metric.passed}/{metric.total} = {metric.accuracy:.0%} "
        f"({' | '.join(extras)})"
    )


def build_markdown_summary(report: BenchmarkReport) -> str:
    if not hasattr(report, "comparative") or report.comparative is None:
        lines = ["# Waggle Benchmark Report", ""]
        if hasattr(report, "error") and report.error:
            lines.append(f"**Error:** {report.error}")
        if hasattr(report, "status") and report.status:
            lines.append(f"**Status:** {report.status}")
        if hasattr(report, "failure_protocol") and report.failure_protocol:
            lines.extend(["", "## Failure Protocol", ""])
            for item in report.failure_protocol:
                lines.append(f"- {item}")
        return "\n".join(lines) + "\n"

    lines = [
        "# Waggle Comparative Evaluation",
        "",
        f"- Scenarios: {report.comparative['corpus']['scenario_count']}",
        f"- Queries: {report.comparative['corpus']['query_count']}",
        f"- Task families: {', '.join(report.comparative['corpus']['task_families'])}",
        "",
        "| System | Hit@k | Exact support | Mean tokens | Median tokens | p95 tokens |",
        "|--------|-------|---------------|-------------|---------------|------------|",
    ]
    for system, metrics in report.comparative["systems"].items():
        lines.append(
            f"| {system} | {metrics['hit_at_k']:.0%} | {metrics['exact_support']:.0%} | "
            f"{metrics['context_tokens']['mean']:.1f} | {metrics['context_tokens']['median']:.1f} | "
            f"{metrics['context_tokens']['p95']:.1f} |"
        )
    waggle_metrics = report.comparative["systems"].get("waggle", {})
    if waggle_metrics.get("query_policy"):
        lines.extend(["", "## Waggle Query Policy", ""])
        for mode, details in waggle_metrics["query_policy"].items():
            lines.append(
                f"- `{mode}`: max_depth={','.join(str(value) for value in details['max_depths'])}; "
                f"families={', '.join(details['task_families'])}"
            )
    if waggle_metrics.get("by_retrieval_mode"):
        lines.extend(
            [
                "",
                "## Waggle Mode Breakdown",
                "",
                "| Mode | Cases | Max depth | Hit@k | Exact support |",
                "|------|-------|-----------|-------|---------------|",
            ]
        )
        for mode, metrics in waggle_metrics["by_retrieval_mode"].items():
            lines.append(
                f"| {mode} | {metrics['case_count']} | {', '.join(str(value) for value in metrics['max_depths'])} | "
                f"{metrics['hit_at_k']:.0%} | {metrics['exact_support']:.0%} |"
            )
    lines.extend(
        [
            "",
            "## Failure Protocol",
            "",
        ]
    )
    for item in report.comparative.get("failure_protocol", []):
        lines.append(f"- {item}")
    if report.stress_eval:
        lines.extend(
            [
                "",
                "## Query Stress Eval",
                "",
                f"- Cases: {report.stress_eval['case_count']}",
                f"- Families: {', '.join(report.stress_eval['task_families'])}",
                "",
                "| System | Hit@k | Exact support | Mean tokens |",
                "|--------|-------|---------------|-------------|",
            ]
        )
        for system, metrics in report.stress_eval["systems"].items():
            lines.append(
                f"| {system} | {metrics['hit_at_k']:.0%} | {metrics['exact_support']:.0%} | "
                f"{metrics['context_tokens']['mean']:.1f} |"
            )
    return "\n".join(lines) + "\n"


def run_benchmarks(
    *,
    extraction_backend: Literal["regex"] = "regex",
    fixtures_dir: Path | str = DEFAULT_FIXTURES_DIR,
    embedding_model: Any | None = None,
    dedup_threshold: float | None = None,
    systems: list[str] | None = None,
) -> BenchmarkReport:
    fixtures = load_benchmark_fixtures(fixtures_dir)
    model_instance = embedding_model or EmbeddingModel()
    selected_systems = systems or ["waggle", "rag_naive", "rag_tuned"]
    report = BenchmarkReport(
        fixtures={
            "directory": fixtures["base_dir"],
            "extraction_cases": len(fixtures["extraction_cases"]),
            "retrieval_nodes": len(fixtures["retrieval_cases"]["nodes"]),
            "retrieval_queries": len(fixtures["retrieval_cases"]["queries"]),
            "dedup_cases": len(fixtures["dedup_cases"]),
            "comparative_scenarios": len(fixtures["comparative_eval"]["scenarios"]),
            "comparative_queries": len(fixtures["comparative_eval"]["queries"]),
            "query_stress_cases": len(fixtures["query_stress_cases"]["queries"]),
        },
        metrics=[],
    )

    report.metrics.append(
        run_extraction_benchmark(fixtures["extraction_cases"], backend="regex")
    )

    embedding_ready = True

    try:
        report.metrics.append(
            run_retrieval_benchmark(fixtures["retrieval_cases"], embedding_model=model_instance)
        )
    except BenchmarkRuntimeError as exc:
        report.errors.append(str(exc))
        embedding_ready = False

    if embedding_ready:
        try:
            if dedup_threshold is None:
                dedup_result, sweep = choose_best_dedup_threshold(
                    fixtures["dedup_cases"],
                    embedding_model=model_instance,
                )
                report.threshold_sweep.extend(sweep)
            else:
                dedup_result = run_dedup_benchmark(
                    fixtures["dedup_cases"],
                    embedding_model=model_instance,
                    dedup_threshold=dedup_threshold,
                )
            report.metrics.append(dedup_result)
        except BenchmarkRuntimeError as exc:
            report.errors.append(str(exc))

        try:
            report.comparative = run_comparative_evaluation(
                fixtures["comparative_eval"],
                embedding_model=model_instance,
                systems=selected_systems,
            )
        except BenchmarkRuntimeError as exc:
            report.errors.append(str(exc))
        try:
            report.stress_eval = run_query_stress_evaluation(
                fixtures["comparative_eval"],
                fixtures["query_stress_cases"],
                embedding_model=model_instance,
            )
        except BenchmarkRuntimeError as exc:
            report.errors.append(str(exc))
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproducible local benchmark harness for waggle-mcp.")
    parser.add_argument(
        "--extraction-backend",
        choices=["regex"],
        default=os.getenv("WAGGLE_BENCHMARK_EXTRACTION_BACKEND", "regex"),
        help="Which extraction benchmark(s) to run.",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=["waggle", "rag_naive", "rag_tuned", "all"],
        default=["all"],
        help="Comparative systems to run. 'all' expands to waggle, rag_naive, rag_tuned.",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=DEFAULT_FIXTURES_DIR,
        help="Directory containing checked-in benchmark fixture JSON files.",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=None,
        help="Optional fixed dedup threshold. If omitted, the harness sweeps checked-in thresholds and picks the best score.",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("WAGGLE_MODEL", "all-MiniLM-L6-v2"),
        help="Embedding model shared by Waggle retrieval and the eval baselines.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. When provided, a Markdown summary is also written beside it.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    selected_systems = ["waggle", "rag_naive", "rag_tuned"] if "all" in args.systems else args.systems
    report = run_benchmarks(
        extraction_backend=args.extraction_backend,
        fixtures_dir=args.fixtures_dir,
        dedup_threshold=args.dedup_threshold,
        embedding_model=EmbeddingModel(args.embedding_model),
        systems=selected_systems,
    )

    print("=" * 72)
    print("waggle-mcp benchmark harness")
    print("=" * 72)
    print(
        f"fixtures: extraction={report.fixtures['extraction_cases']} "
        f"retrieval_nodes={report.fixtures['retrieval_nodes']} "
        f"retrieval_queries={report.fixtures['retrieval_queries']} "
        f"dedup_cases={report.fixtures['dedup_cases']} "
        f"comparative_scenarios={report.fixtures['comparative_scenarios']} "
        f"comparative_queries={report.fixtures['comparative_queries']} "
        f"query_stress_cases={report.fixtures['query_stress_cases']}"
    )
    for metric in report.metrics:
        print(_format_metric(metric))

    if report.threshold_sweep:
        print("dedup threshold sweep:")
        for metric in report.threshold_sweep:
            print(f"  {_format_metric(metric)}")

    if report.comparative:
        print("comparative systems:")
        for system, metrics in report.comparative["systems"].items():
            print(
                f"  {system:<10} hit@k={metrics['hit_at_k']:.0%} "
                f"exact={metrics['exact_support']:.0%} "
                f"tokens(mean/median/p95)={metrics['context_tokens']['mean']:.1f}/"
                f"{metrics['context_tokens']['median']:.1f}/{metrics['context_tokens']['p95']:.1f}"
            )
    if report.stress_eval:
        print("query stress systems:")
        for system, metrics in report.stress_eval["systems"].items():
            print(
                f"  {system:<10} hit@k={metrics['hit_at_k']:.0%} "
                f"exact={metrics['exact_support']:.0%} "
                f"tokens(mean/median/p95)={metrics['context_tokens']['mean']:.1f}/"
                f"{metrics['context_tokens']['median']:.1f}/{metrics['context_tokens']['p95']:.1f}"
            )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        markdown_path = args.output.with_suffix(".md")
        markdown_path.write_text(build_markdown_summary(report), encoding="utf-8")
        print(f"wrote JSON report to {args.output}")
        print(f"wrote Markdown summary to {markdown_path}")

    if report.errors:
        for error in report.errors:
            print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
