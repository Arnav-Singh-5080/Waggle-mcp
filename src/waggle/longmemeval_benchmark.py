from __future__ import annotations

import argparse
import json
from hashlib import sha256
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from waggle.benchmark_harness import BenchmarkRuntimeError
from waggle.embeddings import EmbeddingModel
from waggle.intelligence import infer_temporal_hints, lexical_overlap


@dataclass
class LongMemEvalCaseResult:
    query_id: str
    question: str
    correct_session_ids: list[str]
    retrieved_session_ids: list[str]
    hit_at_5: bool
    exact_at_5: bool


@dataclass(frozen=True)
class PreparedLongMemEvalSession:
    session_id: str
    label: str
    content: str
    updated_at: datetime


@dataclass
class PreparedLongMemEvalEntry:
    query_id: str
    question: str
    correct_session_ids: list[str]
    sessions: list[PreparedLongMemEvalSession]
    embedding_matrix: np.ndarray


@dataclass
class LongMemEvalReport:
    dataset_path: str
    mode: str
    case_count: int
    cache_status: str
    cache_path: str
    prepared_entry_count: int
    prepared_session_count: int
    cache_key: str
    r_at_5: float
    exact_at_5: float
    per_case: list[LongMemEvalCaseResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "mode": self.mode,
            "case_count": self.case_count,
            "cache_status": self.cache_status,
            "cache_path": self.cache_path,
            "prepared_entry_count": self.prepared_entry_count,
            "prepared_session_count": self.prepared_session_count,
            "cache_key": self.cache_key,
            "r_at_5": self.r_at_5,
            "exact_at_5": self.exact_at_5,
            "per_case": [asdict(case) for case in self.per_case],
        }


@dataclass
class PreparedLongMemEvalCache:
    prepared_entries: list[PreparedLongMemEvalEntry]
    question_embeddings: np.ndarray


def _dataset_sha256(dataset_path: str | Path) -> str:
    digest = sha256()
    with Path(dataset_path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_entries(path: str | Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("entries", "data", "questions"):
            value = raw.get(key)
            if isinstance(value, list):
                return value
    raise BenchmarkRuntimeError("Unsupported LongMemEval file shape. Expected a list or dict with entries/data/questions.")


def _extract_correct_session_ids(entry: dict[str, Any]) -> list[str]:
    for key in (
        "correct_session_ids",
        "answer_session_ids",
        "needle_session_ids",
        "ground_truth_session_ids",
        "support_session_ids",
    ):
        value = entry.get(key)
        if isinstance(value, list) and value:
            return [str(item) for item in value]
    for key in ("correct_session_id", "answer_session_id", "needle_session_id"):
        value = entry.get(key)
        if value:
            return [str(value)]
    raise BenchmarkRuntimeError("Could not find ground-truth session IDs in LongMemEval entry.")


def _normalize_timestamp(raw: str) -> str:
    text = str(raw).strip()
    if not text:
        return datetime.now(timezone.utc).isoformat()
    try:
        if "/" in text and " (" in text:
            parsed = datetime.strptime(text.split(" (", 1)[0], "%Y/%m/%d")
        else:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc).isoformat()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _session_text(session: list[dict[str, Any]], *, include_assistant: bool) -> str:
    lines: list[str] = []
    for turn in session:
        role = str(turn.get("role", "unknown")).strip()
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        if include_assistant or role == "user":
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _embed_texts(embedding_model: Any, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    if hasattr(embedding_model, "embed_batch"):
        return np.asarray(embedding_model.embed_batch(texts), dtype=np.float32)
    return np.asarray([embedding_model.embed(text) for text in texts], dtype=np.float32)


def _embed_texts_in_chunks(embedding_model: Any, texts: list[str], *, chunk_size: int = 256) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    chunks = [
        _embed_texts(embedding_model, texts[index : index + chunk_size])
        for index in range(0, len(texts), chunk_size)
    ]
    if len(chunks) == 1:
        return chunks[0]
    return np.vstack(chunks)


def _rank_candidates_heuristic(question: str, sessions: list[PreparedLongMemEvalSession], *, top_k: int) -> list[PreparedLongMemEvalSession]:
    temporal_hints = infer_temporal_hints(question)
    max_timestamp = max((session.updated_at.timestamp() for session in sessions), default=1.0)
    min_timestamp = min((session.updated_at.timestamp() for session in sessions), default=0.0)
    span = max(max_timestamp - min_timestamp, 1.0)
    scored: list[tuple[float, int, PreparedLongMemEvalSession]] = []
    for index, session in enumerate(sessions):
        base_score = 1.0 / (index + 1)
        lexical_score = lexical_overlap(question, session.label, session.content)
        temporal_score = 0.0
        if temporal_hints.recency_mode == "latest":
            temporal_score = (session.updated_at.timestamp() - min_timestamp) / span
        elif temporal_hints.recency_mode == "oldest":
            temporal_score = (max_timestamp - session.updated_at.timestamp()) / span
        score = (0.5 * base_score) + (0.35 * lexical_score) + (0.15 * temporal_score)
        scored.append((score, -index, session))
    return [item[2] for item in sorted(scored, key=lambda item: (-item[0], item[1]))[:top_k]]


def _prepare_entry_specs(entry: dict[str, Any], *, mode: str) -> tuple[str, str, list[str], list[PreparedLongMemEvalSession]]:
    sessions = entry["haystack_sessions"]
    session_ids = entry["haystack_session_ids"]
    dates = entry["haystack_dates"]
    include_assistant = mode == "graph_hybrid"
    prepared_sessions: list[PreparedLongMemEvalSession] = []
    for session, session_id, raw_date in zip(sessions, session_ids, dates, strict=True):
        content = _session_text(session, include_assistant=include_assistant)
        if not content.strip():
            continue
        prepared_sessions.append(
            PreparedLongMemEvalSession(
                session_id=str(session_id),
                label=f"LongMemEval Session {session_id}",
                content=content,
                updated_at=datetime.fromisoformat(_normalize_timestamp(str(raw_date))),
            )
        )
    return (
        str(entry.get("id", "entry")),
        str(entry["question"]),
        _extract_correct_session_ids(entry),
        prepared_sessions,
    )


def _prepare_entries(entries: list[dict[str, Any]], *, mode: str, embedding_model: Any) -> list[PreparedLongMemEvalEntry]:
    entry_specs = [_prepare_entry_specs(entry, mode=mode) for entry in entries]
    unique_texts: list[str] = []
    seen_texts: set[str] = set()
    for _, _, _, sessions in entry_specs:
        for session in sessions:
            if session.content not in seen_texts:
                seen_texts.add(session.content)
                unique_texts.append(session.content)
    embedding_cache: dict[str, np.ndarray] = {}
    if unique_texts:
        for text, embedding in zip(unique_texts, _embed_texts_in_chunks(embedding_model, unique_texts), strict=True):
            embedding_cache[text] = embedding
    prepared_entries: list[PreparedLongMemEvalEntry] = []
    for query_id, question, correct_session_ids, sessions in entry_specs:
        if sessions:
            embedding_matrix = np.asarray([embedding_cache[session.content] for session in sessions], dtype=np.float32)
        else:
            embedding_matrix = np.empty((0, 0), dtype=np.float32)
        prepared_entries.append(
            PreparedLongMemEvalEntry(
                query_id=query_id,
                question=question,
                correct_session_ids=correct_session_ids,
                sessions=sessions,
                embedding_matrix=embedding_matrix,
            )
        )
    return prepared_entries


def _cache_dir_for_dataset(dataset_path: str | Path, cache_dir: str | Path | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    return Path(dataset_path).resolve().parent / ".cache"


def _embedding_model_version(embedding_model: Any) -> str:
    return str(getattr(embedding_model, "model_version", "") or embedding_model.__class__.__name__)


def _cache_key(
    dataset_path: str | Path,
    *,
    mode: str,
    embedding_model: Any,
    limit: int | None,
    dataset_digest: str,
) -> str:
    model_name = getattr(embedding_model, "model_name", embedding_model.__class__.__name__)
    return sha256(
        json.dumps(
            {
                "dataset_sha256": dataset_digest,
                "mode": mode,
                "limit": limit if limit is not None else "full",
                "embedding_model": str(model_name),
                "embedding_model_version": _embedding_model_version(embedding_model),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]


def _cache_file_stem(
    dataset_path: str | Path,
    *,
    cache_key: str,
    cache_dir: str | Path | None,
) -> Path:
    return _cache_dir_for_dataset(dataset_path, cache_dir) / f"longmemeval-{cache_key}"


def _cache_file_paths(
    dataset_path: str | Path,
    *,
    cache_key: str,
    cache_dir: str | Path | None,
) -> tuple[Path, Path]:
    stem = _cache_file_stem(dataset_path, cache_key=cache_key, cache_dir=cache_dir)
    return stem.with_suffix(".json"), stem.with_suffix(".npz")


def _serialize_prepared_entries(prepared_entries: list[PreparedLongMemEvalEntry]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}
    payload: list[dict[str, Any]] = []
    for index, entry in enumerate(prepared_entries):
        embedding_key = f"entry_{index}_embedding_matrix"
        arrays[embedding_key] = np.asarray(entry.embedding_matrix, dtype=np.float32)
        payload.append(
            {
                "query_id": entry.query_id,
                "question": entry.question,
                "correct_session_ids": list(entry.correct_session_ids),
                "embedding_key": embedding_key,
                "sessions": [
                    {
                        "session_id": session.session_id,
                        "label": session.label,
                        "content": session.content,
                        "updated_at": session.updated_at.isoformat(),
                    }
                    for session in entry.sessions
                ],
            }
        )
    return payload, arrays


def _deserialize_prepared_entries(payload: list[dict[str, Any]], arrays: Any) -> list[PreparedLongMemEvalEntry]:
    prepared_entries: list[PreparedLongMemEvalEntry] = []
    for entry in payload:
        sessions = [
            PreparedLongMemEvalSession(
                session_id=str(session["session_id"]),
                label=str(session["label"]),
                content=str(session["content"]),
                updated_at=datetime.fromisoformat(str(session["updated_at"])),
            )
            for session in entry.get("sessions", [])
        ]
        prepared_entries.append(
            PreparedLongMemEvalEntry(
                query_id=str(entry["query_id"]),
                question=str(entry["question"]),
                correct_session_ids=[str(item) for item in entry.get("correct_session_ids", [])],
                sessions=sessions,
                embedding_matrix=np.asarray(arrays[str(entry["embedding_key"])], dtype=np.float32),
            )
        )
    return prepared_entries


def _load_prepared_cache(metadata_path: Path, arrays_path: Path) -> PreparedLongMemEvalCache | None:
    if not metadata_path.exists() or not arrays_path.exists():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(arrays_path, allow_pickle=False) as arrays:
        prepared_entries = _deserialize_prepared_entries(payload.get("prepared_entries", []), arrays)
        question_embeddings = np.asarray(arrays[str(payload["question_embeddings_key"])], dtype=np.float32)
    return PreparedLongMemEvalCache(
        prepared_entries=prepared_entries,
        question_embeddings=question_embeddings,
    )


def _save_prepared_cache(
    metadata_path: Path,
    arrays_path: Path,
    *,
    prepared_entries: list[PreparedLongMemEvalEntry],
    question_embeddings: np.ndarray,
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_payload, arrays = _serialize_prepared_entries(prepared_entries)
    question_embeddings_key = "question_embeddings"
    arrays[question_embeddings_key] = np.asarray(question_embeddings, dtype=np.float32)
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "waggle-longmemeval-cache",
                "prepared_entries": prepared_payload,
                "question_embeddings_key": question_embeddings_key,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    np.savez_compressed(arrays_path, **arrays)


def _vector_similarity_matrix(question_embedding: np.ndarray, entry: PreparedLongMemEvalEntry, embedding_model: Any) -> np.ndarray:
    if not entry.sessions or entry.embedding_matrix.size == 0:
        return np.empty(0, dtype=np.float32)
    question_vector = np.asarray(question_embedding, dtype=np.float32)
    if entry.embedding_matrix.ndim == 2 and entry.embedding_matrix.shape[1] == question_vector.shape[0]:
        return np.asarray(entry.embedding_matrix @ question_vector, dtype=np.float32)
    return np.asarray(
        [embedding_model.cosine_similarity(question_vector, session_embedding) for session_embedding in entry.embedding_matrix],
        dtype=np.float32,
    )


def _raw_candidate_order(question: str, entry: PreparedLongMemEvalEntry, question_embedding: np.ndarray, embedding_model: Any) -> list[PreparedLongMemEvalSession]:
    semantic_scores = _vector_similarity_matrix(question_embedding, entry, embedding_model)
    if semantic_scores.size == 0:
        return []
    lexical_scores = np.asarray(
        [lexical_overlap(question, session.label, session.content) for session in entry.sessions],
        dtype=np.float32,
    )
    temporal_hints = infer_temporal_hints(question)
    temporal_scores = np.zeros(len(entry.sessions), dtype=np.float32)
    if temporal_hints.recency_mode != "default":
        timestamps = np.asarray([session.updated_at.timestamp() for session in entry.sessions], dtype=np.float64)
        max_timestamp = float(np.max(timestamps))
        min_timestamp = float(np.min(timestamps))
        span = max(max_timestamp - min_timestamp, 1.0)
        if temporal_hints.recency_mode == "latest":
            temporal_scores = np.asarray((timestamps - min_timestamp) / span, dtype=np.float32)
        elif temporal_hints.recency_mode == "oldest":
            temporal_scores = np.asarray((max_timestamp - timestamps) / span, dtype=np.float32)
    combined_scores = (0.72 * semantic_scores) + (0.18 * lexical_scores) + (0.10 * temporal_scores)
    ranked_indices = np.argsort(-combined_scores, kind="stable")
    return [entry.sessions[index] for index in ranked_indices]


def evaluate_longmemeval(
    dataset_path: str | Path,
    *,
    embedding_model: Any | None = None,
    mode: Literal["graph_raw", "graph_hybrid"] = "graph_raw",
    limit: int | None = None,
    cache_dir: str | Path | None = None,
) -> LongMemEvalReport:
    entries = _load_entries(dataset_path)
    if limit is not None:
        entries = entries[:limit]
    model_instance = embedding_model or EmbeddingModel()
    dataset_digest = _dataset_sha256(dataset_path)
    cache_key = _cache_key(
        dataset_path,
        mode=mode,
        embedding_model=model_instance,
        limit=limit,
        dataset_digest=dataset_digest,
    )
    cache_metadata_path, cache_arrays_path = _cache_file_paths(
        dataset_path,
        cache_key=cache_key,
        cache_dir=cache_dir,
    )
    cached = _load_prepared_cache(cache_metadata_path, cache_arrays_path)
    if cached is not None:
        cache_status = "warm"
        prepared_entries = cached.prepared_entries
        question_embeddings = cached.question_embeddings
    else:
        cache_status = "cold"
        prepared_entries = _prepare_entries(entries, mode=mode, embedding_model=model_instance)
        question_embeddings = _embed_texts_in_chunks(
            model_instance,
            [prepared_entry.question for prepared_entry in prepared_entries],
        )
        _save_prepared_cache(
            cache_metadata_path,
            cache_arrays_path,
            prepared_entries=prepared_entries,
            question_embeddings=question_embeddings,
        )
    results: list[LongMemEvalCaseResult] = []
    for index, (entry, prepared_entry, question_embedding) in enumerate(
        zip(entries, prepared_entries, question_embeddings, strict=True),
        start=1,
    ):
        question = prepared_entry.question
        ranked_candidates = _raw_candidate_order(question, prepared_entry, question_embedding, model_instance)
        if mode == "graph_raw":
            ranked_sessions = ranked_candidates[:5]
        else:
            ranked_sessions = _rank_candidates_heuristic(question, ranked_candidates[:20], top_k=5)
        retrieved_session_ids = [session.session_id for session in ranked_sessions]
        gold_ids = prepared_entry.correct_session_ids
        retrieved_set = set(retrieved_session_ids[:5])
        gold_set = set(gold_ids)
        results.append(
            LongMemEvalCaseResult(
                query_id=prepared_entry.query_id or str(entry.get("id", f"entry_{index}")),
                question=question,
                correct_session_ids=gold_ids,
                retrieved_session_ids=retrieved_session_ids[:5],
                hit_at_5=bool(retrieved_set & gold_set),
                exact_at_5=gold_set.issubset(retrieved_set),
            )
        )
    case_count = len(results)
    prepared_session_count = sum(len(entry.sessions) for entry in prepared_entries)
    hit_rate = sum(1 if result.hit_at_5 else 0 for result in results) / case_count if case_count else 0.0
    exact_rate = sum(1 if result.exact_at_5 else 0 for result in results) / case_count if case_count else 0.0
    return LongMemEvalReport(
        dataset_path=str(dataset_path),
        mode=mode,
        case_count=case_count,
        cache_status=cache_status,
        cache_path=str(cache_metadata_path),
        prepared_entry_count=len(prepared_entries),
        prepared_session_count=prepared_session_count,
        cache_key=cache_key,
        r_at_5=hit_rate,
        exact_at_5=exact_rate,
        per_case=results,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exploratory LongMemEval adapter for Waggle.")
    parser.add_argument("dataset_path", type=Path, help="Path to longmemeval_s_cleaned.json or equivalent cleaned dataset.")
    parser.add_argument("--mode", choices=["graph_raw", "graph_hybrid"], default="graph_raw")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of entries to evaluate.")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional directory for prepared LongMemEval cache files (JSON metadata plus .npz embeddings).",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = evaluate_longmemeval(
        args.dataset_path,
        embedding_model=EmbeddingModel(args.embedding_model),
        mode=args.mode,
        limit=args.limit,
        cache_dir=args.cache_dir,
    )
    print("=" * 72)
    print("waggle LongMemEval exploratory benchmark")
    print("=" * 72)
    print(f"dataset: {report.dataset_path}")
    print(f"mode: {report.mode}")
    print(f"cases: {report.case_count}")
    if report.cache_status == "warm":
        print(f"cache: warm ({report.cache_path})")
    else:
        print(f"cache: cold (wrote {report.cache_path})")
    print(f"R@5: {report.r_at_5:.1%}")
    print(f"Exact@5: {report.exact_at_5:.1%}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"wrote JSON report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
