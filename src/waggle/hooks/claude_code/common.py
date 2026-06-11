from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from waggle.config import AppConfig


def resolve_scope(payload: dict[str, Any], config: AppConfig) -> dict[str, str]:
    explicit_project = str(payload.get("project", "") or "").strip()
    inferred_project = explicit_project or infer_project_scope(payload)
    return {
        "tenant_id": config.default_tenant_id,
        "project": inferred_project,
        "agent_id": str(payload.get("agent_id", "") or "").strip(),
        "session_id": str(payload.get("session_id", "") or "").strip(),
    }


def infer_project_scope(payload: dict[str, Any]) -> str:
    for key in ("repo_root", "repoRoot", "cwd", "workdir", "working_directory", "workspace_root", "path"):
        candidate = str(payload.get(key, "") or "").strip()
        if not candidate:
            continue
        resolved = Path(candidate).expanduser().resolve()
        root = _detect_repo_root(resolved)
        base = root if root is not None else resolved
        return _stable_project_id(base)
    return ""


def checkpoint_stem(*, config: AppConfig, project: str, session_id: str) -> Path:
    export_root = (
        Path(config.export_dir).expanduser() if config.export_dir else Path(config.db_path).expanduser().parent
    )
    checkpoint_root = export_root / "checkpoints"
    scope_parts = [project.strip() or "default-project", session_id.strip() or "default-session"]
    safe_parts = [_sanitize_path_component(part) for part in scope_parts]
    stem = checkpoint_root.joinpath(*safe_parts)
    stem.parent.mkdir(parents=True, exist_ok=True)
    return stem


def checkpoint_path(
    *,
    config: AppConfig,
    project: str,
    session_id: str,
    explicit_path: str = "",
) -> Path | None:
    if explicit_path.strip():
        return Path(explicit_path).expanduser()
    if not session_id.strip():
        return None
    return checkpoint_stem(config=config, project=project, session_id=session_id).with_suffix(".abhi")


def checkpoint_manifest_path(*, config: AppConfig) -> Path:
    export_root = (
        Path(config.export_dir).expanduser() if config.export_dir else Path(config.db_path).expanduser().parent
    )
    manifest_dir = export_root / "checkpoints"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    return manifest_dir / "manifest.json"


def write_checkpoint_manifest(
    *,
    config: AppConfig,
    project: str,
    agent_id: str,
    session_id: str,
    checkpoint_path: str,
) -> None:
    if not checkpoint_path.strip():
        return
    manifest_path = checkpoint_manifest_path(config=config)
    payload = {
        "project": project.strip(),
        "agent_id": agent_id.strip(),
        "session_id": session_id.strip(),
        "checkpoint_path": str(Path(checkpoint_path).expanduser()),
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_checkpoint_manifest(
    *,
    config: AppConfig,
    project: str,
    agent_id: str,
    session_id: str,
) -> Path | None:
    manifest_path = checkpoint_manifest_path(config=config)
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if str(payload.get("project", "") or "").strip() != project.strip():
        return None
    if str(payload.get("agent_id", "") or "").strip() != agent_id.strip():
        return None
    if str(payload.get("session_id", "") or "").strip() != session_id.strip():
        return None
    raw_path = str(payload.get("checkpoint_path", "") or "").strip()
    if not raw_path:
        return None
    resolved = Path(raw_path).expanduser()
    return resolved if resolved.exists() else None


def _sanitize_path_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _detect_repo_root(path: Path) -> Path | None:
    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _stable_project_id(path: Path) -> str:
    name = _sanitize_path_component(path.name or "workspace") or "workspace"
    import hashlib

    suffix = hashlib.sha1(path.as_posix().encode("utf-8")).hexdigest()[:8]
    return f"{name}@{suffix}"
