from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from waggle.errors import ValidationFailure

DEFAULT_DB_PATH = "~/.waggle/memory.db"

# Valid values for WAGGLE_STARTUP_MODE
STARTUP_MODE_FAST = "fast"      # skip ML warmup; schema/inspection only
STARTUP_MODE_NORMAL = "normal"  # background warmup (default)
STARTUP_MODE_STRICT = "strict"  # block until embeddings ready before serving


@dataclass(slots=True)
class AppConfig:
    backend: str
    transport: str
    model_name: str
    db_path: str
    default_tenant_id: str
    http_host: str
    http_port: int
    log_level: str
    rate_limit_rpm: int
    write_rate_limit_rpm: int
    max_concurrent_requests: int
    max_payload_bytes: int
    request_timeout_seconds: int
    export_dir: str | None
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str
    recency_half_life_days: float = 30.0
    startup_mode: str = STARTUP_MODE_NORMAL  # fast | normal | strict

    @classmethod
    def from_env(cls) -> "AppConfig":
        config = cls(
            backend=os.environ.get("WAGGLE_BACKEND", "sqlite").strip().lower(),
            transport=os.environ.get("WAGGLE_TRANSPORT", "stdio").strip().lower(),
            model_name=os.environ.get("WAGGLE_MODEL", "all-MiniLM-L6-v2"),
            db_path=os.environ.get("WAGGLE_DB_PATH", DEFAULT_DB_PATH),
            default_tenant_id=os.environ.get("WAGGLE_DEFAULT_TENANT_ID", "local-default").strip(),
            http_host=os.environ.get("WAGGLE_HTTP_HOST", "0.0.0.0"),
            http_port=int(os.environ.get("WAGGLE_HTTP_PORT", "8080")),
            log_level=os.environ.get("WAGGLE_LOG_LEVEL", "INFO"),
            rate_limit_rpm=int(os.environ.get("WAGGLE_RATE_LIMIT_RPM", "120")),
            write_rate_limit_rpm=int(os.environ.get("WAGGLE_WRITE_RATE_LIMIT_RPM", "60")),
            max_concurrent_requests=int(os.environ.get("WAGGLE_MAX_CONCURRENT_REQUESTS", "8")),
            max_payload_bytes=int(os.environ.get("WAGGLE_MAX_PAYLOAD_BYTES", str(1024 * 1024))),
            request_timeout_seconds=int(os.environ.get("WAGGLE_REQUEST_TIMEOUT_SECONDS", "30")),
            recency_half_life_days=float(os.environ.get("WAGGLE_RECENCY_HALF_LIFE_DAYS", "30.0")),
            export_dir=os.environ.get("WAGGLE_EXPORT_DIR"),
            neo4j_uri=os.environ.get("WAGGLE_NEO4J_URI", "").strip(),
            neo4j_username=os.environ.get("WAGGLE_NEO4J_USERNAME", "").strip(),
            neo4j_password=os.environ.get("WAGGLE_NEO4J_PASSWORD", ""),
            neo4j_database=os.environ.get("WAGGLE_NEO4J_DATABASE", "").strip(),
            startup_mode=os.environ.get("WAGGLE_STARTUP_MODE", STARTUP_MODE_NORMAL).strip().lower(),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.transport not in {"stdio", "http"}:
            raise ValidationFailure(f"Unsupported WAGGLE_TRANSPORT: {self.transport}")
        if self.backend not in {"sqlite", "neo4j"}:
            raise ValidationFailure(f"Unsupported WAGGLE_BACKEND: {self.backend}")
        if self.transport == "http" and self.backend != "neo4j":
            raise ValidationFailure("HTTP transport requires WAGGLE_BACKEND=neo4j.")
        if not self.default_tenant_id:
            raise ValidationFailure("WAGGLE_DEFAULT_TENANT_ID cannot be empty.")
        if self.backend == "sqlite":
            self.db_path = str(Path(self.db_path).expanduser())
        if self.backend == "neo4j" and (not self.neo4j_uri or not self.neo4j_username or not self.neo4j_password):
            raise ValidationFailure(
                "Neo4j backend requires WAGGLE_NEO4J_URI, WAGGLE_NEO4J_USERNAME, and WAGGLE_NEO4J_PASSWORD."
            )
        if self.startup_mode not in {STARTUP_MODE_FAST, STARTUP_MODE_NORMAL, STARTUP_MODE_STRICT}:
            raise ValidationFailure(
                f"Unsupported WAGGLE_STARTUP_MODE: {self.startup_mode!r}. "
                f"Valid values: fast, normal, strict."
            )
        if self.recency_half_life_days <= 0:
            raise ValidationFailure("WAGGLE_RECENCY_HALF_LIFE_DAYS must be greater than 0.")

    @property
    def is_fast_mode(self) -> bool:
        return self.startup_mode == STARTUP_MODE_FAST

    @property
    def is_strict_mode(self) -> bool:
        return self.startup_mode == STARTUP_MODE_STRICT
