from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

from waggle.runtime_context import get_runtime_context


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        context = get_runtime_context()
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "tenant_id": context.tenant_id,
            "request_id": context.request_id,
            "transport": context.transport,
            "backend": context.backend,
            "api_key_id": context.api_key_id,
            "tool_name": context.tool_name,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True)


def configure_logging(level: str = "INFO", *, stream: object | None = None) -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    handler.setFormatter(JsonLogFormatter())
    root.handlers = [handler]
