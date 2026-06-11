from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(slots=True)
class RuntimeContext:
    request_id: str = ""
    tenant_id: str = ""
    transport: str = ""
    backend: str = ""
    api_key_id: str = ""
    tool_name: str = ""


_runtime_context: contextvars.ContextVar[RuntimeContext | None] = contextvars.ContextVar(
    "waggle_runtime_context",
    default=None,
)


def get_runtime_context() -> RuntimeContext:
    return _runtime_context.get() or RuntimeContext()


@contextmanager
def runtime_context(**updates: str):
    current = get_runtime_context()
    next_context = RuntimeContext(
        request_id=updates.get("request_id", current.request_id),
        tenant_id=updates.get("tenant_id", current.tenant_id),
        transport=updates.get("transport", current.transport),
        backend=updates.get("backend", current.backend),
        api_key_id=updates.get("api_key_id", current.api_key_id),
        tool_name=updates.get("tool_name", current.tool_name),
    )
    token = _runtime_context.set(next_context)
    try:
        yield next_context
    finally:
        _runtime_context.reset(token)
