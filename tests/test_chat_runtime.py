from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from waggle.chat_runtime import OrchestratedChatRuntime
from waggle.orchestrator import AsyncMemoryOrchestrator, MemoryScope


@dataclass
class FakeGraph:
    observed: list[dict[str, str]] = field(default_factory=list)
    queries: list[dict[str, object]] = field(default_factory=list)
    checkpoints: list[dict[str, object]] = field(default_factory=list)
    imports: list[dict[str, object]] = field(default_factory=list)
    restored: bool = False

    def observe_conversation(
        self,
        *,
        user_message: str,
        assistant_response: str,
        agent_id: str = "",
        project: str = "",
        session_id: str = "",
    ) -> dict[str, object]:
        self.observed.append(
            {
                "user_message": user_message,
                "assistant_response": assistant_response,
                "agent_id": agent_id,
                "project": project,
                "session_id": session_id,
            }
        )
        return {"ok": True}

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
    ) -> dict[str, object]:
        payload = {
            "query": query,
            "max_nodes": max_nodes,
            "max_depth": max_depth,
            "agent_id": agent_id,
            "project": project,
            "session_id": session_id,
            "retrieval_mode": retrieval_mode,
        }
        self.queries.append(payload)
        if self.restored:
            return {
                **payload,
                "nodes": [{"label": "Restored decision"}],
                "replay_hits": [],
                "fusion_hits": [],
                "hybrid_hits": [],
            }
        return payload

    def prime_context(
        self,
        *,
        project: str = "",
        agent_id: str = "",
        session_id: str = "",
        max_nodes: int = 25,
    ) -> dict[str, object]:
        if self.restored:
            return {
                "project": project,
                "agent_id": agent_id,
                "session_id": session_id,
                "max_nodes": max_nodes,
                "summary": "Restored context",
                "nodes": [{"label": "Restored decision"}],
            }
        return {
            "project": project,
            "agent_id": agent_id,
            "session_id": session_id,
            "max_nodes": max_nodes,
            "summary": "",
            "nodes": [],
        }

    def export_abhi(
        self,
        *,
        output_path: str | None = None,
        project: str = "",
        agent_id: str = "",
        session_id: str = "",
        scope: str = "all",
        include_embeddings: bool = True,
    ) -> object:
        payload = {
            "output_path": output_path or "checkpoint.abhi",
            "project": project,
            "agent_id": agent_id,
            "session_id": session_id,
            "scope": scope,
            "include_embeddings": include_embeddings,
        }
        self.checkpoints.append(payload)
        return type("ExportResult", (), {"output_path": payload["output_path"]})()

    def import_abhi(
        self,
        *,
        input_path: str | Path,
        merge_strategy: str = "skip-existing",
    ) -> object:
        payload = {
            "input_path": str(input_path),
            "merge_strategy": merge_strategy,
        }
        self.imports.append(payload)
        self.restored = True
        return type(
            "ImportResult",
            (),
            {
                "input_path": str(input_path),
                "nodes_created": 1,
                "nodes_updated": 0,
                "edges_created": 0,
                "edges_updated": 0,
            },
        )()


@dataclass
class FakeModel:
    calls: list[dict[str, object]] = field(default_factory=list)

    async def generate(
        self,
        *,
        user_message: str,
        context: dict[str, object] | None,
        scope: MemoryScope,
    ) -> str:
        self.calls.append(
            {
                "user_message": user_message,
                "context": context,
                "scope": scope,
            }
        )
        return "Stored and answered."


@pytest.mark.asyncio
async def test_runtime_automates_retrieve_then_ingest() -> None:
    graph = FakeGraph()
    model = FakeModel()
    orchestrator = AsyncMemoryOrchestrator(graph)
    runtime = OrchestratedChatRuntime(model=model, orchestrator=orchestrator)
    scope = MemoryScope(project="MCP", session_id="thread-42", agent_id="codex")

    await runtime.start()
    try:
        result = await runtime.handle_turn(
            user_message="What did we decide about MCP registry publishing?",
            scope=scope,
            turn_id="thread-42:msg-1",
        )
        await runtime.flush()
    finally:
        await runtime.stop()

    assert result.context is not None
    assert result.ingest_plan.should_ingest is True
    assert len(graph.queries) == 1
    assert len(graph.observed) == 1
    assert graph.queries[0]["project"] == "MCP"
    assert graph.observed[0]["session_id"] == "thread-42"
    assert model.calls[0]["context"] is not None


@pytest.mark.asyncio
async def test_runtime_can_skip_retrieval_but_still_ingest() -> None:
    graph = FakeGraph()
    model = FakeModel()
    orchestrator = AsyncMemoryOrchestrator(graph)
    runtime = OrchestratedChatRuntime(model=model, orchestrator=orchestrator)
    scope = MemoryScope(project="MCP", session_id="thread-43", agent_id="codex")

    await runtime.start()
    try:
        result = await runtime.handle_turn(
            user_message="Remember this decision.",
            scope=scope,
            turn_id="thread-43:msg-1",
            retrieve=False,
        )
        await runtime.flush()
    finally:
        await runtime.stop()

    assert result.context is None
    assert len(graph.queries) == 0
    assert len(graph.observed) == 1


@pytest.mark.asyncio
async def test_runtime_checkpoint_flushes_and_exports_session_scope(tmp_path: Path) -> None:
    graph = FakeGraph()
    model = FakeModel()
    orchestrator = AsyncMemoryOrchestrator(graph)
    runtime = OrchestratedChatRuntime(model=model, orchestrator=orchestrator)
    scope = MemoryScope(project="MCP", session_id="thread-44", agent_id="codex")

    await runtime.start()
    try:
        await runtime.handle_turn(
            user_message="We decided to checkpoint context before app switching.",
            scope=scope,
            turn_id="thread-44:msg-1",
            retrieve=False,
        )
        checkpoint = await runtime.checkpoint_current_context(
            scope=scope,
            output_path=str(tmp_path / "thread-44.abhi"),
        )
    finally:
        await runtime.stop()

    assert len(graph.observed) == 1
    assert len(graph.checkpoints) == 1
    assert graph.checkpoints[0]["scope"] == "session"
    assert graph.checkpoints[0]["session_id"] == "thread-44"
    assert checkpoint.checkpoint_scope == "session"
    assert checkpoint.output_path.endswith("thread-44.abhi")


@pytest.mark.asyncio
async def test_runtime_resume_context_imports_checkpoint_after_empty_db_lookup(tmp_path: Path) -> None:
    graph = FakeGraph()
    model = FakeModel()
    orchestrator = AsyncMemoryOrchestrator(graph)
    runtime = OrchestratedChatRuntime(model=model, orchestrator=orchestrator)
    scope = MemoryScope(project="MCP", session_id="thread-45", agent_id="codex")
    checkpoint_path = tmp_path / "thread-45.abhi"
    checkpoint_path.write_text("checkpoint")

    resume = await runtime.resume_context(
        scope=scope,
        user_message="What did we decide about the handoff flow?",
        checkpoint_path=str(checkpoint_path),
    )

    assert resume.resumed_from_checkpoint is True
    assert resume.checkpoint_path.endswith("thread-45.abhi")
    assert len(graph.imports) == 1
    assert graph.imports[0]["merge_strategy"] == "skip-existing"
    assert len(graph.queries) == 2
