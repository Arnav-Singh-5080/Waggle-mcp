from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from waggle.models import AbhiImportResult, PrimeContextResult, SubgraphResult
from waggle.orchestrator import (
    AsyncMemoryOrchestrator,
    ConversationTurn,
    IngestPlan,
    MemoryScope,
    RetrieveRequest,
)


class ModelAdapter(Protocol):
    async def generate(
        self,
        *,
        user_message: str,
        context: SubgraphResult | None,
        scope: MemoryScope,
    ) -> str: ...


@dataclass(slots=True, frozen=True)
class RuntimeTurnResult:
    user_message: str
    assistant_response: str
    context: SubgraphResult | None
    ingest_plan: IngestPlan


@dataclass(slots=True, frozen=True)
class RuntimeCheckpointResult:
    output_path: str
    checkpoint_scope: str


@dataclass(slots=True, frozen=True)
class RuntimeResumeResult:
    context: SubgraphResult | None
    prime_context: PrimeContextResult | None
    resumed_from_checkpoint: bool
    checkpoint_path: str = ""
    import_result: AbhiImportResult | None = None


class OrchestratedChatRuntime:
    """
    Runtime integration that automates memory retrieval and ingestion per turn.

    - Before answer: query scoped memory context.
    - After answer: enqueue conversation observation for durable memory storage.
    """

    def __init__(
        self,
        *,
        model: ModelAdapter,
        orchestrator: AsyncMemoryOrchestrator,
        retrieval_mode: str = "graph",
        max_context_tokens: int = 1000,
        max_nodes: int = 12,
        max_depth: int = 2,
    ) -> None:
        self.model = model
        self.orchestrator = orchestrator
        self.retrieval_mode = retrieval_mode
        self.max_context_tokens = max_context_tokens
        self.max_nodes = max_nodes
        self.max_depth = max_depth

    async def start(self) -> None:
        await self.orchestrator.start()

    async def stop(self) -> None:
        await self.orchestrator.stop()

    async def flush(self) -> None:
        await self.orchestrator.flush()

    async def checkpoint_current_context(
        self,
        *,
        scope: MemoryScope,
        output_path: str | None = None,
        include_embeddings: bool = True,
    ) -> RuntimeCheckpointResult:
        await self.orchestrator.flush()

        exporter = getattr(self.orchestrator.graph, "export_abhi", None)
        if not callable(exporter):
            raise RuntimeError("Checkpoint export requires a graph backend that implements export_abhi().")

        checkpoint_scope = "session" if scope.session_id else "project" if scope.project else "all"
        export_result = await asyncio.to_thread(
            exporter,
            output_path=output_path,
            project=scope.project,
            agent_id=scope.agent_id,
            session_id=scope.session_id,
            scope=checkpoint_scope,
            include_embeddings=include_embeddings,
        )
        return RuntimeCheckpointResult(
            output_path=str(export_result.output_path),
            checkpoint_scope=checkpoint_scope,
        )

    async def resume_context(
        self,
        *,
        scope: MemoryScope,
        user_message: str = "",
        checkpoint_path: str | None = None,
        max_nodes: int | None = None,
    ) -> RuntimeResumeResult:
        context: SubgraphResult | None = None
        prime_context: PrimeContextResult | None = None

        if user_message.strip():
            context = await self.orchestrator.build_context(
                scope=scope,
                request=RetrieveRequest(
                    query=user_message.strip(),
                    retrieval_mode=self.retrieval_mode,
                    max_context_tokens=self.max_context_tokens,
                    max_nodes=max_nodes or self.max_nodes,
                    max_depth=self.max_depth,
                ),
            )
        else:
            prime_context = await self.orchestrator.build_prime(
                scope=scope,
                max_nodes=max_nodes or self.max_nodes,
            )

        if self._has_resume_context(context=context, prime_context=prime_context):
            return RuntimeResumeResult(
                context=context,
                prime_context=prime_context,
                resumed_from_checkpoint=False,
            )

        checkpoint = Path(checkpoint_path).expanduser() if checkpoint_path else None
        importer = getattr(self.orchestrator.graph, "import_abhi", None)
        if checkpoint is None or not checkpoint.exists() or not callable(importer):
            return RuntimeResumeResult(
                context=context,
                prime_context=prime_context,
                resumed_from_checkpoint=False,
                checkpoint_path=str(checkpoint) if checkpoint else "",
            )

        import_result = await asyncio.to_thread(
            importer,
            input_path=checkpoint,
            merge_strategy="skip-existing",
        )

        if user_message.strip():
            context = await self.orchestrator.build_context(
                scope=scope,
                request=RetrieveRequest(
                    query=user_message.strip(),
                    retrieval_mode=self.retrieval_mode,
                    max_context_tokens=self.max_context_tokens,
                    max_nodes=max_nodes or self.max_nodes,
                    max_depth=self.max_depth,
                ),
            )
        else:
            prime_context = await self.orchestrator.build_prime(
                scope=scope,
                max_nodes=max_nodes or self.max_nodes,
            )

        return RuntimeResumeResult(
            context=context,
            prime_context=prime_context,
            resumed_from_checkpoint=True,
            checkpoint_path=str(checkpoint),
            import_result=import_result,
        )

    async def handle_turn(
        self,
        *,
        user_message: str,
        scope: MemoryScope,
        turn_id: str = "",
        retrieve: bool = True,
    ) -> RuntimeTurnResult:
        message = user_message.strip()
        if not message:
            raise ValueError("user_message cannot be empty.")

        context: SubgraphResult | None = None
        if retrieve:
            context = await self.orchestrator.build_context(
                scope=scope,
                request=RetrieveRequest(
                    query=message,
                    retrieval_mode=self.retrieval_mode,
                    max_context_tokens=self.max_context_tokens,
                    max_nodes=self.max_nodes,
                    max_depth=self.max_depth,
                ),
            )

        assistant_response = (
            await self.model.generate(
                user_message=message,
                context=context,
                scope=scope,
            )
        ).strip()
        if not assistant_response:
            raise ValueError("Model returned an empty assistant response.")

        ingest_plan = await self.orchestrator.on_assistant_turn(
            scope=scope,
            turn=ConversationTurn(
                user_message=message,
                assistant_response=assistant_response,
                turn_id=turn_id,
            ),
        )
        return RuntimeTurnResult(
            user_message=message,
            assistant_response=assistant_response,
            context=context,
            ingest_plan=ingest_plan,
        )

    @staticmethod
    def _has_resume_context(
        *,
        context: SubgraphResult | None,
        prime_context: PrimeContextResult | None,
    ) -> bool:
        if context is not None and (
            getattr(context, "nodes", None)
            or getattr(context, "replay_hits", None)
            or getattr(context, "fusion_hits", None)
            or getattr(context, "hybrid_hits", None)
            or (
                isinstance(context, dict)
                and any(context.get(key) for key in ("nodes", "replay_hits", "fusion_hits", "hybrid_hits"))
            )
        ):
            return True
        return bool(
            prime_context is not None
            and (
                getattr(prime_context, "nodes", None)
                or (isinstance(prime_context, dict) and prime_context.get("nodes"))
            )
        )
