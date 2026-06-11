from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from waggle.retrieval.hybrid import HybridRetrievalConfig, HybridRetriever


__all__ = ["HybridRetrievalConfig", "HybridRetriever"]


def __getattr__(name: str) -> object:
    if name in __all__:
        from waggle.retrieval.hybrid import HybridRetrievalConfig, HybridRetriever

        exports = {
            "HybridRetrievalConfig": HybridRetrievalConfig,
            "HybridRetriever": HybridRetriever,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
