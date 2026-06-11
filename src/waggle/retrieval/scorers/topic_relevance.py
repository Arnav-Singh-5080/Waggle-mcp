"""Topic relevance scoring for retrieval candidates."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "at",
    "by",
    "for",
    "in",
    "is",
    "latest",
    "newest",
    "of",
    "oldest",
    "the",
    "to",
}


@dataclass(frozen=True)
class TopicScore:
    """Topic relevance score and its component scores."""

    semantic_similarity: float
    lexical_overlap: float
    combined: float


def tokenize(text: str) -> set[str]:
    """Return normalized content tokens, excluding small query stopwords."""
    return {token for token in TOKEN_RE.findall(text.lower()) if token and token not in STOPWORDS}


def lexical_overlap(query: str, document: str) -> float:
    """Compute bag-of-words overlap between a query and candidate text."""
    query_tokens = tokenize(query)
    document_tokens = tokenize(document)
    if not query_tokens or not document_tokens:
        return 0.0
    return len(query_tokens & document_tokens) / len(query_tokens)


def default_semantic_similarity(query: str, document: str) -> float:
    """Lightweight semantic proxy used when no embedding scorer is supplied."""
    query_tokens = tokenize(query)
    document_tokens = tokenize(document)
    if not query_tokens or not document_tokens:
        return 0.0
    intersection = len(query_tokens & document_tokens)
    denominator = math.sqrt(len(query_tokens) * len(document_tokens))
    return intersection / denominator if denominator else 0.0


def candidate_text(candidate: object, fields: Iterable[str] = ("text", "label", "content")) -> str:
    """Extract searchable text from a candidate object."""
    values: list[str] = []
    for field in fields:
        value = getattr(candidate, field, None)
        if isinstance(value, str) and value:
            values.append(value)
    tags = getattr(candidate, "tags", None)
    if tags:
        values.extend(str(tag) for tag in tags)
    return " ".join(values)


def score_topic_relevance(
    query: str,
    candidate: object,
    *,
    semantic_similarity_fn: Callable[[str, str], float] | None = None,
) -> TopicScore:
    """Score how strongly a candidate belongs to the query topic.

    The combined score is intentionally fixed to:
        0.7 * semantic_similarity + 0.3 * lexical_overlap
    """
    text = candidate_text(candidate)
    semantic_fn = semantic_similarity_fn or default_semantic_similarity
    semantic = max(0.0, min(1.0, float(semantic_fn(query, text))))
    lexical = max(0.0, min(1.0, lexical_overlap(query, text)))
    combined = (0.7 * semantic) + (0.3 * lexical)
    return TopicScore(
        semantic_similarity=semantic,
        lexical_overlap=lexical,
        combined=combined,
    )
