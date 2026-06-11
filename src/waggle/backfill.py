from __future__ import annotations

import logging
from collections import defaultdict

from waggle.graph import MemoryGraph
from waggle.models import BackfillStats, Node

LOGGER = logging.getLogger(__name__)


def backfill_context_windows(graph: MemoryGraph, *, dry_run: bool = False) -> BackfillStats:
    """Assign legacy unpartitioned nodes to context windows grouped by project/session."""
    stats = BackfillStats(dry_run=dry_run)
    unassigned_nodes = graph.get_nodes_without_window()
    stats.nodes_scanned = len(unassigned_nodes)

    if not unassigned_nodes:
        return stats

    groups: dict[tuple[str, str], list[Node]] = defaultdict(list)
    for node in unassigned_nodes:
        project = node.project.strip() or "default"
        session_id = node.session_id.strip() or "legacy"
        groups[(project, session_id)].append(node)

    if dry_run:
        return stats

    processed_windows: list[tuple[str, str]] = []
    existing_repos = {repo["id"] for repo in graph.list_repos()}
    existing_windows = {window.id for window in graph.list_context_windows(limit=10_000)}

    for (project, session_id), nodes in groups.items():
        try:
            repo_id = graph.ensure_repo(project)
            if repo_id not in existing_repos:
                stats.repos_created += 1
                existing_repos.add(repo_id)

            window_id = graph.ensure_context_window(session_id, repo_id)
            if window_id not in existing_windows:
                stats.windows_created += 1
                existing_windows.add(window_id)

            assigned = graph.assign_nodes_to_window([node.id for node in nodes], window_id)
            stats.nodes_assigned += assigned
            stats.nodes_skipped_already_assigned += max(len(nodes) - assigned, 0)
            graph.update_window_node_count(window_id)
            embedding = graph.get_window_embedding(window_id)
            if embedding is not None:
                stats.embeddings_computed += 1
            processed_windows.append((repo_id, window_id))
        except Exception as exc:  # pragma: no cover - defensive accounting
            message = f"Error backfilling project={project!r} session_id={session_id!r}: {exc}"
            LOGGER.exception(message)
            stats.errors.append(message)

    for repo_id, window_id in processed_windows:
        try:
            edges = graph.derive_context_window_edges(window_id, repo_id)
            stats.window_edges_created += len(edges)
        except Exception as exc:  # pragma: no cover - defensive accounting
            message = f"Error deriving context window edges for {window_id!r}: {exc}"
            LOGGER.exception(message)
            stats.errors.append(message)

    return stats
