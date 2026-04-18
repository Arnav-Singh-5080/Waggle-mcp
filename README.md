<p align="center">
  <img src="https://raw.githubusercontent.com/Abhigyan-Shekhar/graph-memory-mcp/main/assets/banner.png" alt="waggle-mcp" width="720"/>
</p>

<p align="center">
  <strong>Persistent, structured memory for AI agents — up to 4× fewer tokens than chunk-based retrieval.</strong><br/>
  Your LLM remembers facts, decisions, and context <em>across every conversation</em>, backed by a real knowledge graph.
</p>

<p align="center">
  <a href="https://pypi.org/project/waggle-mcp"><img src="https://img.shields.io/pypi/v/waggle-mcp?color=39d5cf&label=pypi" alt="PyPI"/></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/MCP-compatible-brightgreen" alt="MCP compatible"/>
  <img src="https://img.shields.io/badge/embeddings-local%2C%20no%20API%20key-orange" alt="Local embeddings"/>
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT"/>
</p>

<p align="center">
  <a href="https://glama.ai/mcp/servers/Abhigyan-Shekhar/Waggle-mcp"><img src="https://glama.ai/mcp/servers/Abhigyan-Shekhar/Waggle-mcp/badges/card.svg" alt="Waggle-mcp MCP server"/></a>
  <a href="https://glama.ai/mcp/servers/Abhigyan-Shekhar/Waggle-mcp"><img src="https://glama.ai/mcp/servers/Abhigyan-Shekhar/Waggle-mcp/badges/score.svg" alt="Waggle-mcp MCP server score"/></a>
</p>

---

## Why waggle-mcp?

`waggle-mcp` is a local-first memory layer for MCP-compatible AI clients, built on a persistent knowledge graph. It gives your AI a persistent knowledge graph it can read and write through any MCP-compatible client (Clause Desktop, Cursor, Codex, Antigravity, etc.).

| Stuffed context | Structured retrieval |
|-----------------|----------------------|
| Huge prompts every session | Compact subgraph retrieved at query time |
| Session-local memory | Persistent multi-session memory |
| Flat notes and chunks | Typed nodes and edges: decisions, reasons, contradictions |
| "What changed?" requires replaying logs | Temporal queries and diffs are first-class |

Waggle yields **up to ~4× fewer tokens** than naive chunked retrieval on factual queries. Graph-traversal queries spend more tokens to include necessary reasoning context such as updates, contradictions, and dependencies.

---

## Quick start

```bash
pip install waggle-mcp
waggle-mcp init
# Restart your MCP client. Done.
```

`init` detects your MCP client, writes its config, and creates the local database directory. Default mode is local SQLite with on-device embeddings. Antigravity and manual configuration details are in [docs/reference.md](./docs/reference.md).

---

## See it in action

**Session 1** — April 10
```text
User:  Let's use PostgreSQL. MySQL replication has been painful.
Agent: [calls observe_conversation()]
       → stores decision node: "Chose PostgreSQL over MySQL"
       → stores reason node:   "MySQL replication painful"
       → links them with a depends_on edge
```

**Session 2** — April 12 (fresh context window, no history)
```text
User:  What did we decide about the database?
Agent: [calls query_graph("database decision")]
       → retrieves the decision node + linked reason from April 10

       "You decided on PostgreSQL on April 10. The reason recorded was
        that MySQL replication had been painful."
```

**Session 3** — April 14
```text
User:  Actually, let's reconsider — the team is more familiar with MySQL.
Agent: [calls store_node() + store_edge(new_node → old_node, "contradicts")]
       → both positions are preserved, and the contradiction is explicit
```

---

## Key Features

- **Automatic Extraction**: `observe_conversation` ingests facts into the graph without manual schema work.
- **Portable Context**: `export_context_bundle` generates Markdown/JSON context packs for another AI.
- **Vault Round-trip**: `export_markdown_vault` / `import_markdown_vault` for Obsidian-style node editing.
- **Conflict Resolution**: `list_conflicts` / `resolve_conflict` to manage contradictions without losing history.
- **Deterministic Integrity**: Uses a stable deterministic parser and SHA-256 embeddings for reliable, local-first operation.

---

## Benchmarks & Verification

Waggle performance is verified against checked-in fixtures and automated regression tests.

| Area | Corpus | Result |
|------|--------|--------|
| Extraction | 25-case deterministic fixture | `100%` |
| Retrieval | 18-query retrieval fixture | `83% Hit@k` |
| Comparative graph eval | 27-scenario / 69-query corpus | `88% Hit@k`, `77% exact support`, `58.5` mean tokens |
| Query stress | 40 adversarial retrieval-only cases | `98% Hit@k`, `98% exact support` |
| Deduplication | 22 cases (semi-semantic) | `77% (17/22)`, zero false merges |
| Unit Tests | Infrastructure & Logic | `90 passing tests` |

- **Token efficiency**: Waggle averages `58.5` tokens per retrieval vs `150.9` for naive chunked RAG.
- **Retrieval split**: The flat slice (`factual_recall`, `temporal_*`) measures `85% / 85%`; the graph slice (`change`, `delta`, `synthesis`, paraphrase) measures `93% / 70%`.
- **Deduplication**: Zero false-positive merges across the threshold sweep. Accuracy limited by conservative similarity bounds.

Detailed benchmark artifacts and methodology are in [tests/artifacts/README.md](./tests/artifacts/README.md).

---

## Reference & Docs

Detailed reference material lives in external documentation:

- **[docs/reference.md](./docs/reference.md)**: Environment variables, admin commands, Docker setup, and full tool surface.
- **[deploy/kubernetes/README.md](./deploy/kubernetes/README.md)**: Production deployment.
- **[docs/runbooks/](./docs/runbooks/)**: Operations and troubleshooting.
- **[tests/artifacts/README.md](./tests/artifacts/README.md)**: Benchmark artifacts and traceability.

---

## License

MIT — see [LICENSE](./LICENSE).
