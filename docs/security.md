# Security and Trust

Waggle is local-first by default.

It is designed to give coding agents persistent graph memory without turning that memory layer into a cloud service.

## Defaults

- No cloud account required
- No API key required for local use
- No telemetry by default
- Local SQLite storage by default
- Explicit environment variables for database path and tenant namespace

## What Waggle stores

Waggle may store:

- transcript text
- extracted nodes and edges
- context windows
- repo and session scope metadata
- exported `.abhi` artifacts
- optional Google Drive sync credentials if that feature is enabled

By default, local storage is controlled through `WAGGLE_DB_PATH`, which currently defaults to `~/.waggle/waggle.db`.

## Command and tool boundaries

Waggle’s MCP tools are for memory operations. They should not execute arbitrary shell commands unless a specific tool explicitly implements and documents that behavior.

Destructive graph operations should always be explicit. Examples include:

- `clear-session`
- `clear-project`
- `clear-all`
- import or merge commands that intentionally overwrite existing memory

## Reset, delete, and inspect

To inspect the local graph:

- use Graph Studio with `waggle-mcp graph-studio`
- run `waggle-mcp doctor`
- export a scoped bundle or `.abhi` file for review

To delete or reset local memory:

- remove the configured SQLite database file
- or run explicit Waggle clear commands for session, project, or all-memory deletion

To export or back up memory:

- use `waggle-mcp export`
- use `waggle-mcp export-context-bundle`
- use `waggle-mcp export-markdown-vault`

## Privacy posture

Local-first means the memory database stays on the local machine unless the user deliberately exports, syncs, or shares it.

If you enable Google Drive sync or share exported artifacts, review encryption and redaction options first.

## Threat model notes for MCP clients

The MCP client that launches Waggle controls:

- which workspace the agent can see
- whether project-scoped MCP configs are trusted
- what other tools are available in the same session

Users should review:

- `.vscode/mcp.json`
- `.mcp.json`
- `~/.codex/config.toml`
- client-specific MCP config files

before enabling shared or project-scoped servers.

## Related docs

- [Root security notes](../SECURITY.md)
- [Security model](./security/security-model.md)
- [Hardening checklist](./security/hardening-checklist.md)
