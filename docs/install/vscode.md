# VS Code

Use this when you want Waggle enabled from a VS Code extension with a one-click setup flow for the current workspace.

Waggle is local graph memory for coding agents.

No cloud account. No API key. Local by default.

## One-line install

Install the Marketplace extension, then run:

```bash
Waggle: Enable for this Workspace
```

## Manual config

VS Code MCP examples commonly use `.vscode/mcp.json` with a `servers` root:

```json
{
  "servers": {
    "waggle": {
      "type": "stdio",
      "command": "waggle-mcp",
      "args": ["serve", "--transport", "stdio"],
      "env": {
        "WAGGLE_DEFAULT_TENANT_ID": "${workspaceFolderBasename}",
        "WAGGLE_DB_PATH": "~/.waggle/waggle.db"
      }
    }
  }
}
```

## Verify

```bash
waggle-mcp doctor
```

The extension will install `waggle-mcp` if needed, write `.vscode/mcp.json` after confirmation, and run `waggle-mcp doctor`.

Reload VS Code, switch the agent UI into MCP-capable mode, and confirm the Waggle server is enabled.

## Troubleshooting

See [troubleshooting.md](./troubleshooting.md).

## Security and privacy

Workspace MCP config is visible in the repo. That keeps the command path, tenant ID, and database path auditable during code review.
