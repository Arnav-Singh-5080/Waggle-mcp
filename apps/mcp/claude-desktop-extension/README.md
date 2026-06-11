# Waggle Claude Desktop Extension

This package is the current Claude Desktop MCPB bundle for Waggle.

It packages the metadata, launcher wrappers, and user configuration needed for Claude Desktop installation while Waggle still ships primarily as a Python package and stdio MCP server.

## Current status

- Manifest implemented
- User config schema included for database path, tenant ID, and model
- Platform launch wrappers included
- MCPB packaging scripts included

## Current limitation

This bundle still expects `waggle-mcp` to already be installed on the host machine. It is not yet a fully self-contained `.mcpb` distribution because the Python runtime and local ML dependencies are not bundled in this pass.

## Layout

- `manifest.json`
- `server/run-waggle.sh`
- `server/run-waggle.cmd`
- `package.json`
- `icon.png`

## Build notes

Anthropic’s current Desktop Extensions packaging flow uses the `mcpb` toolchain:

```bash
cd apps/mcp/claude-desktop-extension
npm install
npm run pack
```

To validate locally without producing the final archive:

```bash
cd apps/mcp/claude-desktop-extension
npm install
npm run validate
```

## Claude availability

Waggle is already prepared for Claude in two ways:

- Claude Code via the MCP commands documented in [docs/install/claude-code.md](../../../docs/install/claude-code.md)
- Claude Desktop via this `.mcpb` bundle or the manual config in [docs/install/claude-desktop.md](../../../docs/install/claude-desktop.md)

For the official Claude Desktop extension directory, Anthropic currently asks developers to submit interest through its desktop extensions intake process.
