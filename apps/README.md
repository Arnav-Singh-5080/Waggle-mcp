# App Surfaces

This repository groups product-facing surfaces under `apps/`:

- `apps/cli/`
  CLI-facing entrypoints and notes. The Python CLI currently shares implementation with the MCP server in `src/waggle/server.py`.
- `apps/mcp/`
  MCP-adjacent app surfaces, including Graph Studio and the Claude Desktop bundle.
- `apps/vscode-extension/`
  The VS Code extension package and release artifacts.

The Python package itself remains rooted at `src/` and `pyproject.toml` because packaging, import paths, and external registries still expect that layout.
