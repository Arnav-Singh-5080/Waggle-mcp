# CLI Surface

The Waggle CLI currently ships from the main Python package:

- entrypoint metadata: `pyproject.toml`
- implementation: `src/waggle/server.py`
- packaging tests: `tests/test_packaging_metadata.py`

This folder is the canonical place for future CLI-only assets if the command surface is split further from the MCP server.
