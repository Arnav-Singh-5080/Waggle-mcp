# `waggle_plus` Private Package Scaffold

This folder is a public scaffold for the separate private `waggle_plus` package.

Use it to bootstrap a private repository, not to ship paid logic inside `Waggle Core`.

## Suggested bootstrap flow

1. Create a new private repository such as `waggle-plus`.
2. Copy this folder's contents into that repository, or run:

```bash
./scripts/bootstrap_waggle_plus.sh /path/to/private/waggle-plus
```

from the Core repo.
3. Replace the example provider implementation with your real OIDC, claims mapping, and RBAC logic.
4. Publish the private package to your own registry.
5. Install it alongside `waggle-mcp` in environments that should unlock paid features.

## Local smoke test

From your private repo:

```bash
pip install -e .
waggle-mcp plus --json
```

If the package is importable, `waggle-mcp` should detect it and expose the reserved Plus contract routes.

You can also run the scaffold contract test:

```bash
pip install pytest
python -m pytest tests/test_contract.py
```

## What to replace

The scaffold intentionally includes only placeholder behavior:

- static OIDC metadata
- deterministic fake callback session payload
- deterministic role mapping
- deterministic permission checks

That is enough to exercise the Core contract, but not enough for real identity or access control.
