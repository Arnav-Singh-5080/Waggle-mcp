# Starter Issues

These are candidate issues that would make the repository easier to contribute to and easier to trust as an open source project. Each item is intentionally scoped so a contributor can pick it up without needing full product context.

## 1. Add a label sync script for repository labels

- Recommended labels: `good first issue`, `help wanted`, `documentation`, `bug`, `enhancement`, `testing`, `tooling`, `onboarding`, `performance`, `graph`, `retrieval`, `windows`, `release`, `security`, `needs-triage`, `blocked`
- Suggested files: `.github/labels.yml`, `scripts/`, `README.md`
- Acceptance criteria:
  - Provide a documented script or GitHub Actions workflow that syncs `.github/labels.yml` to the GitHub repository.
  - Make the sync safe to re-run without duplicating labels.
  - Document how maintainers should use it.
- Suggested labels: `good first issue`, `tooling`, `onboarding`

Status:
- Implemented by `scripts/sync_github_labels.py` and `.github/workflows/sync-labels.yml`
- Maintainers can run it locally with `python3 scripts/sync_github_labels.py --dry-run`
- The workflow also runs on pushes to `main` that change the label catalog or sync script

## 2. Add a `doctor --json` mode

- Problem: `waggle-mcp doctor` is useful for humans, but automation and issue templates benefit from structured output.
- Suggested files: `src/waggle/server.py`, `tests/test_server.py`, `docs/reference.md`
- Acceptance criteria:
  - Add a machine-readable JSON mode.
  - Preserve the current human-friendly output by default.
  - Cover the new mode with tests.
- Suggested labels: `good first issue`, `enhancement`, `tooling`

## 3. Improve Windows troubleshooting coverage

- Problem: the repo documents Windows UTF-8 constraints, but setup failures and path issues are still harder to diagnose than on macOS/Linux.
- Suggested files: `docs/install/troubleshooting.md`, `docs/install/README.md`, `.github/ISSUE_TEMPLATE/bug_report.yml`
- Acceptance criteria:
  - Add a Windows-specific troubleshooting section with common symptoms and fixes.
  - Include path examples and shell differences where relevant.
  - Link the new section from the main install docs.
- Suggested labels: `good first issue`, `documentation`, `windows`

## 4. Add a focused Neo4j parity test pass

- Problem: the SQLite path has broader day-to-day coverage than the Neo4j implementation.
- Suggested files: `src/waggle/neo4j_graph.py`, `tests/`
- Acceptance criteria:
  - Identify a small, high-value set of graph operations that should behave the same in both backends.
  - Add tests or fixtures that make backend drift visible.
  - Document any intentionally unsupported behavior.
- Suggested labels: `help wanted`, `testing`, `graph`

## 5. Add screenshots for Graph Studio and setup flows

- Problem: the repo explains the product well, but new users still have to imagine the UI and onboarding experience.
- Suggested files: `README.md`, `assets/`, `docs/install/`
- Acceptance criteria:
  - Capture stable screenshots or small annotated images for setup and Graph Studio.
  - Keep images lightweight and place them under versioned assets.
  - Update docs to reference the images cleanly.
- Suggested labels: `good first issue`, `documentation`, `onboarding`

## 6. Tighten issue triage docs for maintainers

- Problem: contributors can open issues, but maintainers do not yet have a documented triage loop for labels, reproduction, and follow-up.
- Suggested files: `CONTRIBUTING.md`, `.github/labels.yml`, `docs/good-first-issues.md`
- Acceptance criteria:
  - Add a short maintainer triage rubric.
  - Define when to use `good first issue` vs `help wanted`.
  - Define when an issue should be marked `blocked` or `needs-triage`.
- Suggested labels: `documentation`, `onboarding`

## Label usage guidance

- Use `good first issue` only for tasks with clear acceptance criteria, a small blast radius, and obvious files to change.
- Use `help wanted` for larger tasks that are still open to outside contribution but need more codebase context.
- Pair broad labels with domain labels. Example: `bug` + `graph`, or `documentation` + `windows`.
