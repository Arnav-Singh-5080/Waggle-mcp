# Waggle Hook Integration

Waggle supports automatic memory capture via client hooks, eliminating the need for prompt rules in most cases.

## Auto-capture matrix

| Client | Method | Auto-capture quality |
|--------|--------|----------------------|
| Claude Code | Hooks | Strong (deterministic) |
| Codex | AGENTS.md prompt rule | Moderate (prompt-driven) |
| Cursor | User Rules | Moderate |
| Antigravity | User Rules | Moderate |

Hooks are preferred where supported. They fire deterministically on IDE events, independent of whether the model follows prompt instructions.

## Claude Code hooks

Three hook scripts are installed under `src/waggle/hooks/claude_code/`:

| Script | Claude Code event | What it does |
|--------|-------------------|--------------|
| `pre_response.py` | `UserPromptSubmit` | Tries scoped DB recall first; if the scope is cold and a session checkpoint exists, imports the `.abhi` checkpoint and retries before responding |
| `post_response.py` | `Stop` | Applies Waggle's durable-ingest policy, then calls `observe_conversation` only for turns worth remembering |
| `pre_compact.py` | `PreCompact` | Calls `ingest_transcript_handoff` to preserve durable info before context compression and emit a session-scoped `.abhi` checkpoint under the local checkpoints directory |

### Installation

Hooks are installed automatically when you run:

```bash
waggle-mcp setup --yes
```

To skip hook installation:

```bash
waggle-mcp setup --yes --no-hooks
```

To remove hooks:

```bash
waggle-mcp uninstall-hooks
```

### How it works

Each hook script:
- Reads JSON from stdin per the Claude Code hook protocol
- Calls the local Waggle in-process API (no network required)
- Writes JSON to stdout per the protocol
- **Always exits 0** — a Waggle bug never blocks your session
- Has a **5-second timeout** — if exceeded, exits silently

### Security

`post_response.py` scans turn text for likely secrets (API keys, tokens, passwords) before any ingestion work. If secrets are detected, the turn is skipped silently.

The hook is also policy-gated:
- short acknowledgements are skipped
- low-value chatter is skipped
- durable turns are ingested
- hook execution still always exits `0`

### Manual verification

After running `waggle-mcp setup --yes`, check `~/.claude/settings.json` for a `hooks` block containing entries with `waggle` in the command path.

Have a 2-turn conversation in Claude Code, then in a fresh session ask about the previous turn — it should recall it without any prompt rule.
