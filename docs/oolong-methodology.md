# OOLONG Methodology

This repo now includes an OOLONG-oriented runner for evaluating `LLM + Waggle` on long-context tasks.

## What it does

`src/waggle/oolong_benchmark.py` loads a local OOLONG JSON or JSONL export, indexes each `context_window_text` into Waggle chunk nodes, links adjacent chunks with graph edges, runs scoped retrieval for each question, and optionally asks an external LLM to answer from the retrieved Waggle context.

The goal is to measure the actual two-stage flow:

1. `Waggle retrieval` finds the relevant parts of a long context window.
2. `LLM answering` produces the final benchmark answer from that retrieved context.

## Supported dataset shapes

The loader supports local exports that look like:

- `Oolong-real`: rows containing fields such as `context_window_text`, `question`, `answer`, `question_type`, `context_window_id`
- `Oolong-synth`: rows containing fields such as `context_window_text` or `context_window_text_with_labels`, `question`, `answer`, `answer_type`, `task_group`

`answer` may be a plain string, JSON list, or Python-literal list string such as `['ham']`.

## Modes

- `retrieval_only`: runs Waggle retrieval and reports the retrieved bundle size per case, but does not score answer accuracy
- `waggle_llm`: runs retrieval, builds a benchmark prompt from the retrieved Waggle nodes, sends that prompt to an external command, and scores normalized exact match

## CLI

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_oolong.py /path/to/oolong.jsonl \
  --eval-mode retrieval_only
```

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_oolong.py /path/to/oolong.jsonl \
  --eval-mode waggle_llm \
  --llm-command "python my_llm_runner.py {prompt_file}" \
  --output benchmarks/oolong/results.json
```

## Notes

- `--llm-command` must print only the final answer to stdout.
- The command template receives `{prompt_file}` and `{prompt}` placeholders.
- Retrieval defaults to `graph` mode with chunk-to-chunk edges so Waggle can expand around a strong seed chunk.
- This runner is intentionally local-first. It does not download OOLONG automatically; provide a local export path.
