#!/usr/bin/env bash
# load_test.sh — convenience wrapper around load_test.py
# Usage: ./scripts/load_test.sh [--light|--medium|--heavy]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

HOST="${WAGGLE_HOST:-http://localhost:8080}"
API_KEY="${WAGGLE_API_KEY:-}"
PRESET="${1:-medium}"

if [[ -z "$API_KEY" ]]; then
  echo "ERROR: set WAGGLE_API_KEY before running this script."
  exit 1
fi

case "$PRESET" in
  --light|light)
    CONCURRENCY=5; DURATION=30; WRITE_RATIO=0.2
    ;;
  --medium|medium)
    CONCURRENCY=20; DURATION=60; WRITE_RATIO=0.3
    ;;
  --heavy|heavy)
    CONCURRENCY=50; DURATION=120; WRITE_RATIO=0.4
    ;;
  *)
    echo "Unknown preset '$PRESET'. Use: light | medium | heavy"
    exit 1
    ;;
esac

echo "==> Load test preset: $PRESET (concurrency=$CONCURRENCY, duration=${DURATION}s, write_ratio=$WRITE_RATIO)"
echo "==> Target: $HOST"

"$PROJECT_ROOT/.venv/bin/python" "$SCRIPT_DIR/load_test.py" \
  --host "$HOST" \
  --api-key "$API_KEY" \
  --concurrency "$CONCURRENCY" \
  --duration "$DURATION" \
  --write-ratio "$WRITE_RATIO"
