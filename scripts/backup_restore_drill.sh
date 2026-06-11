#!/usr/bin/env bash
# backup_restore_drill.sh
# End-to-end backup → restore drill against a running waggle-mcp HTTP server.
#
# Usage:
#   WAGGLE_HOST=http://localhost:8080 \
#   WAGGLE_API_KEY=<key> \
#   ./scripts/backup_restore_drill.sh
#
# Exit codes: 0 = PASS, 1 = FAIL
set -euo pipefail

HOST="${WAGGLE_HOST:-http://localhost:8080}"
API_KEY="${WAGGLE_API_KEY:-}"
TENANT="${WAGGLE_TENANT_ID:-workspace-default}"

if [[ -z "$API_KEY" ]]; then
  echo "ERROR: WAGGLE_API_KEY must be set."
  exit 1
fi

PASS=0; FAIL=0
TMPDIR_DRILL="$(mktemp -d)"
BACKUP_PATH="$TMPDIR_DRILL/drill_backup.json"

cleanup() { rm -rf "$TMPDIR_DRILL"; }
trap cleanup EXIT

mcp_call() {
  # $1 = tool name, $2 = JSON arguments string
  curl -sf -X POST "$HOST/mcp" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -d "{\"jsonrpc\":\"2.0\",\"id\":$RANDOM,\"method\":\"tools/call\",\"params\":{\"name\":\"$1\",\"arguments\":$2}}"
}

check() {
  local label="$1" result="$2"
  if [[ "$result" == "ok" ]]; then
    echo "  [PASS] $label"
    ((PASS++))
  else
    echo "  [FAIL] $label — $result"
    ((FAIL++))
  fi
}

echo "=============================="
echo " waggle-mcp Backup/Restore Drill"
echo " Host   : $HOST"
echo " Tenant : $TENANT"
echo "=============================="

# ── 1. Health check ────────────────────────────────────────────────────────
echo
echo "── Phase 1: Health check"
HTTP_STATUS=$(curl -so /dev/null -w "%{http_code}" "$HOST/health/ready")
[[ "$HTTP_STATUS" == "200" ]] && check "health/ready returns 200" ok \
  || check "health/ready returns 200" "got HTTP $HTTP_STATUS"

# ── 2. Seed data ───────────────────────────────────────────────────────────
echo
echo "── Phase 2: Seed test data"
for i in 1 2 3; do
  LABEL="drill-node-$i-$$"
  RESP=$(mcp_call "store_node" \
    "{\"label\":\"$LABEL\",\"content\":\"Drill test node $i for backup restore validation.\",\"node_type\":\"fact\"}")
  NODE_ID=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['structuredContent']['id'])" 2>/dev/null || echo "")
  [[ -n "$NODE_ID" ]] && check "store node $i (id=$NODE_ID)" ok \
    || check "store node $i" "no id in response"
done

# ── 3. Get node count before backup ───────────────────────────────────────
echo
echo "── Phase 3: Stats before backup"
STATS_BEFORE=$(mcp_call "get_stats" "{}")
NODES_BEFORE=$(echo "$STATS_BEFORE" | python3 -c "import sys,json; print(json.load(sys.stdin)['structuredContent']['total_nodes'])" 2>/dev/null || echo 0)
EDGES_BEFORE=$(echo "$STATS_BEFORE" | python3 -c "import sys,json; print(json.load(sys.stdin)['structuredContent']['total_edges'])" 2>/dev/null || echo 0)
echo "  Nodes before backup: $NODES_BEFORE"
echo "  Edges before backup: $EDGES_BEFORE"
[[ "$NODES_BEFORE" -ge 3 ]] && check "at least 3 nodes exist" ok \
  || check "at least 3 nodes exist" "only $NODES_BEFORE nodes found"

# ── 4. Export backup ──────────────────────────────────────────────────────
echo
echo "── Phase 4: Export backup"
EXPORT_RESP=$(mcp_call "export_graph_backup" "{\"output_path\":\"$BACKUP_PATH\"}")
EXPORTED_NODES=$(echo "$EXPORT_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['structuredContent']['node_count'])" 2>/dev/null || echo -1)
[[ -f "$BACKUP_PATH" ]] && check "backup file created" ok \
  || check "backup file created" "file not found at $BACKUP_PATH"
[[ "$EXPORTED_NODES" -ge 3 ]] && check "backup contains ≥3 nodes ($EXPORTED_NODES)" ok \
  || check "backup contains ≥3 nodes" "got $EXPORTED_NODES"

# Validate JSON structure
python3 -c "
import json, sys
with open('$BACKUP_PATH') as f: d = json.load(f)
assert 'schema_version' in d, 'missing schema_version'
assert 'nodes' in d, 'missing nodes key'
assert 'edges' in d, 'missing edges key'
assert 'tenant_id' in d, 'missing tenant_id'
" 2>/dev/null && check "backup JSON structure valid" ok \
  || check "backup JSON structure valid" "schema validation failed"

# ── 5. Import backup (merge) ──────────────────────────────────────────────
echo
echo "── Phase 5: Import backup"
IMPORT_RESP=$(mcp_call "import_graph_backup" "{\"input_path\":\"$BACKUP_PATH\"}")
NODES_CREATED=$(echo "$IMPORT_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['structuredContent']['nodes_created'])" 2>/dev/null || echo -1)
NODES_UPDATED=$(echo "$IMPORT_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['structuredContent']['nodes_updated'])" 2>/dev/null || echo -1)
echo "  nodes_created: $NODES_CREATED"
echo "  nodes_updated: $NODES_UPDATED"
# On re-import into same tenant, nodes deduplicate → nodes_created=0 or nodes_updated>0
TOTAL_IMPORTED=$((NODES_CREATED + NODES_UPDATED))
[[ "$TOTAL_IMPORTED" -ge 3 ]] && check "import processed ≥3 nodes" ok \
  || check "import processed ≥3 nodes" "total=$TOTAL_IMPORTED"

# ── 6. Post-import stats ──────────────────────────────────────────────────
echo
echo "── Phase 6: Stats after import"
STATS_AFTER=$(mcp_call "get_stats" "{}")
NODES_AFTER=$(echo "$STATS_AFTER" | python3 -c "import sys,json; print(json.load(sys.stdin)['structuredContent']['total_nodes'])" 2>/dev/null || echo 0)
echo "  Nodes after import: $NODES_AFTER"
[[ "$NODES_AFTER" -ge "$NODES_BEFORE" ]] && check "node count stable or increased" ok \
  || check "node count stable or increased" "before=$NODES_BEFORE after=$NODES_AFTER"

# ── Summary ───────────────────────────────────────────────────────────────
echo
echo "=============================="
echo " RESULTS: $PASS passed, $FAIL failed"
echo "=============================="
[[ "$FAIL" -eq 0 ]] && echo " STATUS: PASS ✓" || echo " STATUS: FAIL ✗"
echo "=============================="

[[ "$FAIL" -eq 0 ]]
