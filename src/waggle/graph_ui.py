from __future__ import annotations


def render_graph_editor_html(*, mode: str = "edit") -> str:
    read_only = mode.strip().lower() == "view"
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Waggle Graph Studio</title>
  <style>
    :root {
      --bg: #f4efe6;
      --panel: rgba(255,255,255,0.84);
      --panel-strong: rgba(255,255,255,0.94);
      --line: #d6c9b5;
      --text: #1e1b16;
      --muted: #6b6256;
      --accent: #0e7490;
      --accent-2: #d97706;
      --danger: #b42318;
      --node: #125b68;
      --node-text: #fffdf7;
      --edge: #8f7b63;
      --selected: #e11d48;
      --query: #16a34a;
      --group-fill: rgba(14,116,144,0.08);
      --group-stroke: rgba(14,116,144,0.38);
      --shadow: 0 18px 50px rgba(41, 30, 16, 0.12);
      --radius: 18px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(14,116,144,0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(217,119,6,0.13), transparent 24%),
        linear-gradient(180deg, #f7f2ea 0%, var(--bg) 100%);
      min-height: 100vh;
    }
    button, select, input, textarea {
      font: inherit;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: var(--panel-strong);
      color: var(--text);
    }
    button {
      cursor: pointer;
      padding: 9px 13px;
      background: linear-gradient(180deg, #fdf9f1 0%, #efe5d6 100%);
      transition: transform 140ms ease, box-shadow 140ms ease, opacity 140ms ease;
    }
    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: 0.45; cursor: default; transform: none; }
    button.primary { background: linear-gradient(180deg, #11839d 0%, #0e7490 100%); color: white; border-color: #0c6a83; }
    button.warn { background: linear-gradient(180deg, #f59e0b 0%, #d97706 100%); color: white; border-color: #c26704; }
    button.danger { background: linear-gradient(180deg, #d94b45 0%, #b42318 100%); color: white; border-color: #a11d15; }
    .shell {
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr) 380px;
      gap: 16px;
      padding: 18px;
      min-height: 100vh;
    }
    body.focus-graph .shell { grid-template-columns: 92px minmax(0, 1fr) 0; }
    body.focus-graph .right { display: none; }
    body.focus-graph .left { width: 92px; overflow: hidden; }
    body.focus-graph .left .section:not(.brand),
    body.focus-graph .left .stats,
    body.focus-graph .left .hint,
    body.focus-graph .left .collapsible { display: none; }
    body.focus-graph .left .brand p { display: none; }
    body.focus-graph .left .brand h1 {
      font-size: 1.1rem;
      writing-mode: vertical-rl;
      transform: rotate(180deg);
      margin: 0 auto;
    }
    body.focus-graph .left .toolbar { justify-content: center; }
    body.readonly-mode .section-card {
      opacity: 0.55;
      pointer-events: none;
    }
    .panel {
      background: var(--panel);
      backdrop-filter: blur(14px);
      border: 1px solid rgba(255,255,255,0.55);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .section { padding: 16px 18px; border-top: 1px solid rgba(107,98,86,0.12); }
    .section:first-child { border-top: 0; }
    .column { display: flex; flex-direction: column; min-height: 0; }
    .brand h1 { margin: 0; font-size: 1.55rem; letter-spacing: 0.02em; }
    .brand p { margin: 8px 0 0; color: var(--muted); line-height: 1.35; }
    .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 14px; }
    .stat {
      background: rgba(255,255,255,0.55);
      border: 1px solid rgba(107,98,86,0.12);
      border-radius: 14px;
      padding: 10px 12px;
    }
    .substats {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-top: 10px;
    }
    .substat {
      background: rgba(255,255,255,0.34);
      border: 1px solid rgba(107,98,86,0.1);
      border-radius: 12px;
      padding: 9px 10px;
    }
    .substat-label {
      font-size: 0.68rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .substat-value {
      font-size: 1rem;
      margin-top: 4px;
    }
    .stat-label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
    .stat-value { font-size: 1.3rem; margin-top: 4px; }
    .toolbar, .toolbar-tight { display: flex; flex-wrap: wrap; gap: 10px; }
    .toolbar-tight { gap: 8px; }
    .field { display: flex; flex-direction: column; gap: 6px; margin-bottom: 12px; }
    .field label { font-size: 0.82rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
    .field input, .field select, .field textarea { width: 100%; padding: 10px 12px; }
    .field textarea { min-height: 84px; resize: vertical; }
    .hint { font-size: 0.84rem; color: var(--muted); line-height: 1.4; }
    .list-panel { flex: 0 0 auto; }
    .collapsible {
      background: rgba(255,255,255,0.38);
      border: 1px solid rgba(107,98,86,0.12);
      border-radius: 14px;
      overflow: hidden;
    }
    .collapsible summary {
      list-style: none;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      cursor: pointer;
      padding: 12px 14px;
      color: var(--muted);
      font-size: 0.95rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      user-select: none;
    }
    .collapsible summary::-webkit-details-marker { display: none; }
    .collapsible summary::after {
      content: "▾";
      font-size: 0.95rem;
      color: var(--accent);
      transition: transform 140ms ease;
    }
    .collapsible:not([open]) summary::after { transform: rotate(-90deg); }
    .collapsible summary span:last-child {
      color: var(--accent);
      font-size: 0.78rem;
      letter-spacing: 0.04em;
    }
    .collapsible.section-card { margin-top: 0; }
    .collapsible.section-card .section-inner { padding: 0 18px 18px; }
    .scroll {
      overflow: auto;
      min-height: 0;
      max-height: 260px;
      padding: 0 14px 14px;
    }
    .list-item {
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid transparent;
      background: rgba(255,255,255,0.45);
      margin-bottom: 10px;
      cursor: pointer;
    }
    .list-item.active { border-color: rgba(225,29,72,0.35); background: rgba(255,240,244,0.9); }
    .list-item.query { border-color: rgba(22,163,74,0.35); background: rgba(239,252,241,0.92); }
    .list-item-title { font-weight: 600; }
    .list-item-meta { color: var(--muted); font-size: 0.84rem; margin-top: 4px; }
    .main {
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr) auto;
      min-height: 0;
    }
    .scopebar {
      padding: 14px 18px;
      display: grid;
      grid-template-columns: repeat(3, 1fr) auto;
      gap: 10px;
      align-items: end;
      border-bottom: 1px solid rgba(107,98,86,0.12);
    }
    .actionbar {
      padding: 12px 18px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      border-bottom: 1px solid rgba(107,98,86,0.12);
      align-items: center;
    }
    .actionbar .spacer { flex: 1; }
    .graph-wrap {
      position: relative;
      min-height: 0;
      background:
        linear-gradient(rgba(145,125,97,0.09) 1px, transparent 1px),
        linear-gradient(90deg, rgba(145,125,97,0.09) 1px, transparent 1px);
      background-size: 28px 28px;
    }
    #graph {
      width: 100%;
      height: 100%;
      min-height: 680px;
      display: block;
      user-select: none;
    }
    .legend {
      padding: 12px 18px;
      color: var(--muted);
      border-top: 1px solid rgba(107,98,86,0.12);
      font-size: 0.9rem;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 9px;
      border-radius: 999px;
      background: rgba(17,131,157,0.12);
      color: var(--accent);
      font-size: 0.78rem;
      margin-right: 8px;
      margin-bottom: 6px;
    }
    .pill.warn { background: rgba(217,119,6,0.12); color: var(--accent-2); }
    .pill.alert { background: rgba(180,35,24,0.12); color: var(--danger); }
    .right .section h2, .left .section h2 { margin: 0 0 10px; font-size: 0.95rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
    .readonly { opacity: 0.55; pointer-events: none; }
    .actionbar .toggle.active {
      background: linear-gradient(180deg, #11839d 0%, #0e7490 100%);
      color: white;
      border-color: #0c6a83;
    }
    .kv { display: grid; grid-template-columns: 120px 1fr; gap: 8px; font-size: 0.88rem; }
    .kv div:nth-child(odd) { color: var(--muted); }
    .codebox {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.82rem;
      background: rgba(255,255,255,0.62);
      border: 1px solid rgba(107,98,86,0.12);
      border-radius: 14px;
      padding: 10px 12px;
      white-space: pre-wrap;
      max-height: 220px;
      overflow: auto;
    }
    .status {
      position: fixed;
      right: 18px;
      bottom: 18px;
      max-width: 420px;
      padding: 12px 14px;
      border-radius: 14px;
      background: rgba(30,27,22,0.92);
      color: #fff;
      box-shadow: var(--shadow);
      opacity: 0;
      transform: translateY(12px);
      transition: opacity 140ms ease, transform 140ms ease;
      pointer-events: none;
    }
    .status.visible { opacity: 1; transform: translateY(0); }
    .selection-box { fill: rgba(14,116,144,0.12); stroke: rgba(14,116,144,0.5); stroke-dasharray: 6 6; }
    .query-summary { margin-top: 8px; }
    @media (max-width: 1400px) {
      .shell { grid-template-columns: 280px minmax(0, 1fr); }
      .right { grid-column: 1 / -1; }
    }
    @media (max-width: 900px) {
      .shell { grid-template-columns: 1fr; }
      .scopebar { grid-template-columns: 1fr; }
      #graph { min-height: 460px; }
    }
  </style>
</head>
<body class='""" + ("readonly-mode" if read_only else "") + """'>
  <div class="shell">
    <aside class="panel left column">
      <div class="section brand">
        <h1>Waggle Graph Studio</h1>
        <p>""" + ("Inspect live memory as a graph workspace without mutating it." if read_only else "Work directly on the live memory graph: move, connect, group, inspect, query, export, and edit.") + """</p>
        <div class="stats">
          <div class="stat"><div class="stat-label">Nodes</div><div class="stat-value" id="stat-nodes">0</div></div>
          <div class="stat"><div class="stat-label">Edges</div><div class="stat-value" id="stat-edges">0</div></div>
          <div class="stat"><div class="stat-label">Tenant</div><div class="stat-value" id="stat-tenant">-</div></div>
        </div>
        <div class="substats">
          <div class="substat"><div class="substat-label">Connected</div><div class="substat-value" id="stat-connected">0</div></div>
          <div class="substat"><div class="substat-label">Isolates</div><div class="substat-value" id="stat-isolates">0</div></div>
          <div class="substat"><div class="substat-label">Clusters</div><div class="substat-value" id="stat-clusters">0</div></div>
        </div>
        <div class="toolbar" style="margin-top:14px;">
          <button class="primary" id="refresh-btn">Refresh</button>
          <button id="export-json-btn">Export JSON</button>
          <button id="export-abhi-btn">Export ABHI</button>
        </div>
      </div>
      <div class="section">
        <h2>Search</h2>
        <div class="field"><input id="search-input" placeholder="Find labels, content, types, tags"></div>
        <div class="hint">""" + ("Read-only mode. Search, inspect, run saved queries, and export." if read_only else "Shift-click for multi-select. Drag empty canvas for box select. Shift-drag from a node to create an edge.") + """</div>
      </div>
      <div class="section list-panel">
        <details class="collapsible" id="node-panel">
          <summary><span>Nodes</span><span id="node-panel-count">0</span></summary>
          <div class="scroll" id="node-list"></div>
        </details>
      </div>
      <div class="section list-panel">
        <details class="collapsible" id="edge-panel">
          <summary><span>Edges</span><span id="edge-panel-count">0</span></summary>
          <div class="scroll" id="edge-list"></div>
        </details>
      </div>
    </aside>

    <main class="panel main">
      <div class="scopebar">
        <div class="field"><label>Project</label><input id="scope-project" placeholder="optional project scope"></div>
        <div class="field"><label>Agent</label><input id="scope-agent" placeholder="optional agent scope"></div>
        <div class="field"><label>Session</label><input id="scope-session" placeholder="optional session scope"></div>
        <button class="primary" id="apply-scope-btn">Apply Scope</button>
      </div>
      <div class="actionbar">
        <button id="undo-btn">Undo Layout</button>
        <button id="redo-btn">Redo Layout</button>
        <button id="fit-btn">Fit Layout</button>
        <button id="focus-mode-btn" class="toggle">Focus Mode</button>
        <button id="labels-btn" class="toggle">Labels Off</button>
        <button id="clear-selection-btn">Clear Selection</button>
        <button id="duplicate-btn">Duplicate Selected</button>
        <button class="danger" id="delete-selected-btn">Delete Selected</button>
        <div class="spacer"></div>
        <span class="pill" id="selection-pill">0 selected</span>
        <span class="pill warn" id="mode-pill">""" + ("View mode" if read_only else "Edit mode") + """</span>
      </div>
      <div class="graph-wrap">
        <svg id="graph" viewBox="0 0 1400 860" preserveAspectRatio="xMidYMid meet"></svg>
      </div>
      <div class="legend">
        <span class="pill">Green highlight = query match</span>
        <span class="pill">Red highlight = current selection</span>
        <span class="pill">Blue frames = groups</span>
      </div>
    </main>

    <aside class="panel right column">
      <details class="collapsible section-card">
        <summary><span>Create Node</span><span>Composer</span></summary>
        <div class="section-inner">
          <div class="field"><label>Label</label><input id="create-node-label"></div>
          <div class="field"><label>Type</label><select id="create-node-type"></select></div>
          <div class="field"><label>Content</label><textarea id="create-node-content"></textarea></div>
          <div class="field"><label>Tags</label><input id="create-node-tags" placeholder="comma,separated,tags"></div>
          <div class="toolbar-tight">
            <button class="primary" id="create-node-btn">Add Node</button>
          </div>
        </div>
      </details>
      <details class="collapsible section-card" open>
        <summary><span>Edit Selection</span><span>Inspector</span></summary>
        <div class="section-inner">
          <div class="hint" id="selection-hint">Select a node or edge to inspect it.</div>
          <div class="field"><label>Node ID</label><input id="edit-node-id" readonly></div>
          <div class="field"><label>Label</label><input id="edit-node-label"></div>
          <div class="field"><label>Content</label><textarea id="edit-node-content"></textarea></div>
          <div class="field"><label>Tags</label><input id="edit-node-tags"></div>
          <div class="toolbar-tight">
            <button id="update-node-btn">Save Node</button>
            <button class="danger" id="delete-node-btn">Delete Node</button>
          </div>
          <div class="field" style="margin-top:12px;"><label>Edge ID</label><input id="edit-edge-id" readonly></div>
          <div class="field"><label>Source</label><select id="edit-edge-source"></select></div>
          <div class="field"><label>Target</label><select id="edit-edge-target"></select></div>
          <div class="field"><label>Relationship</label><select id="edit-edge-relationship"></select></div>
          <div class="field"><label>Weight</label><input id="edit-edge-weight" type="number" step="0.1" min="0" max="1"></div>
          <div class="toolbar-tight">
            <button id="update-edge-btn">Save Edge</button>
            <button class="danger" id="delete-edge-btn">Delete Edge</button>
          </div>
        </div>
      </details>
      <details class="collapsible section-card">
        <summary><span>Edges And Groups</span><span>Structure</span></summary>
        <div class="section-inner">
          <div class="field"><label>Connect Relationship</label><select id="create-edge-relationship"></select></div>
          <div class="field"><label>Group Label</label><input id="group-label" placeholder="Decision cluster"></div>
          <div class="field"><label>Group Color</label><input id="group-color" value="#4A90D9"></div>
          <div class="toolbar-tight">
            <button class="primary" id="create-group-btn">Group Selection</button>
            <button id="toggle-group-btn">Collapse Selected Group</button>
            <button id="delete-group-btn">Delete Selected Group</button>
          </div>
        </div>
      </details>
      <details class="collapsible section-card">
        <summary><span>Saved Queries</span><span>Recall</span></summary>
        <div class="section-inner">
          <div class="toolbar-tight" id="query-buttons"></div>
          <div class="field" style="margin-top:12px;"><label>Custom Query</label><textarea id="custom-query" placeholder="FIND nodes WHERE type='decision'"></textarea></div>
          <div class="toolbar-tight">
            <button id="run-custom-query-btn">Run Query</button>
            <button id="clear-query-btn">Clear Query Highlight</button>
          </div>
          <div class="query-summary hint" id="query-summary">No query active.</div>
        </div>
      </details>
      <details class="collapsible section-card">
        <summary><span>ABHI Surface</span><span>Export</span></summary>
        <div class="section-inner">
          <div class="kv" id="abhi-overview"></div>
          <div class="hint" id="abhi-validation-hint" style="margin-top:10px;"></div>
          <div class="codebox" id="abhi-codebox"></div>
        </div>
      </details>
      <details class="collapsible section-card">
        <summary><span>Import</span><span>Ingest</span></summary>
        <div class="section-inner">
          <div class="field"><label>Format</label><select id="import-format"><option value="abhi">ABHI</option><option value="json">JSON backup</option></select></div>
          <div class="field"><label>File</label><input id="import-file" type="file"></div>
          <div class="toolbar-tight"><button class="warn" id="import-btn">Import File</button></div>
        </div>
      </details>
      <details class="collapsible section-card">
        <summary><span>Recent Activity</span><span>Timeline</span></summary>
        <div class="section-inner">
          <div class="toolbar-tight" style="margin-bottom:10px;">
            <button id="activity-24h-btn">24h</button>
            <button id="activity-7d-btn">7d</button>
            <button id="activity-30d-btn">30d</button>
          </div>
          <div class="codebox" id="diff-box"></div>
        </div>
      </details>
    </aside>
  </div>
  <div class="status" id="status"></div>
  <script>
    const READ_ONLY = """ + ("true" if read_only else "false") + """;
    const SVG_WIDTH = 1400;
    const SVG_HEIGHT = 860;
    const state = {
      tenantId: "",
      nodes: [],
      edges: [],
      ui: {positions: {}, zoom: 1, viewport: {center_x: 0, center_y: 0}, groups: [], collapsed_groups: [], selected_nodes: []},
      positions: {},
      selectedNodeIds: [],
      selectedEdgeId: "",
      selectedGroupId: "",
      search: "",
      scope: {project: "", agent_id: "", session_id: ""},
      draggingNodeIds: [],
      dragAnchor: null,
      selectionBox: null,
      connectFromId: "",
      connectPreview: null,
      queryMatchNodeIds: [],
      queryMatchEdgeIds: [],
      querySummary: "",
      abhi: null,
      diff: null,
      degreeMap: {},
      view: {focusMode: false, showLabels: false},
      historyPast: [],
      historyFuture: [],
      activitySince: "24h",
      saveLayoutTimer: 0,
    };

    const els = {
      graph: document.getElementById("graph"),
      nodeList: document.getElementById("node-list"),
      edgeList: document.getElementById("edge-list"),
      nodePanelCount: document.getElementById("node-panel-count"),
      edgePanelCount: document.getElementById("edge-panel-count"),
      status: document.getElementById("status"),
      statNodes: document.getElementById("stat-nodes"),
      statEdges: document.getElementById("stat-edges"),
      statTenant: document.getElementById("stat-tenant"),
      statConnected: document.getElementById("stat-connected"),
      statIsolates: document.getElementById("stat-isolates"),
      statClusters: document.getElementById("stat-clusters"),
      scopeProject: document.getElementById("scope-project"),
      scopeAgent: document.getElementById("scope-agent"),
      scopeSession: document.getElementById("scope-session"),
      searchInput: document.getElementById("search-input"),
      selectionPill: document.getElementById("selection-pill"),
      querySummary: document.getElementById("query-summary"),
      abhiOverview: document.getElementById("abhi-overview"),
      abhiHint: document.getElementById("abhi-validation-hint"),
      abhiCodebox: document.getElementById("abhi-codebox"),
      diffBox: document.getElementById("diff-box"),
      queryButtons: document.getElementById("query-buttons"),
      selectionHint: document.getElementById("selection-hint"),
      focusModeBtn: document.getElementById("focus-mode-btn"),
      labelsBtn: document.getElementById("labels-btn"),
    };

    const NODE_TYPES = ["fact", "entity", "concept", "preference", "decision", "question", "note"];
    const RELATION_TYPES = ["relates_to", "contradicts", "depends_on", "part_of", "updates", "derived_from", "similar_to"];

    function initSelect(id, values, includeBlank = false) {
      const select = document.getElementById(id);
      const options = values.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join("");
      select.innerHTML = (includeBlank ? `<option value="">Select</option>` : "") + options;
    }

    initSelect("create-node-type", NODE_TYPES);
    initSelect("create-edge-relationship", RELATION_TYPES);
    initSelect("edit-edge-relationship", RELATION_TYPES);

    function showStatus(message, isError = false) {
      els.status.textContent = message;
      els.status.style.background = isError ? "rgba(138, 23, 23, 0.94)" : "rgba(30, 27, 22, 0.92)";
      els.status.classList.add("visible");
      window.clearTimeout(showStatus._timer);
      showStatus._timer = window.setTimeout(() => els.status.classList.remove("visible"), 2600);
    }

    function scopeQuery() {
      const params = new URLSearchParams();
      if (state.scope.project) params.set("project", state.scope.project);
      if (state.scope.agent_id) params.set("agent_id", state.scope.agent_id);
      if (state.scope.session_id) params.set("session_id", state.scope.session_id);
      const query = params.toString();
      return query ? `?${query}` : "";
    }

    async function request(path, options = {}) {
      const response = await fetch(path, {
        headers: {"Content-Type": "application/json", ...(options.headers || {})},
        ...options,
      });
      if (!response.ok) {
        let message = `${response.status} ${response.statusText}`;
        try {
          const payload = await response.json();
          message = payload.message || payload.error || message;
        } catch (_) {}
        throw new Error(message);
      }
      const contentType = response.headers.get("content-type") || "";
      if (contentType.includes("application/json")) return response.json();
      return response.text();
    }

    function escapeHtml(value) {
      return String(value || "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function tagString(tags) {
      return Array.isArray(tags) ? tags.join(", ") : "";
    }

    function parseTags(value) {
      return String(value || "").split(",").map((item) => item.trim()).filter(Boolean);
    }

    function shortLabel(value, maxLength) {
      const text = String(value || "");
      return text.length > maxLength ? `${text.slice(0, maxLength - 1)}…` : text;
    }

    function deepClone(value) {
      return JSON.parse(JSON.stringify(value));
    }

    function degreeMapForCurrentGraph() {
      const degree = Object.fromEntries(state.nodes.map((node) => [node.id, 0]));
      for (const edge of state.edges) {
        degree[edge.source_id] = (degree[edge.source_id] || 0) + 1;
        degree[edge.target_id] = (degree[edge.target_id] || 0) + 1;
      }
      return degree;
    }

    function nodeColor(node) {
      const palette = {
        fact: "#125b68",
        entity: "#7c5cff",
        concept: "#0f766e",
        preference: "#d97706",
        decision: "#b42318",
        question: "#2563eb",
        note: "#4b5563",
      };
      return palette[node.node_type] || "var(--node)";
    }

    function syncViewButtons() {
      document.body.classList.toggle("focus-graph", state.view.focusMode);
      els.focusModeBtn.classList.toggle("active", state.view.focusMode);
      els.labelsBtn.classList.toggle("active", state.view.showLabels);
      els.labelsBtn.textContent = state.view.showLabels ? "Labels On" : "Labels Off";
    }

    function rememberUiState() {
      const snapshot = {
        positions: deepClone(state.positions),
        groups: deepClone(state.ui.groups || []),
        collapsed_groups: deepClone(state.ui.collapsed_groups || []),
        selected_nodes: [...state.selectedNodeIds],
      };
      const previous = state.historyPast[state.historyPast.length - 1];
      if (previous && JSON.stringify(previous) === JSON.stringify(snapshot)) return;
      state.historyPast.push(snapshot);
      if (state.historyPast.length > 40) state.historyPast.shift();
      state.historyFuture = [];
      syncUndoButtons();
    }

    function syncUndoButtons() {
      document.getElementById("undo-btn").disabled = state.historyPast.length === 0;
      document.getElementById("redo-btn").disabled = state.historyFuture.length === 0;
    }

    function restoreUiState(snapshot) {
      state.positions = deepClone(snapshot.positions || {});
      state.ui.groups = deepClone(snapshot.groups || []);
      state.ui.collapsed_groups = deepClone(snapshot.collapsed_groups || []);
      state.selectedNodeIds = [...(snapshot.selected_nodes || [])];
      renderAll();
      scheduleSaveLayout();
    }

    function undoUiState() {
      if (!state.historyPast.length) return;
      const current = {
        positions: deepClone(state.positions),
        groups: deepClone(state.ui.groups || []),
        collapsed_groups: deepClone(state.ui.collapsed_groups || []),
        selected_nodes: [...state.selectedNodeIds],
      };
      state.historyFuture.push(current);
      restoreUiState(state.historyPast.pop());
      syncUndoButtons();
    }

    function redoUiState() {
      if (!state.historyFuture.length) return;
      const current = {
        positions: deepClone(state.positions),
        groups: deepClone(state.ui.groups || []),
        collapsed_groups: deepClone(state.ui.collapsed_groups || []),
        selected_nodes: [...state.selectedNodeIds],
      };
      state.historyPast.push(current);
      restoreUiState(state.historyFuture.pop());
      syncUndoButtons();
    }

    function collapsedMembers() {
      const ids = new Set();
      const collapsed = new Set(state.ui.collapsed_groups || []);
      for (const group of state.ui.groups || []) {
        if (!collapsed.has(group.id)) continue;
        for (const member of group.members || []) ids.add(member);
      }
      return ids;
    }

    function filteredNodes() {
      const term = state.search.trim().toLowerCase();
      const hidden = collapsedMembers();
      return state.nodes.filter((node) => {
        if (hidden.has(node.id) && !state.selectedNodeIds.includes(node.id)) return false;
        if (!term) return true;
        const haystack = [node.label, node.content, node.node_type, tagString(node.tags)].join(" ").toLowerCase();
        return haystack.includes(term);
      });
    }

    function filteredEdges() {
      const visibleIds = new Set(filteredNodes().map((node) => node.id));
      return state.edges.filter((edge) => visibleIds.has(edge.source_id) && visibleIds.has(edge.target_id));
    }

    function ensureLayout(nodes) {
      if (nodes.length >= 80 && layoutLooksLikeRing(nodes)) {
        smartLayout(nodes);
        return;
      }
      const centerX = SVG_WIDTH / 2;
      const centerY = SVG_HEIGHT / 2;
      const radius = Math.min(SVG_WIDTH, SVG_HEIGHT) * 0.34;
      nodes.forEach((node, index) => {
        if (!state.positions[node.id]) {
          const angle = (Math.PI * 2 * index) / Math.max(nodes.length, 1);
          state.positions[node.id] = {
            x: centerX + Math.cos(angle) * radius,
            y: centerY + Math.sin(angle) * radius,
          };
        }
      });
    }

    function layoutLooksLikeRing(nodes) {
      const points = nodes.map((node) => state.positions[node.id]).filter(Boolean);
      if (points.length < Math.min(nodes.length, 40)) return false;
      const centerX = SVG_WIDTH / 2;
      const centerY = SVG_HEIGHT / 2;
      const radii = points.map((pos) => Math.hypot(pos.x - centerX, pos.y - centerY));
      const mean = radii.reduce((sum, value) => sum + value, 0) / radii.length;
      if (!mean) return false;
      const variance = radii.reduce((sum, value) => sum + ((value - mean) ** 2), 0) / radii.length;
      return Math.sqrt(variance) / mean < 0.16;
    }

    function connectedComponents(nodes, edges) {
      const nodeIds = nodes.map((node) => node.id);
      const nodeSet = new Set(nodeIds);
      const adjacency = new Map(nodeIds.map((id) => [id, new Set()]));
      for (const edge of edges) {
        if (!nodeSet.has(edge.source_id) || !nodeSet.has(edge.target_id)) continue;
        adjacency.get(edge.source_id).add(edge.target_id);
        adjacency.get(edge.target_id).add(edge.source_id);
      }
      const seen = new Set();
      const components = [];
      for (const id of nodeIds) {
        if (seen.has(id)) continue;
        const stack = [id];
        const component = [];
        seen.add(id);
        while (stack.length) {
          const current = stack.pop();
          component.push(current);
          for (const next of adjacency.get(current) || []) {
            if (seen.has(next)) continue;
            seen.add(next);
            stack.push(next);
          }
        }
        components.push(component);
      }
      return components.sort((left, right) => right.length - left.length);
    }

    function graphStats() {
      const nodes = filteredNodes();
      const edges = filteredEdges();
      const components = connectedComponents(nodes, edges);
      const connectedCount = components
        .filter((component) => component.length > 1)
        .reduce((sum, component) => sum + component.length, 0);
      return {
        connectedCount,
        isolateCount: Math.max(0, nodes.length - connectedCount),
        clusterCount: components.filter((component) => component.length > 1).length,
      };
    }

    function smartLayout(nodes = filteredNodes()) {
      const edges = filteredEdges();
      const components = connectedComponents(nodes, edges);
      const connected = components.filter((component) => component.length > 1);
      const isolates = components.filter((component) => component.length === 1).map((component) => component[0]);
      const gridColumns = Math.max(1, Math.ceil(Math.sqrt(Math.max(connected.length, 1))));
      const cellWidth = 280;
      const cellHeight = 220;

      connected.forEach((component, index) => {
        const col = index % gridColumns;
        const row = Math.floor(index / gridColumns);
        const centerX = 200 + col * cellWidth;
        const centerY = 160 + row * cellHeight;
        component.forEach((nodeId, componentIndex) => {
          const degree = state.degreeMap[nodeId] || 0;
          const ring = Math.floor(componentIndex / 10);
          const slot = componentIndex % 10;
          const angle = (Math.PI * 2 * slot) / 10;
          const radius = 24 + ring * 24 + Math.max(0, 16 - degree * 2);
          state.positions[nodeId] = {
            x: centerX + Math.cos(angle) * radius,
            y: centerY + Math.sin(angle) * radius,
          };
        });
      });

      const isolateColumns = Math.max(1, Math.floor((SVG_WIDTH - 80) / 36));
      const isolateStartY = Math.max(140 + Math.ceil(connected.length / gridColumns) * cellHeight, 140);
      isolates.forEach((nodeId, index) => {
        const col = index % isolateColumns;
        const row = Math.floor(index / isolateColumns);
        state.positions[nodeId] = {
          x: 60 + col * 34,
          y: isolateStartY + row * 32,
        };
      });
    }

    function nodeById(id) {
      return state.nodes.find((node) => node.id === id) || null;
    }

    function edgeById(id) {
      return state.edges.find((edge) => edge.id === id) || null;
    }

    function renderNodeSelectors() {
      const options = state.nodes.map((node) => `<option value="${escapeHtml(node.id)}">${escapeHtml(node.label)} (${escapeHtml(node.node_type)})</option>`).join("");
      ["edit-edge-source", "edit-edge-target"].forEach((id) => {
        document.getElementById(id).innerHTML = `<option value="">Select node</option>${options}`;
      });
    }

    function currentGroup() {
      return (state.ui.groups || []).find((group) => group.id === state.selectedGroupId) || null;
    }

    function groupBounds(group) {
      const members = (group.members || []).map((id) => state.positions[id]).filter(Boolean);
      if (!members.length) return null;
      const xs = members.map((item) => item.x);
      const ys = members.map((item) => item.y);
      return {
        x: Math.min(...xs) - 54,
        y: Math.min(...ys) - 64,
        width: Math.max(...xs) - Math.min(...xs) + 108,
        height: Math.max(...ys) - Math.min(...ys) + 108,
      };
    }

    function groupSvg(group) {
      const bounds = groupBounds(group);
      if (!bounds) return "";
      const collapsed = (state.ui.collapsed_groups || []).includes(group.id);
      const selected = state.selectedGroupId === group.id;
      const stroke = selected ? "var(--selected)" : "var(--group-stroke)";
      return `
        <g class="group" data-group-id="${escapeHtml(group.id)}">
          <rect x="${bounds.x}" y="${bounds.y}" width="${bounds.width}" height="${bounds.height}" rx="24"
            fill="${group.color ? `${group.color}18` : "var(--group-fill)"}"
            stroke="${stroke}" stroke-width="${selected ? 3 : 2}" stroke-dasharray="${collapsed ? "9 7" : "0"}"></rect>
          <text x="${bounds.x + 16}" y="${bounds.y + 24}" font-size="14" font-weight="600" fill="${stroke}">
            ${escapeHtml(group.label || "Group")}${collapsed ? " (collapsed)" : ""}
          </text>
        </g>`;
    }

    function lineForEdge(edge) {
      const source = state.positions[edge.source_id];
      const target = state.positions[edge.target_id];
      if (!source || !target) return "";
      const selected = state.selectedEdgeId === edge.id;
      const queryMatch = state.queryMatchEdgeIds.includes(edge.id);
      const stroke = selected ? "var(--selected)" : (queryMatch ? "var(--query)" : "var(--edge)");
      const showLabel = selected || queryMatch || filteredEdges().length <= 20;
      return `
        <g class="edge" data-edge-id="${escapeHtml(edge.id)}">
          <line x1="${source.x}" y1="${source.y}" x2="${target.x}" y2="${target.y}" stroke="${stroke}" stroke-width="${selected ? 4 : 1.7}" stroke-linecap="round" opacity="${selected || queryMatch ? "1" : "0.65"}"></line>
          ${showLabel ? `<text x="${(source.x + target.x) / 2}" y="${(source.y + target.y) / 2 - 10}" text-anchor="middle" font-size="11" fill="var(--muted)">${escapeHtml(edge.relationship)}</text>` : ""}
        </g>`;
    }

    function nodeForGraph(node) {
      const pos = state.positions[node.id];
      const selected = state.selectedNodeIds.includes(node.id);
      const queryMatch = state.queryMatchNodeIds.includes(node.id);
      const degree = state.degreeMap[node.id] || 0;
      const radius = selected ? 16 : Math.max(7, Math.min(13, 7 + degree * 1.3));
      const fill = selected ? "var(--selected)" : (queryMatch ? "var(--query)" : nodeColor(node));
      const showLabel = state.view.showLabels || selected || queryMatch || degree >= 3;
      return `
        <g class="node" data-node-id="${escapeHtml(node.id)}" transform="translate(${pos.x}, ${pos.y})" style="cursor:move;">
          <circle r="${radius}" fill="${fill}" stroke="rgba(255,255,255,0.9)" stroke-width="${selected ? 3 : 2}"></circle>
          ${showLabel ? `<text y="${radius + 16}" text-anchor="middle" fill="var(--text)" font-size="${selected ? "13" : "11"}" font-weight="${selected ? "600" : "500"}">${escapeHtml(shortLabel(node.label, 28))}</text>` : ""}
        </g>`;
    }

    function selectionBoxSvg() {
      if (!state.selectionBox) return "";
      const box = normalizeBox(state.selectionBox);
      return `<rect class="selection-box" x="${box.x}" y="${box.y}" width="${box.width}" height="${box.height}"></rect>`;
    }

    function connectPreviewSvg() {
      if (!state.connectPreview || !state.connectFromId) return "";
      const source = state.positions[state.connectFromId];
      if (!source) return "";
      return `<line x1="${source.x}" y1="${source.y}" x2="${state.connectPreview.x}" y2="${state.connectPreview.y}" stroke="var(--accent)" stroke-width="3" stroke-dasharray="8 6"></line>`;
    }

    function renderGraph() {
      const nodes = filteredNodes();
      const edges = filteredEdges();
      ensureLayout(nodes);
      els.graph.innerHTML = `
        <rect data-canvas-bg="1" x="0" y="0" width="${SVG_WIDTH}" height="${SVG_HEIGHT}" fill="transparent"></rect>
        ${(state.ui.groups || []).map(groupSvg).join("")}
        ${edges.map(lineForEdge).join("")}
        ${connectPreviewSvg()}
        ${nodes.map(nodeForGraph).join("")}
        ${selectionBoxSvg()}
      `;

      const canvas = els.graph.querySelector("[data-canvas-bg]");
      canvas.addEventListener("mousedown", startBoxSelect);
      els.graph.querySelectorAll("[data-node-id]").forEach((element) => {
        element.addEventListener("click", onNodeClick);
        element.addEventListener("mousedown", onNodeMouseDown);
      });
      els.graph.querySelectorAll("[data-edge-id]").forEach((element) => {
        element.addEventListener("click", () => selectEdge(element.dataset.edgeId));
      });
      els.graph.querySelectorAll("[data-group-id]").forEach((element) => {
        element.addEventListener("click", () => selectGroup(element.dataset.groupId));
      });
    }

    function renderNodeList() {
      const nodes = filteredNodes();
      els.nodePanelCount.textContent = String(nodes.length);
      els.nodeList.innerHTML = nodes.map((node) => `
        <div class="list-item ${state.selectedNodeIds.includes(node.id) ? "active" : ""} ${state.queryMatchNodeIds.includes(node.id) ? "query" : ""}" data-node-list-id="${escapeHtml(node.id)}">
          <div class="list-item-title">${escapeHtml(node.label)}</div>
          <div class="list-item-meta">${escapeHtml(node.node_type)} • ${escapeHtml(shortLabel(node.content, 72))}</div>
        </div>
      `).join("") || `<div class="hint">No nodes match the current scope or filters.</div>`;
      els.nodeList.querySelectorAll("[data-node-list-id]").forEach((element) => {
        element.addEventListener("click", (event) => {
          selectNode(element.dataset.nodeListId, event.shiftKey || event.metaKey || event.ctrlKey);
        });
      });
    }

    function renderEdgeList() {
      const edges = filteredEdges();
      els.edgePanelCount.textContent = String(edges.length);
      els.edgeList.innerHTML = edges.map((edge) => `
        <div class="list-item ${state.selectedEdgeId === edge.id ? "active" : ""} ${state.queryMatchEdgeIds.includes(edge.id) ? "query" : ""}" data-edge-list-id="${escapeHtml(edge.id)}">
          <div class="list-item-title">${escapeHtml(edge.relationship)}</div>
          <div class="list-item-meta">${escapeHtml(nodeById(edge.source_id)?.label || edge.source_id)} → ${escapeHtml(nodeById(edge.target_id)?.label || edge.target_id)}</div>
        </div>
      `).join("") || `<div class="hint">No edges match the current scope or filters.</div>`;
      els.edgeList.querySelectorAll("[data-edge-list-id]").forEach((element) => {
        element.addEventListener("click", () => selectEdge(element.dataset.edgeListId));
      });
    }

    function renderQueryButtons() {
      const saved = state.abhi?.queries?.saved || [];
      els.queryButtons.innerHTML = saved.map((item) => `<button data-query-id="${escapeHtml(item.id)}">${escapeHtml(item.name)}</button>`).join("") || `<span class="hint">No saved queries.</span>`;
      els.queryButtons.querySelectorAll("[data-query-id]").forEach((element) => {
        element.addEventListener("click", () => runSavedQuery(element.dataset.queryId).catch((error) => showStatus(error.message, true)));
      });
    }

    function renderAbhiPanel() {
      if (!state.abhi) {
        els.abhiOverview.innerHTML = "";
        els.abhiHint.textContent = "No ABHI metadata loaded yet.";
        els.abhiCodebox.textContent = "";
        return;
      }
      const validation = state.abhi.validation || {};
      const integrity = state.abhi.integrity || {};
      els.abhiOverview.innerHTML = [
        ["Hash", shortLabel(integrity.content_hash || "-", 24)],
        ["Nodes", String(integrity.node_count || 0)],
        ["Edges", String(integrity.edge_count || 0)],
        ["Versions", String((state.abhi.versions || []).length)],
        ["Queries", String((state.abhi.queries?.saved || []).length)],
        ["Events", String(Object.keys(state.abhi.events || {}).length)],
      ].map(([key, value]) => `<div>${escapeHtml(key)}</div><div>${escapeHtml(value)}</div>`).join("");
      els.abhiHint.textContent = validation.valid ? "Live graph currently satisfies ABHI validation." : `Validation errors: ${(validation.errors || []).join(" | ")}`;
      els.abhiCodebox.textContent = JSON.stringify({
        constraints: state.abhi.constraints || [],
        queries: state.abhi.queries || {},
        events: state.abhi.events || {},
        versions: state.abhi.versions || [],
      }, null, 2);
    }

    function renderDiffPanel() {
      if (!state.diff) {
        els.diffBox.textContent = "No activity loaded yet.";
        return;
      }
      const lines = [
        `Since: ${state.diff.since}`,
        `Added nodes: ${(state.diff.added_nodes || []).length}`,
        `Updated nodes: ${(state.diff.updated_nodes || []).length}`,
        `Created edges: ${(state.diff.created_edges || []).length}`,
        `Contradictions: ${(state.diff.contradiction_edges || []).length}`,
        "",
      ];
      for (const node of (state.diff.added_nodes || []).slice(0, 8)) {
        lines.push(`+ [${node.node_type}] ${node.label}`);
      }
      for (const edge of (state.diff.created_edges || []).slice(0, 8)) {
        lines.push(`> ${shortLabel(edge.source_id, 8)} --${edge.relationship}--> ${shortLabel(edge.target_id, 8)}`);
      }
      els.diffBox.textContent = lines.join("\\n");
    }

    function renderInspector() {
      const group = currentGroup();
      if (group) {
        els.selectionHint.textContent = `Group selected: ${group.label || group.id}`;
      } else if (state.selectedEdgeId) {
        const edge = edgeById(state.selectedEdgeId);
        els.selectionHint.textContent = edge ? `Edge selected: ${edge.relationship}` : "Select a node or edge to inspect it.";
      } else if (state.selectedNodeIds.length === 1) {
        const node = nodeById(state.selectedNodeIds[0]);
        els.selectionHint.textContent = node ? `${node.node_type} node selected` : "Select a node or edge to inspect it.";
      } else if (state.selectedNodeIds.length > 1) {
        els.selectionHint.textContent = `${state.selectedNodeIds.length} nodes selected`;
      } else {
        els.selectionHint.textContent = "Select a node or edge to inspect it.";
      }

      const node = state.selectedNodeIds.length === 1 ? nodeById(state.selectedNodeIds[0]) : null;
      document.getElementById("edit-node-id").value = node?.id || "";
      document.getElementById("edit-node-label").value = node?.label || "";
      document.getElementById("edit-node-content").value = node?.content || "";
      document.getElementById("edit-node-tags").value = node ? tagString(node.tags) : "";

      const edge = state.selectedEdgeId ? edgeById(state.selectedEdgeId) : null;
      document.getElementById("edit-edge-id").value = edge?.id || "";
      document.getElementById("edit-edge-source").value = edge?.source_id || "";
      document.getElementById("edit-edge-target").value = edge?.target_id || "";
      document.getElementById("edit-edge-relationship").value = edge?.relationship || RELATION_TYPES[0];
      document.getElementById("edit-edge-weight").value = edge ? String(edge.weight ?? 1) : "";

      els.selectionPill.textContent = `${state.selectedNodeIds.length + (state.selectedEdgeId ? 1 : 0)} selected`;
    }

    function renderAll() {
      const stats = graphStats();
      els.statConnected.textContent = String(stats.connectedCount);
      els.statIsolates.textContent = String(stats.isolateCount);
      els.statClusters.textContent = String(stats.clusterCount);
      renderNodeSelectors();
      renderNodeList();
      renderEdgeList();
      renderGraph();
      renderInspector();
      renderAbhiPanel();
      renderDiffPanel();
      renderQueryButtons();
      syncUndoButtons();
      syncViewButtons();
    }

    function scheduleSaveLayout() {
      if (READ_ONLY) return;
      window.clearTimeout(state.saveLayoutTimer);
      state.saveLayoutTimer = window.setTimeout(() => {
        request("/api/graph/ui", {
          method: "PATCH",
          body: JSON.stringify({
            project: state.scope.project,
            agent_id: state.scope.agent_id,
            session_id: state.scope.session_id,
            positions: state.positions,
            zoom: state.ui.zoom,
            viewport: state.ui.viewport,
            groups: state.ui.groups,
            collapsed_groups: state.ui.collapsed_groups,
            selected_nodes: state.selectedNodeIds,
          }),
        }).catch((error) => showStatus(error.message, true));
      }, 180);
    }

    function applyScopeInputs() {
      state.scope.project = els.scopeProject.value.trim();
      state.scope.agent_id = els.scopeAgent.value.trim();
      state.scope.session_id = els.scopeSession.value.trim();
    }

    async function loadGraph() {
      const payload = await request(`/api/graph${scopeQuery()}`);
      state.tenantId = payload.tenant_id || "";
      state.nodes = payload.nodes || [];
      state.edges = payload.edges || [];
      state.degreeMap = degreeMapForCurrentGraph();
      state.ui = payload.ui || state.ui;
      state.positions = {...(payload.ui?.positions || {})};
      state.selectedNodeIds = [...(payload.ui?.selected_nodes || [])];
      state.selectedEdgeId = "";
      state.selectedGroupId = "";
      els.statNodes.textContent = String(state.nodes.length);
      els.statEdges.textContent = String(state.edges.length);
      els.statTenant.textContent = state.tenantId || "-";
      rememberUiState();
      await Promise.all([loadAbhiPreview(), loadDiff(state.activitySince)]);
      renderAll();
    }

    async function loadAbhiPreview() {
      state.abhi = await request(`/api/graph/abhi${scopeQuery()}`);
    }

    async function loadDiff(since) {
      state.activitySince = since;
      state.diff = await request(`/api/graph/diff?since=${encodeURIComponent(since)}`);
    }

    function clearSelection() {
      rememberUiState();
      state.selectedNodeIds = [];
      state.selectedEdgeId = "";
      state.selectedGroupId = "";
      renderAll();
      scheduleSaveLayout();
    }

    function selectGroup(groupId) {
      state.selectedGroupId = groupId;
      state.selectedEdgeId = "";
      state.selectedNodeIds = [];
      renderAll();
    }

    function selectNode(nodeId, additive = false) {
      const exists = state.selectedNodeIds.includes(nodeId);
      rememberUiState();
      state.selectedGroupId = "";
      state.selectedEdgeId = "";
      if (additive) {
        state.selectedNodeIds = exists ? state.selectedNodeIds.filter((id) => id !== nodeId) : [...state.selectedNodeIds, nodeId];
      } else {
        state.selectedNodeIds = [nodeId];
      }
      renderAll();
      scheduleSaveLayout();
    }

    function selectEdge(edgeId) {
      rememberUiState();
      state.selectedEdgeId = edgeId;
      state.selectedNodeIds = [];
      state.selectedGroupId = "";
      renderAll();
      scheduleSaveLayout();
    }

    function normalizeBox(box) {
      return {
        x: Math.min(box.startX, box.endX),
        y: Math.min(box.startY, box.endY),
        width: Math.abs(box.endX - box.startX),
        height: Math.abs(box.endY - box.startY),
      };
    }

    function clientToSvg(clientX, clientY) {
      const point = els.graph.createSVGPoint();
      point.x = clientX;
      point.y = clientY;
      const inverse = els.graph.getScreenCTM().inverse();
      return point.matrixTransform(inverse);
    }

    function onNodeClick(event) {
      event.stopPropagation();
      selectNode(event.currentTarget.dataset.nodeId, event.shiftKey || event.metaKey || event.ctrlKey);
    }

    function onNodeMouseDown(event) {
      event.stopPropagation();
      const nodeId = event.currentTarget.dataset.nodeId;
      if (event.shiftKey && !READ_ONLY) {
        startConnect(nodeId, event);
        return;
      }
      if (!state.selectedNodeIds.includes(nodeId)) {
        state.selectedNodeIds = [nodeId];
      }
      rememberUiState();
      state.draggingNodeIds = [...state.selectedNodeIds];
      const anchor = clientToSvg(event.clientX, event.clientY);
      state.dragAnchor = {
        pointer: anchor,
        original: Object.fromEntries(state.draggingNodeIds.map((id) => [id, deepClone(state.positions[id])])),
      };
      window.addEventListener("mousemove", onDragMove);
      window.addEventListener("mouseup", stopDrag);
    }

    function onDragMove(event) {
      if (!state.dragAnchor || !state.draggingNodeIds.length) return;
      const point = clientToSvg(event.clientX, event.clientY);
      const dx = point.x - state.dragAnchor.pointer.x;
      const dy = point.y - state.dragAnchor.pointer.y;
      for (const nodeId of state.draggingNodeIds) {
        const original = state.dragAnchor.original[nodeId];
        state.positions[nodeId] = {x: original.x + dx, y: original.y + dy};
      }
      renderGraph();
    }

    function stopDrag() {
      if (state.draggingNodeIds.length) scheduleSaveLayout();
      state.draggingNodeIds = [];
      state.dragAnchor = null;
      window.removeEventListener("mousemove", onDragMove);
      window.removeEventListener("mouseup", stopDrag);
      renderAll();
    }

    function startBoxSelect(event) {
      if (event.target.dataset.nodeId || event.target.dataset.edgeId || event.target.dataset.groupId) return;
      rememberUiState();
      const point = clientToSvg(event.clientX, event.clientY);
      state.selectionBox = {startX: point.x, startY: point.y, endX: point.x, endY: point.y};
      state.selectedEdgeId = "";
      state.selectedGroupId = "";
      window.addEventListener("mousemove", moveBoxSelect);
      window.addEventListener("mouseup", finishBoxSelect);
      renderGraph();
    }

    function moveBoxSelect(event) {
      if (!state.selectionBox) return;
      const point = clientToSvg(event.clientX, event.clientY);
      state.selectionBox.endX = point.x;
      state.selectionBox.endY = point.y;
      renderGraph();
    }

    function finishBoxSelect() {
      if (!state.selectionBox) return;
      const box = normalizeBox(state.selectionBox);
      state.selectedNodeIds = filteredNodes()
        .filter((node) => {
          const pos = state.positions[node.id];
          return pos && pos.x >= box.x && pos.x <= box.x + box.width && pos.y >= box.y && pos.y <= box.y + box.height;
        })
        .map((node) => node.id);
      state.selectionBox = null;
      renderAll();
      scheduleSaveLayout();
      window.removeEventListener("mousemove", moveBoxSelect);
      window.removeEventListener("mouseup", finishBoxSelect);
    }

    function startConnect(nodeId, event) {
      event.preventDefault();
      state.connectFromId = nodeId;
      const point = clientToSvg(event.clientX, event.clientY);
      state.connectPreview = {x: point.x, y: point.y};
      window.addEventListener("mousemove", moveConnect);
      window.addEventListener("mouseup", finishConnect);
      renderGraph();
    }

    function moveConnect(event) {
      const point = clientToSvg(event.clientX, event.clientY);
      state.connectPreview = {x: point.x, y: point.y};
      renderGraph();
    }

    async function finishConnect(event) {
      window.removeEventListener("mousemove", moveConnect);
      window.removeEventListener("mouseup", finishConnect);
      if (!state.connectFromId) return;
      const target = event.target.closest?.("[data-node-id]");
      const sourceId = state.connectFromId;
      const targetId = target?.dataset?.nodeId || "";
      state.connectFromId = "";
      state.connectPreview = null;
      renderGraph();
      if (!targetId || targetId === sourceId || READ_ONLY) return;
      try {
        await request("/api/graph/edges", {
          method: "POST",
          body: JSON.stringify({
            source_id: sourceId,
            target_id: targetId,
            relationship: document.getElementById("create-edge-relationship").value,
            weight: 1,
          }),
        });
        showStatus("Edge created.");
        await loadGraph();
      } catch (error) {
        showStatus(error.message, true);
      }
    }

    async function createNode() {
      if (READ_ONLY) throw new Error("Read-only mode.");
      await request("/api/graph/nodes", {
        method: "POST",
        body: JSON.stringify({
          label: document.getElementById("create-node-label").value.trim(),
          content: document.getElementById("create-node-content").value.trim(),
          node_type: document.getElementById("create-node-type").value,
          tags: parseTags(document.getElementById("create-node-tags").value),
          project: state.scope.project,
          agent_id: state.scope.agent_id,
          session_id: state.scope.session_id,
        }),
      });
      document.getElementById("create-node-label").value = "";
      document.getElementById("create-node-content").value = "";
      document.getElementById("create-node-tags").value = "";
      showStatus("Node created.");
      await loadGraph();
    }

    async function updateNode() {
      if (READ_ONLY) throw new Error("Read-only mode.");
      const nodeId = document.getElementById("edit-node-id").value.trim();
      if (!nodeId) throw new Error("Select one node first.");
      await request(`/api/graph/nodes/${encodeURIComponent(nodeId)}`, {
        method: "PATCH",
        body: JSON.stringify({
          label: document.getElementById("edit-node-label").value.trim(),
          content: document.getElementById("edit-node-content").value.trim(),
          tags: parseTags(document.getElementById("edit-node-tags").value),
        }),
      });
      showStatus("Node updated.");
      await loadGraph();
      selectNode(nodeId);
    }

    async function deleteNode() {
      if (READ_ONLY) throw new Error("Read-only mode.");
      const nodeId = document.getElementById("edit-node-id").value.trim();
      if (!nodeId) throw new Error("Select one node first.");
      await request(`/api/graph/nodes/${encodeURIComponent(nodeId)}`, {method: "DELETE"});
      showStatus("Node deleted.");
      await loadGraph();
    }

    async function updateEdge() {
      if (READ_ONLY) throw new Error("Read-only mode.");
      const edgeId = document.getElementById("edit-edge-id").value.trim();
      if (!edgeId) throw new Error("Select an edge first.");
      await request(`/api/graph/edges/${encodeURIComponent(edgeId)}`, {
        method: "PATCH",
        body: JSON.stringify({
          source_id: document.getElementById("edit-edge-source").value,
          target_id: document.getElementById("edit-edge-target").value,
          relationship: document.getElementById("edit-edge-relationship").value,
          weight: Number(document.getElementById("edit-edge-weight").value || "1"),
        }),
      });
      showStatus("Edge updated.");
      await loadGraph();
      selectEdge(edgeId);
    }

    async function deleteEdge() {
      if (READ_ONLY) throw new Error("Read-only mode.");
      const edgeId = document.getElementById("edit-edge-id").value.trim();
      if (!edgeId) throw new Error("Select an edge first.");
      await request(`/api/graph/edges/${encodeURIComponent(edgeId)}`, {method: "DELETE"});
      showStatus("Edge deleted.");
      await loadGraph();
    }

    async function deleteSelected() {
      if (READ_ONLY) throw new Error("Read-only mode.");
      if (state.selectedEdgeId) {
        await deleteEdge();
        return;
      }
      if (!state.selectedNodeIds.length) throw new Error("Select node(s) or an edge first.");
      for (const nodeId of [...state.selectedNodeIds]) {
        await request(`/api/graph/nodes/${encodeURIComponent(nodeId)}`, {method: "DELETE"});
      }
      showStatus("Selection deleted.");
      await loadGraph();
    }

    async function duplicateSelected() {
      if (READ_ONLY) throw new Error("Read-only mode.");
      if (!state.selectedNodeIds.length) throw new Error("Select at least one node.");
      for (const nodeId of state.selectedNodeIds) {
        const node = nodeById(nodeId);
        if (!node) continue;
        await request("/api/graph/nodes", {
          method: "POST",
          body: JSON.stringify({
            label: `${node.label} Copy`,
            content: node.content,
            node_type: node.node_type,
            tags: node.tags || [],
            project: node.project || state.scope.project,
            agent_id: node.agent_id || state.scope.agent_id,
            session_id: node.session_id || state.scope.session_id,
          }),
        });
      }
      showStatus("Selection duplicated.");
      await loadGraph();
    }

    function fitLayout() {
      rememberUiState();
      smartLayout(filteredNodes());
      renderAll();
      scheduleSaveLayout();
    }

    function createGroupFromSelection() {
      if (READ_ONLY) throw new Error("Read-only mode.");
      if (!state.selectedNodeIds.length) throw new Error("Select node(s) to group.");
      rememberUiState();
      const label = document.getElementById("group-label").value.trim() || `Group ${state.ui.groups.length + 1}`;
      const color = document.getElementById("group-color").value.trim() || "#4A90D9";
      const group = {
        id: `group-${Math.random().toString(36).slice(2, 10)}`,
        label,
        members: [...state.selectedNodeIds],
        color,
      };
      state.ui.groups = [...(state.ui.groups || []), group];
      state.selectedGroupId = group.id;
      renderAll();
      scheduleSaveLayout();
    }

    function toggleSelectedGroup() {
      if (READ_ONLY) throw new Error("Read-only mode.");
      const group = currentGroup();
      if (!group) throw new Error("Select a group first.");
      rememberUiState();
      const collapsed = new Set(state.ui.collapsed_groups || []);
      if (collapsed.has(group.id)) collapsed.delete(group.id);
      else collapsed.add(group.id);
      state.ui.collapsed_groups = [...collapsed];
      renderAll();
      scheduleSaveLayout();
    }

    function deleteSelectedGroup() {
      if (READ_ONLY) throw new Error("Read-only mode.");
      const group = currentGroup();
      if (!group) throw new Error("Select a group first.");
      rememberUiState();
      state.ui.groups = (state.ui.groups || []).filter((item) => item.id !== group.id);
      state.ui.collapsed_groups = (state.ui.collapsed_groups || []).filter((id) => id !== group.id);
      state.selectedGroupId = "";
      renderAll();
      scheduleSaveLayout();
    }

    async function importGraph() {
      if (READ_ONLY) throw new Error("Read-only mode.");
      const file = document.getElementById("import-file").files[0];
      if (!file) throw new Error("Choose a file to import.");
      await request("/api/graph/import", {
        method: "POST",
        body: JSON.stringify({
          format: document.getElementById("import-format").value,
          content: await file.text(),
        }),
      });
      showStatus("Import completed.");
      await loadGraph();
    }

    async function runSavedQuery(queryId) {
      const payload = await request("/api/graph/query", {
        method: "POST",
        body: JSON.stringify({...state.scope, query_id: queryId}),
      });
      applyQueryPayload(payload);
    }

    async function runCustomQuery() {
      const text = document.getElementById("custom-query").value.trim();
      if (!text) throw new Error("Enter a query first.");
      const payload = await request("/api/graph/query", {
        method: "POST",
        body: JSON.stringify({...state.scope, query: text}),
      });
      applyQueryPayload(payload);
    }

    function applyQueryPayload(payload) {
      state.queryMatchNodeIds = (payload.nodes || []).map((node) => node.id);
      state.queryMatchEdgeIds = (payload.edges || []).map((edge) => edge.id);
      state.querySummary = payload.summary || "Query complete.";
      els.querySummary.textContent = `${payload.name ? `${payload.name}: ` : ""}${state.querySummary}`;
      renderAll();
      showStatus("Query executed.");
    }

    function clearQueryHighlight() {
      state.queryMatchNodeIds = [];
      state.queryMatchEdgeIds = [];
      state.querySummary = "";
      els.querySummary.textContent = "No query active.";
      renderAll();
    }

    function wireEvents() {
      document.getElementById("refresh-btn").addEventListener("click", () => loadGraph().then(() => showStatus("Graph refreshed.")).catch((error) => showStatus(error.message, true)));
      document.getElementById("apply-scope-btn").addEventListener("click", () => {
        applyScopeInputs();
        loadGraph().then(() => showStatus("Scope applied.")).catch((error) => showStatus(error.message, true));
      });
      document.getElementById("undo-btn").addEventListener("click", undoUiState);
      document.getElementById("redo-btn").addEventListener("click", redoUiState);
      document.getElementById("fit-btn").addEventListener("click", fitLayout);
      document.getElementById("focus-mode-btn").addEventListener("click", () => {
        state.view.focusMode = !state.view.focusMode;
        syncViewButtons();
      });
      document.getElementById("labels-btn").addEventListener("click", () => {
        state.view.showLabels = !state.view.showLabels;
        renderGraph();
        syncViewButtons();
      });
      document.getElementById("clear-selection-btn").addEventListener("click", clearSelection);
      document.getElementById("duplicate-btn").addEventListener("click", () => duplicateSelected().catch((error) => showStatus(error.message, true)));
      document.getElementById("delete-selected-btn").addEventListener("click", () => deleteSelected().catch((error) => showStatus(error.message, true)));
      document.getElementById("create-node-btn").addEventListener("click", () => createNode().catch((error) => showStatus(error.message, true)));
      document.getElementById("update-node-btn").addEventListener("click", () => updateNode().catch((error) => showStatus(error.message, true)));
      document.getElementById("delete-node-btn").addEventListener("click", () => deleteNode().catch((error) => showStatus(error.message, true)));
      document.getElementById("update-edge-btn").addEventListener("click", () => updateEdge().catch((error) => showStatus(error.message, true)));
      document.getElementById("delete-edge-btn").addEventListener("click", () => deleteEdge().catch((error) => showStatus(error.message, true)));
      document.getElementById("create-group-btn").addEventListener("click", () => { try { createGroupFromSelection(); } catch (error) { showStatus(error.message, true); } });
      document.getElementById("toggle-group-btn").addEventListener("click", () => { try { toggleSelectedGroup(); } catch (error) { showStatus(error.message, true); } });
      document.getElementById("delete-group-btn").addEventListener("click", () => { try { deleteSelectedGroup(); } catch (error) { showStatus(error.message, true); } });
      document.getElementById("run-custom-query-btn").addEventListener("click", () => runCustomQuery().catch((error) => showStatus(error.message, true)));
      document.getElementById("clear-query-btn").addEventListener("click", clearQueryHighlight);
      document.getElementById("import-btn").addEventListener("click", () => importGraph().catch((error) => showStatus(error.message, true)));
      document.getElementById("export-json-btn").addEventListener("click", () => window.open(`/api/graph/export?format=json${scopeQuery().replace("?", "&")}`, "_blank"));
      document.getElementById("export-abhi-btn").addEventListener("click", () => window.open(`/api/graph/export?format=abhi${scopeQuery().replace("?", "&")}`, "_blank"));
      document.getElementById("activity-24h-btn").addEventListener("click", () => loadDiff("24h").then(renderDiffPanel).catch((error) => showStatus(error.message, true)));
      document.getElementById("activity-7d-btn").addEventListener("click", () => loadDiff("7d").then(renderDiffPanel).catch((error) => showStatus(error.message, true)));
      document.getElementById("activity-30d-btn").addEventListener("click", () => loadDiff("30d").then(renderDiffPanel).catch((error) => showStatus(error.message, true)));
      els.searchInput.addEventListener("input", (event) => {
        state.search = event.target.value;
        renderAll();
      });
      document.addEventListener("keydown", (event) => {
        if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z") {
          event.preventDefault();
          if (event.shiftKey) redoUiState();
          else undoUiState();
          return;
        }
        if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "d" && !READ_ONLY) {
          event.preventDefault();
          duplicateSelected().catch((error) => showStatus(error.message, true));
          return;
        }
        if ((event.key === "Delete" || event.key === "Backspace") && !READ_ONLY) {
          const target = document.activeElement;
          const tag = target?.tagName?.toLowerCase?.() || "";
          if (tag === "input" || tag === "textarea") return;
          deleteSelected().catch((error) => showStatus(error.message, true));
          return;
        }
        if (event.key === "Escape") {
          clearSelection();
        }
      });
    }

    wireEvents();
    loadGraph().catch((error) => showStatus(error.message, true));
  </script>
</body>
</html>"""
