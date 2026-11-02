"""
DrugLab Interactive Structure Inspection & Preparation Review GUI.

Provides an interactive 3D Web interface (powered by 3Dmol.js) and a CLI fallback
for transparent receptor review before docking:
1. Interactive 3D visualization of protein structure with toggleable representations:
   - Cartoon, Ball & Stick, Stick, Ribbon, Sphere (Spacefill), Surface.
   - Coloring by Chain, Secondary Structure, pLDDT/B-factor, Element (CPK).
   - Component visibility toggles (Waters, Metals, Ligands, Cofactors).
2. Multi-chain selection with checkboxes:
   - Select all, individual chains (Chain A, Chain B, etc.), or any combination.
   - Real-time 3D visual feedback (unselected chains dimmed).
3. Component preparation table with scientific rationales, docking consequences,
   and engine-specific safety constraints.
4. Terminal review fallback with keyboard shortcuts.
"""

from __future__ import annotations

import os
import re
import sys
import json
import socket
import select
import threading
import webbrowser
import http.server
from pathlib import Path
from typing import Any, Tuple

from utils.logger import setup_logger

logger = setup_logger("structure_review_gui", level="INFO")

# Chain color palette for distinct 3D visualization
CHAIN_COLORS = [
    "#3b82f6",  # Blue
    "#10b981",  # Emerald Green
    "#8b5cf6",  # Purple
    "#f59e0b",  # Amber
    "#ec4899",  # Pink
    "#06b6d4",  # Cyan
    "#84cc16",  # Lime
    "#e11d48",  # Rose
    "#6366f1",  # Indigo
    "#14b8a6",  # Teal
]


def find_free_port(preferred: int = 8770) -> int:
    """Find an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _StructureReviewHandler(http.server.BaseHTTPRequestHandler):
    """HTTP Request Handler for the Structure Review GUI."""

    def log_message(self, format, *args):
        # Suppress noisy HTTP request logs
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            content = self.server.html_content.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/confirm_preparation":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(body) if body else {}
                self.server.user_decisions = data.get("decisions", {})
                if hasattr(self.server, "done_event"):
                    self.server.done_event.set()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"confirmed"}')
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        elif self.path == "/update_decisions":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(body) if body else {}
                self.server.user_decisions = data.get("decisions", {})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"updated"}')
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def generate_review_html(
    pdb_path: str | Path,
    inspection_data: dict[str, Any],
    port: int,
) -> str:
    """Generate the complete 3Dmol.js HTML review application."""
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        pdb_content = f.read()

    # Safely JSON-encode PDB contents and inspection metadata for JS injection
    protein_json = json.dumps(pdb_content)
    inspection_json = json.dumps(inspection_data)

    workflow_mode = inspection_data.get("workflow_mode", "small_molecule")
    workflow_label = (
        "Protein–Small Molecule Docking (Vina)"
        if workflow_mode == "small_molecule"
        else "Protein–Peptide Docking (LightDock)"
    )

    # Chain color map
    chain_ids = inspection_data.get("chain_ids", [])
    chain_color_map = {
        cid: CHAIN_COLORS[i % len(CHAIN_COLORS)]
        for i, cid in enumerate(chain_ids)
    }
    chain_color_json = json.dumps(chain_color_map)

    # Pre-render static HTML for chain checkboxes
    chains = inspection_data.get("chains", [])
    if not chains and chain_ids:
        chains = [{"chain_id": cid, "num_residues": 0, "sequence_preview": ""} for cid in chain_ids]

    chains_html_parts = []
    if not chains:
        chains_html_parts.append('<div style="font-size:0.78rem;color:#9ca3af;">Single chain detected (Chain A).</div>')
    else:
        for ch in chains:
            cid = ch.get("chain_id", "A")
            color = chain_color_map.get(cid, "#3b82f6")
            seq_snippet = (ch.get("sequence_preview") or "")[:16]
            num_res = ch.get("num_residues", 0)
            chains_html_parts.append(
                f'<label class="chain-row" for="chk-chain-{cid}">'
                f'<input type="checkbox" class="chain-cb" id="chk-chain-{cid}" value="{cid}" checked onchange="onChainCheckboxChange(\'{cid}\', this.checked)">'
                f'<span class="chain-pill" style="background:{color};">{cid}</span>'
                f'<div class="chain-info">'
                f'<span style="font-weight:600;">Chain {cid} ({num_res} AA)</span>'
                f'<span class="chain-seq">{seq_snippet}</span>'
                f'</div>'
                f'</label>'
            )
    rendered_chains_html = "\n".join(chains_html_parts)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>DrugLab — Receptor Structure Inspection & Preparation Review</title>
  <!-- Load 3Dmol.js from CDN with fallback -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"></script>
  <script>
    if (typeof $3Dmol === 'undefined') {{
      document.write('<script src="https://3Dmol.org/build/3Dmol-min.js"><\\/script>');
    }}
  </script>
  <style>
    :root {{
      --bg: #090d16;
      --card-bg: #111827;
      --card-border: #1f2937;
      --accent: #3b82f6;
      --accent-hover: #2563eb;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --danger: #ef4444;
      --success: #10b981;
      --warning: #f59e0b;
      --purple: #8b5cf6;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
    body {{
      background: var(--bg);
      color: var(--text);
      display: flex;
      height: 100vh;
      overflow: hidden;
    }}
    #viewer-container {{
      flex: 1;
      position: relative;
      height: 100%;
      background: radial-gradient(circle at center, #172033 0%, #090d16 100%);
    }}
    #gmol {{
      width: 100%;
      height: 100%;
    }}
    #sidebar {{
      width: 480px;
      min-width: 440px;
      height: 100%;
      background: var(--card-bg);
      border-left: 1px solid var(--card-border);
      display: flex;
      flex-direction: column;
      z-index: 10;
      box-shadow: -4px 0 24px rgba(0,0,0,0.5);
    }}
    .sidebar-header {{
      padding: 16px 20px;
      border-bottom: 1px solid var(--card-border);
      background: #0f172a;
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.02em;
    }}
    .brand-badge {{
      background: rgba(59, 130, 246, 0.2);
      color: #60a5fa;
      border: 1px solid rgba(59, 130, 246, 0.4);
      font-size: 0.72rem;
      padding: 2px 7px;
      border-radius: 4px;
      font-weight: 600;
      text-transform: uppercase;
    }}
    .workflow-pill {{
      margin-top: 5px;
      font-size: 0.8rem;
      color: var(--text-muted);
    }}
    .sidebar-body {{
      flex: 1;
      overflow-y: auto;
      padding: 16px 20px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .meta-box {{
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 12px 14px;
      font-size: 0.83rem;
    }}
    .meta-row {{
      display: flex;
      justify-content: space-between;
      margin-bottom: 5px;
    }}
    .meta-row:last-child {{ margin-bottom: 0; }}
    .meta-label {{ color: var(--text-muted); }}
    .meta-val {{ font-weight: 600; }}

    /* Chain selection styling */
    .chains-box {{
      background: #141c2e;
      border: 1px solid #253349;
      border-radius: 8px;
      padding: 12px 14px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .chains-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .chains-title {{
      font-size: 0.82rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #93c5fd;
    }}
    .chain-action-links {{
      display: flex;
      gap: 8px;
      font-size: 0.75rem;
    }}
    .chain-link {{
      color: var(--accent);
      cursor: pointer;
      text-decoration: underline;
    }}
    .chain-link:hover {{ color: #93c5fd; }}
    .chains-list {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .chain-row {{
      display: flex;
      align-items: center;
      gap: 9px;
      padding: 6px 8px;
      background: #0f172a;
      border: 1px solid #1e293b;
      border-radius: 6px;
      font-size: 0.82rem;
      cursor: pointer;
      user-select: none;
      transition: background 0.15s;
    }}
    .chain-row:hover {{
      background: #1e293b;
    }}
    .chain-pill {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 22px;
      height: 22px;
      border-radius: 4px;
      color: white;
      font-weight: 700;
      font-size: 0.75rem;
    }}
    .chain-info {{
      flex: 1;
      display: flex;
      justify-content: space-between;
    }}
    .chain-seq {{
      color: var(--text-muted);
      font-family: monospace;
      font-size: 0.75rem;
    }}

    .af-warning {{
      background: rgba(245, 158, 11, 0.12);
      border: 1px solid rgba(245, 158, 11, 0.35);
      border-radius: 8px;
      padding: 12px;
      font-size: 0.82rem;
      color: #fde68a;
      line-height: 1.45;
    }}
    .af-badge {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      background: rgba(139, 92, 246, 0.2);
      color: #c084fc;
      border: 1px solid rgba(139, 92, 246, 0.4);
      padding: 2px 7px;
      border-radius: 4px;
      font-weight: 600;
      font-size: 0.76rem;
    }}
    .section-title {{
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      font-weight: 700;
      margin-top: 4px;
      margin-bottom: 2px;
    }}
    .component-card {{
      background: #161e2e;
      border: 1px solid #283548;
      border-radius: 8px;
      padding: 12px 14px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .card-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .component-name {{
      font-weight: 600;
      font-size: 0.88rem;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .component-color-dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
    }}
    .action-select {{
      background: #0f172a;
      border: 1px solid #334155;
      color: var(--text);
      font-size: 0.82rem;
      padding: 4px 8px;
      border-radius: 5px;
      font-weight: 600;
      cursor: pointer;
    }}
    .action-select:disabled {{
      opacity: 0.6;
      cursor: not-allowed;
    }}
    .rationale-text {{
      font-size: 0.8rem;
      color: #cbd5e1;
      line-height: 1.4;
    }}
    .consequence-tag {{
      font-size: 0.77rem;
      color: #94a3b8;
      background: #0f172a;
      padding: 6px 8px;
      border-radius: 5px;
      border-left: 3px solid var(--accent);
      line-height: 1.35;
    }}
    .unsafe-warn {{
      font-size: 0.76rem;
      color: #f87171;
    }}
    .sidebar-footer {{
      padding: 14px 20px;
      border-top: 1px solid var(--card-border);
      background: #0f172a;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .btn-confirm {{
      background: var(--accent);
      color: white;
      border: none;
      border-radius: 6px;
      padding: 12px;
      font-weight: 700;
      font-size: 0.92rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: background 0.15s ease;
    }}
    .btn-confirm:hover {{
      background: var(--accent-hover);
    }}
    .confirmed-msg {{
      color: var(--success);
      font-weight: 600;
      font-size: 0.85rem;
      text-align: center;
      display: none;
    }}

    /* Top Controls / View Styles Toolbar */
    .top-toolbar {{
      position: absolute;
      top: 16px;
      left: 16px;
      background: rgba(15, 23, 42, 0.94);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.15);
      border-radius: 8px;
      padding: 8px 12px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.6);
      z-index: 50;
      pointer-events: auto;
    }}
    .toolbar-group {{
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 0.78rem;
    }}
    .toolbar-label {{
      color: var(--text-muted);
      font-weight: 600;
      text-transform: uppercase;
      font-size: 0.7rem;
      letter-spacing: 0.05em;
    }}
    .toolbar-select {{
      background: #1e293b;
      border: 1px solid #334155;
      color: var(--text);
      font-size: 0.78rem;
      padding: 4px 8px;
      border-radius: 5px;
      cursor: pointer;
    }}
    .toggle-pill {{
      background: #1e293b;
      border: 1px solid #334155;
      color: var(--text-muted);
      font-size: 0.74rem;
      padding: 4px 9px;
      border-radius: 4px;
      cursor: pointer;
      font-weight: 600;
      transition: all 0.15s ease;
    }}
    .toggle-pill.active {{
      background: rgba(59, 130, 246, 0.25);
      border-color: #3b82f6;
      color: #93c5fd;
    }}
    .ctrl-btn {{
      background: #1e293b;
      border: 1px solid #334155;
      color: var(--text);
      font-size: 0.76rem;
      padding: 4px 10px;
      border-radius: 5px;
      cursor: pointer;
      font-weight: 600;
    }}
    .ctrl-btn:hover {{ background: #334155; }}

    /* Viewer legend */
    .viewer-legend {{
      position: absolute;
      bottom: 20px;
      left: 20px;
      background: rgba(15, 23, 42, 0.9);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 8px;
      padding: 12px 14px;
      font-size: 0.76rem;
      color: var(--text);
      display: flex;
      flex-direction: column;
      gap: 6px;
      z-index: 5;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .legend-color {{
      width: 12px;
      height: 12px;
      border-radius: 3px;
    }}
  </style>
</head>
<body>
  <div id="viewer-container">
    <div id="gmol"></div>

    <!-- View & Display Controls Toolbar -->
    <div class="top-toolbar">
      <div class="toolbar-group">
        <span class="toolbar-label">View Style:</span>
        <select id="select-rep" class="toolbar-select" onchange="changeRepresentation(this.value)">
          <option value="cartoon" selected>Cartoon (Default)</option>
          <option value="ballstick">Ball & Stick</option>
          <option value="stick">Stick</option>
          <option value="ribbon">Ribbon</option>
          <option value="sphere">Sphere (Spacefill)</option>
          <option value="surface">Surface</option>
        </select>
      </div>

      <div class="toolbar-group">
        <span class="toolbar-label">Color:</span>
        <select id="select-color" class="toolbar-select" onchange="changeColoring(this.value)">
          <option value="chain" selected>By Chain</option>
          <option value="ss">By Secondary Structure</option>
          <option value="plddt">By B-factor / pLDDT</option>
          <option value="element">By Element (CPK)</option>
        </select>
      </div>

      <div class="toolbar-group" style="border-left: 1px solid #334155; padding-left: 10px;">
        <span class="toolbar-label">Show:</span>
        <button type="button" id="btn-show-waters" class="toggle-pill active" onclick="toggleComponentDisplay('waters')">Waters: ON</button>
        <button type="button" id="btn-show-ligands" class="toggle-pill active" onclick="toggleComponentDisplay('ligands')">Ligands: ON</button>
        <button type="button" id="btn-show-metals" class="toggle-pill active" onclick="toggleComponentDisplay('metals')">Metals: ON</button>
        <button type="button" id="btn-show-cofactors" class="toggle-pill active" onclick="toggleComponentDisplay('cofactors')">Cofactors: ON</button>
      </div>

      <div class="toolbar-group" style="border-left: 1px solid #334155; padding-left: 10px;">
        <button type="button" class="ctrl-btn" onclick="resetCamera()">Reset View</button>
        <button type="button" id="btn-spin" class="ctrl-btn" onclick="toggleSpin()">Spin: OFF</button>
      </div>
    </div>

    <!-- Color Legend -->
    <div class="viewer-legend">
      <div style="font-weight:700;margin-bottom:2px;color:var(--text-muted);font-size:0.7rem;letter-spacing:0.05em;">VISUAL LEGEND</div>
      <div class="legend-item"><span class="legend-color" style="background:#3b82f6;"></span> Receptor Chains (Colored or Dimmed)</div>
      <div class="legend-item"><span class="legend-color" style="background:#ef4444;"></span> Waters (Red Spheres)</div>
      <div class="legend-item"><span class="legend-color" style="background:#10b981;"></span> Ligands / Hetero (Green Sticks)</div>
      <div class="legend-item"><span class="legend-color" style="background:#f59e0b;"></span> Metals (Orange Spheres)</div>
      <div class="legend-item"><span class="legend-color" style="background:#d946ef;"></span> Cofactors (Magenta Sticks)</div>
      <div class="legend-item"><span class="legend-color" style="background:#06b6d4;"></span> Altloc Conformations (Cyan)</div>
    </div>
  </div>

  <div id="sidebar">
    <div class="sidebar-header">
      <div class="brand">
        DrugLab <span>Structure Review</span>
        <span class="brand-badge">Step 1 of 2</span>
      </div>
      <div class="workflow-pill">{workflow_label}</div>
    </div>

    <div class="sidebar-body">
      <!-- Target Metadata Box -->
      <div class="meta-box">
        <div class="meta-row"><span class="meta-label">Structure:</span><span class="meta-val">{inspection_data.get("file_name")}</span></div>
        <div class="meta-row"><span class="meta-label">Source:</span><span class="meta-val">{inspection_data.get("structure_source")}</span></div>
        <div class="meta-row"><span class="meta-label">Total Residues / Atoms:</span><span class="meta-val">{inspection_data.get("total_residues")} res / {inspection_data.get("total_atoms")} atoms</span></div>
      </div>

      <!-- AlphaFold Caveats Section -->
      <div id="af-container"></div>

      <!-- Multiple Protein Chains Selection Card (Issue 2) -->
      <div class="chains-box">
        <div class="chains-header">
          <span class="chains-title">Protein Chains ({len(chain_ids)} found)</span>
          <div class="chain-action-links">
            <span class="chain-link" onclick="selectAllChains(true)">Select All</span> |
            <span class="chain-link" onclick="selectFirstChainOnly()">Chain {chain_ids[0] if chain_ids else 'A'} Only</span>
          </div>
        </div>
        <div style="font-size:0.75rem; color:var(--text-muted);">
          Select chains to keep for docking. Unchecked chains will be removed during preparation:
        </div>
        <div id="chains-checkbox-list" class="chains-list">
{rendered_chains_html}
        </div>
      </div>

      <!-- Detected Components & Proposed Actions -->
      <div>
        <div class="section-title">Detected Components &amp; Preparation Decisions</div>
        <div id="components-list" style="display:flex; flex-direction:column; gap:10px;"></div>
      </div>
    </div>

    <div class="sidebar-footer">
      <button id="btn-confirm" class="btn-confirm" onclick="confirmPreparation()">
        <span>Confirm Preparation &amp; Proceed to Docking</span>
      </button>
      <div id="confirmed-msg" class="confirmed-msg">
        ✓ Preparation settings confirmed! Proceeding in DrugLab...
      </div>
      <div style="font-size:0.74rem; color:var(--text-muted); text-align:center;">
        You can also confirm by pressing [Enter] in your terminal.
      </div>
    </div>
  </div>

  <script>
    const INSPECTION = {inspection_json};
    const PDB_DATA = {protein_json};
    const CHAIN_COLORS = {chain_color_json};

    let viewer = null;
    let isSpinning = false;
    let currentRep = 'cartoon';
    let currentColoring = 'chain';
    let currentSurface = null;
    let currentDecisions = {{}};
    let selectedChains = [...(INSPECTION.chain_ids || ['A'])];

    // Visibility toggles for non-protein components
    let showComponents = {{
      waters: true,
      ligands: true,
      metals: true,
      cofactors: true
    }};

    // Common solvent/heteroatom residue names to exclude from protein backbone
    const SOLVENT_HETS = ['HOH', 'WAT', 'H2O', 'TIP3', 'DOD', 'PEG', 'EDO', 'GOL', 'DMS', 'SO4', 'PO4', 'CIT', 'ACT'];

    window.addEventListener('DOMContentLoaded', () => {{
      initViewer();
      setupInitialDecisions();
      renderAlphaFoldSection();
      renderChainsList();
      renderComponentsList();
      apply3DStyles();
    }});

    function initViewer() {{
      const elem = document.getElementById('gmol');
      try {{
        if (typeof $3Dmol === 'undefined') {{
          console.warn("Waiting for $3Dmol library to load...");
          setTimeout(initViewer, 80);
          return;
        }}
        if (!viewer) {{
          viewer = $3Dmol.createViewer(elem, {{ backgroundColor: '#090d16' }});
          if (PDB_DATA && PDB_DATA.length > 20) {{
            viewer.addModel(PDB_DATA, "pdb");
            viewer.zoomTo();
            viewer.render();
          }}
          apply3DStyles();
        }}
      }} catch (err) {{
        console.error("Error creating 3Dmol viewer:", err);
      }}
    }}

    function setupInitialDecisions() {{
      (INSPECTION.review_items || []).forEach(item => {{
        currentDecisions[item.id] = item.proposed_action;
      }});
      currentDecisions['selected_chains'] = [...selectedChains];
    }}

    function renderAlphaFoldSection() {{
      const af = INSPECTION.alphafold_caveats;
      const c = document.getElementById('af-container');
      if (!af || !c) return;

      const stats = af.plddt_stats || {{}};
      c.innerHTML = `
        <div class="af-warning">
          <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:5px;">
            <span class="af-badge">AlphaFold Predicted Model</span>
            <span style="font-weight:700;">Mean pLDDT: ${{af.mean_plddt || 'N/A'}}</span>
          </div>
          <div style="margin-bottom:6px;">${{af.disclaimer || ''}}</div>
          <div style="display:flex; gap:8px; font-size:0.75rem; font-weight:600;">
            <span style="color:#60a5fa;">Very High (>90): ${{stats.very_high_pct || 0}}%</span>
            <span style="color:#34d399;">Confident (70-90): ${{stats.confident_pct || 0}}%</span>
            <span style="color:#fbbf24;">Low (<70): ${{stats.low_pct || 0}}%</span>
          </div>
        </div>
      `;
    }}

    // Multi-Chain Selection (Issue 2)
    function renderChainsList() {{
      const container = document.getElementById('chains-checkbox-list');
      if (!container) return;
      container.innerHTML = '';

      const chains = INSPECTION.chains || [];
      if (chains.length === 0) {{
        container.innerHTML = '<div style="font-size:0.78rem;color:#9ca3af;">Single chain detected (Chain A).</div>';
        return;
      }}

      chains.forEach((ch, idx) => {{
        const cid = ch.chain_id;
        const color = CHAIN_COLORS[cid] || '#3b82f6';
        const isChecked = selectedChains.includes(cid);

        const row = document.createElement('label');
        row.className = 'chain-row';
        row.htmlFor = 'chk-chain-' + cid;

        row.innerHTML = `
          <input type="checkbox" class="chain-cb" id="chk-chain-${{cid}}" value="${{cid}}" ${{isChecked ? 'checked' : ''}} onchange="onChainCheckboxChange('${{cid}}', this.checked)">
          <span class="chain-pill" style="background:${{color}};">${{cid}}</span>
          <div class="chain-info">
            <span style="font-weight:600;">Chain ${{cid}} (${{ch.num_residues}} AA)</span>
            <span class="chain-seq">${{ch.sequence_preview ? ch.sequence_preview.slice(0, 16) : ''}}</span>
          </div>
        `;
        container.appendChild(row);
      }});
    }}

    function onChainCheckboxChange(cid, isChecked) {{
      if (isChecked) {{
        if (!selectedChains.includes(cid)) selectedChains.push(cid);
      }} else {{
        // Prevent deselecting all chains
        if (selectedChains.length <= 1) {{
          alert("At least one protein chain must remain selected for docking.");
          document.getElementById('chk-chain-' + cid).checked = true;
          return;
        }}
        selectedChains = selectedChains.filter(c => c !== cid);
      }}

      currentDecisions['selected_chains'] = [...selectedChains];
      apply3DStyles();
      syncDecisionsToServer();
    }}

    function selectAllChains(select) {{
      const allIds = (INSPECTION.chains || []).map(c => c.chain_id);
      if (allIds.length === 0) return;

      selectedChains = [...allIds];
      document.querySelectorAll('.chain-cb').forEach(cb => cb.checked = true);
      currentDecisions['selected_chains'] = [...selectedChains];
      apply3DStyles();
      syncDecisionsToServer();
    }}

    function selectFirstChainOnly() {{
      const allIds = (INSPECTION.chains || []).map(c => c.chain_id);
      if (allIds.length === 0) return;
      const firstId = allIds[0];

      selectedChains = [firstId];
      document.querySelectorAll('.chain-cb').forEach(cb => {{
        cb.checked = (cb.value === firstId);
      }});
      currentDecisions['selected_chains'] = [...selectedChains];
      apply3DStyles();
      syncDecisionsToServer();
    }}

    // View Style Controls (Issue 1)
    function changeRepresentation(rep) {{
      currentRep = rep;
      apply3DStyles();
    }}

    function changeColoring(col) {{
      currentColoring = col;
      apply3DStyles();
    }}

    function toggleComponentDisplay(type) {{
      showComponents[type] = !showComponents[type];
      const btn = document.getElementById('btn-show-' + type);
      if (btn) {{
        const label = type.charAt(0).toUpperCase() + type.slice(1);
        btn.textContent = label + ': ' + (showComponents[type] ? 'ON' : 'OFF');
        if (showComponents[type]) btn.classList.add('active');
        else btn.classList.remove('active');
      }}
      apply3DStyles();
    }}

    function resetCamera() {{
      if (!viewer) return;
      viewer.zoomTo();
      viewer.render();
    }}

    function toggleSpin() {{
      if (!viewer) return;
      isSpinning = !isSpinning;
      viewer.spin(isSpinning);
      const btn = document.getElementById('btn-spin');
      if (btn) {{
        btn.textContent = 'Spin: ' + (isSpinning ? 'ON' : 'OFF');
        if (isSpinning) btn.classList.add('active');
        else btn.classList.remove('active');
      }}
    }}

    // Component review cards
    function getComponentDotColor(type) {{
      switch(type) {{
        case 'water': return '#ef4444';
        case 'ligand': return '#10b981';
        case 'metal': return '#f59e0b';
        case 'cofactor': return '#d946ef';
        case 'altloc': return '#06b6d4';
        case 'extra_chain': return '#8b5cf6';
        default: return '#3b82f6';
      }}
    }}

    function renderComponentsList() {{
      const list = document.getElementById('components-list');
      if (!list) return;
      list.innerHTML = '';

      const items = (INSPECTION.review_items || []).filter(it => it.id !== 'chains');
      if (items.length === 0) {{
        list.innerHTML = '<div style="font-size:0.83rem; color:#9ca3af;">Receptor contains standard amino acids only.</div>';
        return;
      }}

      items.forEach(item => {{
        const card = document.createElement('div');
        card.className = 'component-card';

        const dotColor = getComponentDotColor(item.component_type);
        const optionsHtml = (item.allowed_actions || [item.proposed_action]).map(act => {{
          const sel = (currentDecisions[item.id] === act) ? 'selected' : '';
          const label = act.replace('_', ' ').toUpperCase();
          return `<option value="${{act}}" ${{sel}}>${{label}}</option>`;
        }}).join('');

        const disabledAttr = (!item.is_safe_to_override) ? 'disabled title="Unsafe to override for the selected docking engine"' : '';

        card.innerHTML = `
          <div class="card-top">
            <span class="component-name">
              <span class="component-color-dot" style="background:${{dotColor}}"></span>
              ${{item.title}}
            </span>
            <select class="action-select" ${{disabledAttr}} onchange="onActionChange('${{item.id}}', this.value)">
              ${{optionsHtml}}
            </select>
          </div>
          <div class="rationale-text">
            <strong>Why:</strong> ${{item.scientific_rationale}}
          </div>
          <div class="consequence-tag">
            <strong>Docking Impact:</strong> ${{item.docking_consequences}}
          </div>
          ${{(!item.is_safe_to_override && item.safety_warning) ? `<div class="unsafe-warn">⚠ ${{item.safety_warning}}</div>` : ''}}
        `;
        list.appendChild(card);
      }});
    }}

    function onActionChange(id, value) {{
      currentDecisions[id] = value;
      apply3DStyles();
      syncDecisionsToServer();
    }}

    function syncDecisionsToServer() {{
      fetch('/update_decisions', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ decisions: currentDecisions }})
      }}).catch(e => console.error(e));
    }}

    // Master 3D Rendering & Styling Function
    function apply3DStyles() {{
      if (!viewer) return;
      try {{
        // Reset all atomic styles, labels, and surfaces
        viewer.setStyle({{}}, {{}});
        viewer.removeAllLabels();
        viewer.removeAllSurfaces();
        currentSurface = null;

        const allChains = (INSPECTION.chain_ids || ['A']);

        // 1. Render Protein Backbone / Selected Chains
        allChains.forEach(cid => {{
          const isSelected = selectedChains.includes(cid);
          const chainColor = CHAIN_COLORS[cid] || '#3b82f6';
          // Select only polymer protein atoms for the chain (exclude HETATMs like waters/ligands)
          const chainSel = {{ chain: cid, hetflag: false }};

          if (isSelected) {{
            let styleObj = {{}};

            // Determine coloring properties
            let cs = chainColor;
            if (currentColoring === 'chain') cs = chainColor;
            else if (currentColoring === 'ss') cs = 'ssJmol';
            else if (currentColoring === 'plddt') cs = {{ prop: 'b', gradient: 'roygb', min: 50, max: 95 }};
            else if (currentColoring === 'element') cs = 'default';

            const isHex = (typeof cs === 'string' && cs.startsWith('#'));

            if (currentRep === 'cartoon') {{
              styleObj = {{ cartoon: isHex ? {{ color: cs }} : {{ colorscheme: cs }} }};
            }} else if (currentRep === 'ballstick') {{
              styleObj = {{
                stick: isHex ? {{ color: cs, radius: 0.16 }} : {{ colorscheme: cs, radius: 0.16 }},
                sphere: isHex ? {{ color: cs, scale: 0.28 }} : {{ colorscheme: cs, scale: 0.28 }}
              }};
            }} else if (currentRep === 'stick') {{
              styleObj = {{ stick: isHex ? {{ color: cs, radius: 0.22 }} : {{ colorscheme: cs, radius: 0.22 }} }};
            }} else if (currentRep === 'ribbon') {{
              styleObj = {{ cartoon: isHex ? {{ color: cs, ribbon: true, thickness: 0.2 }} : {{ colorscheme: cs, ribbon: true, thickness: 0.2 }} }};
            }} else if (currentRep === 'sphere') {{
              styleObj = {{ sphere: isHex ? {{ color: cs }} : {{ colorscheme: cs }} }};
            }} else if (currentRep === 'surface') {{
              styleObj = {{ cartoon: {{ color: chainColor, opacity: 0.3 }} }};
              try {{
                viewer.addSurface($3Dmol.SurfaceType.VDW, {{ opacity: 0.65, color: chainColor }}, {{ chain: cid, hetflag: false }});
              }} catch (surfErr) {{
                console.warn("Surface render warning:", surfErr);
              }}
            }}

            viewer.setStyle(chainSel, styleObj);

          }} else {{
            // Unselected chain: Dim out to visually signal removal
            viewer.setStyle(chainSel, {{
              cartoon: {{ color: '#475569', opacity: 0.2 }}
            }});
          }}
        }});

        // 2. Waters Display Toggle
        if (showComponents.waters) {{
          const waterDec = currentDecisions['waters'] || 'remove';
          const waterSel = {{ resn: ['HOH', 'WAT', 'H2O', 'TIP3', 'DOD', 'TIP'] }};
          if (waterDec === 'keep') {{
            viewer.setStyle(waterSel, {{
              sphere: {{ color: '#ef4444', radius: 0.4, opacity: 1.0 }}
            }});
          }} else if (waterDec === 'remove_bulk') {{
            viewer.setStyle(waterSel, {{
              sphere: {{ color: '#f87171', radius: 0.28, opacity: 0.65 }}
            }});
          }} else {{
            // Faint dots for waters proposed to be removed
            viewer.setStyle(waterSel, {{
              sphere: {{ color: '#ef4444', radius: 0.18, opacity: 0.35 }}
            }});
          }}
        }}

        // 3. Metals Display Toggle
        if (showComponents.metals) {{
          const metalDec = currentDecisions['metals'] || 'keep';
          const opacity = (metalDec === 'remove') ? 0.35 : 1.0;
          const metalList = (INSPECTION.metals && INSPECTION.metals.length > 0)
            ? [...new Set(INSPECTION.metals)]
            : ['MG', 'ZN', 'MN', 'CA', 'FE', 'NA', 'K', 'CU', 'CO', 'NI', 'CD'];
          metalList.forEach(m => {{
            viewer.setStyle({{ resn: m }}, {{
              sphere: {{ color: '#f59e0b', radius: 1.15, opacity: opacity }}
            }});
            if (metalDec !== 'remove') {{
              viewer.addResLabels({{ resn: m }}, {{ fontSize: 11, fontColor: '#ffffff', backgroundColor: '#b45309' }});
            }}
          }});
        }}

        // 4. Cofactors Display Toggle
        if (showComponents.cofactors && INSPECTION.cofactors && INSPECTION.cofactors.length > 0) {{
          const cofDec = currentDecisions['cofactors'] || 'keep';
          const opacity = (cofDec === 'remove') ? 0.35 : 1.0;
          const cofList = [...new Set(INSPECTION.cofactors)];
          cofList.forEach(c => {{
            viewer.setStyle({{ resn: c }}, {{
              stick: {{ colorscheme: 'magentaCarbon', radius: 0.25, opacity: opacity }}
            }});
          }});
        }}

        // 5. Ligands Display Toggle
        if (showComponents.ligands) {{
          const ligDec = currentDecisions['ligands'] || 'remove';
          const opacity = (ligDec === 'remove') ? 0.35 : 1.0;
          const ligList = (INSPECTION.ligands && INSPECTION.ligands.length > 0)
            ? [...new Set(INSPECTION.ligands)]
            : [];
          if (ligList.length > 0) {{
            ligList.forEach(l => {{
              viewer.setStyle({{ resn: l }}, {{
                stick: {{ colorscheme: 'greenCarbon', radius: 0.28, opacity: opacity }}
              }});
            }});
          }} else {{
            viewer.setStyle({{ hetflag: true, not: {{ resn: ['HOH', 'WAT', 'H2O', 'TIP3', 'DOD', 'TIP', 'PEG', 'EDO', 'GOL', 'DMS', 'SO4', 'PO4', 'CIT', 'ACT', 'MG', 'ZN', 'CA', 'FE', 'MN', 'NA', 'K'] }} }}, {{
              stick: {{ colorscheme: 'greenCarbon', radius: 0.28, opacity: opacity }}
            }});
          }}
        }}

        // 6. Buffer additives (slate sticks)
        viewer.setStyle({{ resn: ['PEG', 'EDO', 'GOL', 'DMS', 'SO4', 'PO4', 'CIT', 'ACT'] }}, {{
          stick: {{ color: '#64748b', radius: 0.18, opacity: 0.35 }}
        }});

        viewer.render();
      }} catch (err) {{
        console.error("Error in apply3DStyles:", err);
      }}
    }}

    function confirmPreparation() {{
      const btn = document.getElementById('btn-confirm');
      btn.disabled = true;
      btn.innerText = 'Confirming...';

      fetch('/confirm_preparation', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ decisions: currentDecisions }})
      }})
      .then(r => r.json())
      .then(res => {{
        btn.style.display = 'none';
        document.getElementById('confirmed-msg').style.display = 'block';
      }})
      .catch(e => {{
        alert('Error confirming: ' + e);
        btn.disabled = false;
        btn.innerText = 'Confirm Preparation & Proceed to Docking';
      }});
    }}
  </script>
</body>
</html>
"""
    return html


def review_structure_preparation(
    pdb_path: str | Path,
    inspection_data: dict[str, Any],
    launch_gui: bool = True,
    auto_confirm: bool = False,
    timeout_sec: int = 120,
) -> dict[str, Any]:
    """
    Launch the structure preparation review workflow.

    Opens the interactive 3Dmol.js browser interface and renders a terminal summary table.
    Allows user to inspect components, scientific rationales, and docking consequences,
    override decisions, and confirm. Falls back to terminal interaction if headless.

    Args:
        pdb_path: Path to target PDB structure.
        inspection_data: Result from inspect_structure_components.
        launch_gui: Whether to launch local web GUI if display is available.
        auto_confirm: If True, uses proposed actions immediately without waiting.
        timeout_sec: Maximum time to wait for browser interaction before checking terminal.

    Returns:
        Reviewed decisions dictionary for preparation execution.
    """
    # 1. Build initial default decisions from proposals
    decisions: dict[str, Any] = {}
    for item in inspection_data.get("review_items", []):
        decisions[item["id"]] = item["proposed_action"]

    # All detected chains selected by default
    decisions["selected_chains"] = list(inspection_data.get("chain_ids", []))
    decisions["remove_water"] = (decisions.get("waters", "remove") in ("remove", "remove_bulk"))
    decisions["water_policy"] = decisions.get("waters", "remove_bulk" if inspection_data.get("workflow_mode") == "peptide" else "remove")
    decisions["keep_metals"] = (decisions.get("metals", "keep") == "keep")
    decisions["keep_cofactors"] = (decisions.get("cofactors", "keep") == "keep")
    decisions["keep_ligands"] = []  # ligands removed by default
    decisions["resolve_altlocs"] = (decisions.get("altlocs", "modify") == "modify")
    decisions["standardize_nonstandard"] = (decisions.get("nonstandard", "modify") == "modify")

    # 2. Print clear Terminal Summary Table
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()
        workflow_label = (
            "Protein–Small Molecule Docking"
            if inspection_data.get("workflow_mode") == "small_molecule"
            else "Protein–Peptide Docking"
        )

        title = f"[bold cyan]Structure Inspection & Preparation Review[/bold cyan] ({workflow_label})"
        console.print(Panel(
            f"[bold]Target Structure:[/bold] {inspection_data.get('file_name')}\n"
            f"[bold]Source:[/bold] {inspection_data.get('structure_source')}\n"
            f"[bold]Chains:[/bold] {', '.join(inspection_data.get('chain_ids', []))} | "
            f"[bold]Residues:[/bold] {inspection_data.get('total_residues')} | "
            f"[bold]Atoms:[/bold] {inspection_data.get('total_atoms')}",
            title=title,
            border_style="cyan"
        ))

        if inspection_data.get("is_alphafold"):
            af = inspection_data.get("alphafold_caveats", {})
            console.print(Panel(
                f"[yellow]{af.get('disclaimer', '')}[/yellow]\n"
                f"[bold yellow]Docking Guidance:[/bold yellow] [italic]{af.get('docking_suitability', '')}[/italic]",
                title="[bold yellow]⚠ AlphaFold Prediction Notice[/bold yellow]",
                border_style="yellow"
            ))

        table = Table(title="Proposed Preparation Actions & Scientific Rationales", border_style="blue")
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Count", justify="center", style="bold")
        table.add_column("Proposed Action", justify="center", style="green")
        table.add_column("Scientific Rationale & Docking Impact", style="white")

        for item in inspection_data.get("review_items", []):
            act_style = "bold red" if "remove" in item["proposed_action"] else "bold green"
            table.add_row(
                item["title"],
                str(item["count"]),
                f"[{act_style}]{item['proposed_action'].upper()}[/{act_style}]",
                f"{item['scientific_rationale']}\n[dim italic]Impact: {item['docking_consequences']}[/dim italic]",
            )

        console.print(table)

    except Exception:
        print(f"\n--- Structure Inspection: {inspection_data.get('file_name')} ---")
        for item in inspection_data.get("review_items", []):
            print(f" * {item['title']}: {item['proposed_action'].upper()} - {item['scientific_rationale']}")

    if auto_confirm or not sys.stdin.isatty():
        logger.info("Auto-confirming structure preparation with recommended defaults.")
        return decisions

    # 3. Launch Interactive 3D Web Review if requested and available
    server = None
    server_thread = None
    done_event = threading.Event()

    can_launch_web = launch_gui and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or sys.platform in ("darwin", "win32"))

    if can_launch_web:
        try:
            port = find_free_port(8770)
            html_content = generate_review_html(pdb_path, inspection_data, port)

            server = http.server.HTTPServer(("127.0.0.1", port), _StructureReviewHandler)
            server.html_content = html_content
            server.done_event = done_event
            server.user_decisions = decisions

            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()

            url = f"http://127.0.0.1:{port}"
            print(f"\n[+] Launched Interactive 3Dmol.js Structure Review at: {url}")
            print("    • View representations: Cartoon, Ball & Stick, Stick, Ribbon, Sphere, Surface")
            print("    • Multi-chain selection: Check/uncheck individual chains (Chain A, B, etc.) or Select All")
            print("    • Click 'Confirm Preparation' in browser or press [Enter] here to confirm defaults.")

            try:
                # Suppress browser stderr noise (e.g. chromium zygote errors)
                null_dev = open(os.devnull, "wb")
                old_stderr = os.dup(2)
                os.dup2(null_dev.fileno(), 2)
                try:
                    webbrowser.open(url)
                finally:
                    os.dup2(old_stderr, 2)
                    os.close(old_stderr)
                    null_dev.close()
            except Exception:
                pass

        except Exception as e:
            logger.warning(f"Could not start 3Dmol review web server: {e}")
            server = None

    # 4. Wait for confirmation via Browser POST or Terminal Enter
    print("\nPress [Enter] to confirm proposed preparation actions, or type 'o' to customize: ", end="", flush=True)

    while not done_event.is_set():
        if sys.stdin in select.select([sys.stdin], [], [], 0.5)[0]:
            try:
                line = sys.stdin.readline().strip()
                if line.lower() in ("o", "override", "custom"):
                    print("\nCustomizing preparation actions:")
                    # Option to select chains in terminal
                    avail_chains = inspection_data.get("chain_ids", [])
                    if len(avail_chains) > 1:
                        c_prompt = f"Enter chains to keep (comma-separated, available: {', '.join(avail_chains)}) [default: all]: "
                        c_val = input(c_prompt).strip()
                        if c_val:
                            chosen = [c.strip().upper() for c in re.split(r"[, ]+", c_val) if c.strip()]
                            valid_chosen = [c for c in chosen if c in avail_chains]
                            if valid_chosen:
                                decisions["selected_chains"] = valid_chosen
                                print(f"  ✓ Retaining chains: {', '.join(valid_chosen)}")

                    for item in inspection_data.get("review_items", []):
                        if item["id"] == "chains":
                            continue
                        if item.get("is_safe_to_override"):
                            allowed = item.get("allowed_actions", [item["proposed_action"]])
                            prompt = f"Action for {item['title']} ({'/'.join(allowed)}) [default: {item['proposed_action']}]: "
                            val = input(prompt).strip().lower()
                            if val in allowed:
                                decisions[item["id"]] = val
                done_event.set()
                break
            except (EOFError, KeyboardInterrupt):
                done_event.set()
                break

    # If browser submitted updated decisions, update our local dictionary
    if server and hasattr(server, "user_decisions") and server.user_decisions:
        for k, v in server.user_decisions.items():
            decisions[k] = v

    if server:
        try:
            server.shutdown()
            server.server_close()
        except Exception:
            pass

    # Re-synchronize high-level flags from confirmed item decisions
    decisions["remove_water"] = (decisions.get("waters", "remove") in ("remove", "remove_bulk"))
    decisions["water_policy"] = decisions.get("waters", "remove_bulk" if inspection_data.get("workflow_mode") == "peptide" else "remove")
    decisions["keep_metals"] = (decisions.get("metals", "keep") == "keep")
    decisions["keep_cofactors"] = (decisions.get("cofactors", "keep") == "keep")
    decisions["resolve_altlocs"] = (decisions.get("altlocs", "modify") == "modify")
    decisions["standardize_nonstandard"] = (decisions.get("nonstandard", "modify") == "modify")

    print("\n✓ Structure preparation review confirmed. Proceeding with transparent preparation...\n")
    return decisions
