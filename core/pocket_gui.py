"""
DrugLab Interactive Graphical Pocket & Docking Box Tuner.

Provides a lightweight, interactive 3D Web interface (powered by 3Dmol.js)
allowing users to visually inspect the protein binding cavity, dynamically
tune Center (X, Y, Z) and Size (X, Y, Z) coordinates with real-time feedback,
and confirm their selection by pressing Enter.
"""

import os
import sys
import io
import json
import time
import socket
import select
import threading
import webbrowser
import http.server
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from utils.logger import setup_logger

logger = setup_logger("pocket_gui", level="INFO")


def find_free_port(preferred: int = 8765) -> int:
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


class _PocketGUIHandler(http.server.BaseHTTPRequestHandler):
    """HTTP Request Handler for the Pocket Tuner GUI."""

    def log_message(self, format, *args):
        # Suppress noisy standard HTTP access logs
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
        if self.path == "/select_pocket":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(body)
                center = [float(x) for x in data.get("center", [0.0, 0.0, 0.0])]
                size = [float(x) for x in data.get("size", [30.0, 30.0, 30.0])]
                self.server.selected_coordinates = (center, size)
                if hasattr(self.server, "done_event"):
                    self.server.done_event.set()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        elif self.path == "/update_preview":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(body)
                center = [float(x) for x in data.get("center", [0.0, 0.0, 0.0])]
                size = [float(x) for x in data.get("size", [30.0, 30.0, 30.0])]
                self.server.selected_coordinates = (center, size)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def _build_pocket_gui_html(
    protein_pdb_text: str,
    pockets: list[dict],
    initial_center: list[float],
    initial_size: list[float],
    max_box_dimension: float = 30.0,
    coord_ranges: dict = None,
) -> str:
    """Generate the self-contained HTML for the 3D Pocket Tuner."""
    if coord_ranges is None:
        coord_ranges = {
            "min_x": initial_center[0] - 50.0,
            "max_x": initial_center[0] + 50.0,
            "min_y": initial_center[1] - 50.0,
            "max_y": initial_center[1] + 50.0,
            "min_z": initial_center[2] - 50.0,
            "max_z": initial_center[2] + 50.0,
        }

    pockets_json = json.dumps(pockets or [])
    protein_json = json.dumps(protein_pdb_text or "")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>DrugLab 3D Pocket &amp; Docking Box Tuner</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"></script>
  <style>
    *, *::before, *::after {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      border-radius: 0 !important;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #0f172a;
      color: #f1f5f9;
      height: 100vh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }}
    header {{
      background: #1e293b;
      border-bottom: 1px solid #334155;
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-shrink: 0;
    }}
    .header-title {{
      font-size: 18px;
      font-weight: 700;
      color: #38bdf8;
      letter-spacing: 0.5px;
    }}
    .header-sub {{
      font-size: 12px;
      color: #94a3b8;
    }}
    .main-layout {{
      display: flex;
      flex: 1;
      height: calc(100vh - 60px);
      overflow: hidden;
    }}
    #viewer_container {{
      flex: 1;
      position: relative;
      background: #090d16;
    }}
    #glviewer_canvas {{
      width: 100%;
      height: 100%;
    }}
    .sidebar {{
      width: 440px;
      background: #1e293b;
      border-left: 1px solid #334155;
      display: flex;
      flex-direction: column;
      overflow-y: auto;
      flex-shrink: 0;
      padding: 18px 20px;
      gap: 16px;
    }}
    .card {{
      background: #0f172a;
      border: 1px solid #334155;
      padding: 14px 16px;
    }}
    .card-title {{
      font-size: 12px;
      font-weight: 700;
      color: #94a3b8;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      margin-bottom: 8px;
    }}
    .coords-display {{
      background: #020617;
      border: 1px solid #1e293b;
      padding: 12px 14px;
      font-family: 'Consolas', 'Courier New', monospace;
      font-size: 13px;
      line-height: 1.6;
      color: #38bdf8;
      white-space: pre;
    }}
    .coords-display b {{
      color: #f8fafc;
    }}
    .btn {{
      background: #334155;
      color: #f8fafc;
      border: 1px solid #475569;
      padding: 7px 12px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
    }}
    .btn:hover {{
      background: #475569;
      border-color: #64748b;
    }}
    .btn-primary {{
      background: #0284c7 !important;
      border-color: #38bdf8 !important;
      color: #ffffff !important;
      font-size: 14px;
      font-weight: 700;
      padding: 12px 16px;
      width: 100%;
      letter-spacing: 0.5px;
    }}
    .btn-primary:hover {{
      background: #0369a1 !important;
    }}
    .btn-pocket {{
      background: #1e293b;
      border: 1px solid #334155;
      padding: 6px 10px;
      font-size: 11px;
      font-weight: 600;
      color: #cbd5e1;
    }}
    .btn-pocket:hover, .btn-pocket.active {{
      background: #0284c7;
      color: #ffffff;
      border-color: #38bdf8;
    }}
    .preset-group {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .ctrl-row {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 10px;
    }}
    .ctrl-label {{
      width: 60px;
      font-size: 12px;
      font-weight: 600;
      color: #cbd5e1;
    }}
    .ctrl-slider {{
      flex: 1;
      height: 6px;
      background: #334155;
      outline: none;
      -webkit-appearance: none;
      cursor: pointer;
    }}
    .ctrl-slider::-webkit-slider-thumb {{
      -webkit-appearance: none;
      width: 14px;
      height: 14px;
      background: #38bdf8;
      cursor: pointer;
    }}
    .ctrl-input {{
      width: 80px;
      background: #020617;
      border: 1px solid #334155;
      color: #f8fafc;
      font-family: 'Consolas', monospace;
      font-size: 12px;
      padding: 4px 6px;
      text-align: right;
    }}
    .step-btn-group {{
      display: flex;
      gap: 2px;
    }}
    .step-btn {{
      background: #1e293b;
      border: 1px solid #334155;
      color: #94a3b8;
      font-size: 10px;
      padding: 3px 6px;
      cursor: pointer;
    }}
    .step-btn:hover {{
      background: #334155;
      color: #ffffff;
    }}
    .viewer-overlay {{
      position: absolute;
      top: 14px;
      left: 14px;
      display: flex;
      gap: 8px;
      z-index: 10;
    }}
    .overlay-modal {{
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(15, 23, 42, 0.85);
      z-index: 1000;
      justify-content: center;
      align-items: center;
    }}
    .modal-box {{
      background: #1e293b;
      border: 1px solid #0284c7;
      padding: 24px 32px;
      max-width: 440px;
      text-align: center;
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <div class="header-title">DrugLab Pocket &amp; Docking Box Tuner</div>
      <div class="header-sub">Interactive 3D Cavity Inspection &amp; Grid Coordinate Selector</div>
    </div>
    <div style="font-size: 12px; color: #38bdf8; font-weight: 600;">
      Press [Enter] to Confirm Selection
    </div>
  </header>

  <div class="main-layout">
    <!-- 3Dmol WebGL Canvas -->
    <div id="viewer_container">
      <div class="viewer-overlay">
        <button class="btn" onclick="focusBox()" title="Focus camera on active pocket cavity and docking box">Focus Pocket</button>
        <button class="btn" onclick="zoomFullProtein()" title="View whole target protein">Full Protein</button>
        <button class="btn" id="btn_toggle_residues" onclick="togglePocketResidues()" title="Toggle display of pocket lining residues">Residues: ON</button>
        <button class="btn" onclick="resetToInitial()" title="Reset to Pocket 1 (Best Pocket)">Reset</button>
      </div>
      <div id="glviewer_canvas"></div>
    </div>

    <!-- Controls Sidebar -->
    <div class="sidebar">
      <!-- Quick Pocket Presets -->
      <div class="card">
        <div class="card-title">Detected Pocket Presets</div>
        <div id="preset_buttons" class="preset-group">
          <!-- Populated dynamically -->
        </div>
      </div>

      <!-- Live Coordinates Card (User Required Format) -->
      <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
          <div class="card-title" style="margin-bottom: 0;">Active Coordinates</div>
          <span id="pocket_badge" style="background: #0284c7; color: #ffffff; font-size: 10px; font-weight: 700; padding: 2px 8px; text-transform: uppercase; letter-spacing: 0.5px;">
            Pocket 1 (Best)
          </span>
        </div>
        <div id="live_coords_panel" class="coords-display">
    Center:
      X = {initial_center[0]:.3f}
      Y = {initial_center[1]:.3f}
      Z = {initial_center[2]:.3f}

    Size:
      X = {initial_size[0]:.1f}
      Y = {initial_size[1]:.1f}
      Z = {initial_size[2]:.1f}
        </div>
      </div>

      <!-- Center (X, Y, Z) Controls -->
      <div class="card">
        <div class="card-title">Box Center (Å)</div>
        <div class="ctrl-row">
          <span class="ctrl-label">Center X</span>
          <input type="range" id="slider_cx" class="ctrl-slider" min="{coord_ranges['min_x']}" max="{coord_ranges['max_x']}" step="0.1" value="{initial_center[0]}" oninput="syncFromSlider('cx')"/>
          <input type="number" id="input_cx" class="ctrl-input" step="0.1" value="{initial_center[0]:.3f}" onchange="syncFromInput('cx')"/>
          <div class="step-btn-group">
            <button type="button" class="step-btn" onclick="adjustValue('cx', -1.0)">-1</button>
            <button type="button" class="step-btn" onclick="adjustValue('cx', 1.0)">+1</button>
          </div>
        </div>

        <div class="ctrl-row">
          <span class="ctrl-label">Center Y</span>
          <input type="range" id="slider_cy" class="ctrl-slider" min="{coord_ranges['min_y']}" max="{coord_ranges['max_y']}" step="0.1" value="{initial_center[1]}" oninput="syncFromSlider('cy')"/>
          <input type="number" id="input_cy" class="ctrl-input" step="0.1" value="{initial_center[1]:.3f}" onchange="syncFromInput('cy')"/>
          <div class="step-btn-group">
            <button type="button" class="step-btn" onclick="adjustValue('cy', -1.0)">-1</button>
            <button type="button" class="step-btn" onclick="adjustValue('cy', 1.0)">+1</button>
          </div>
        </div>

        <div class="ctrl-row">
          <span class="ctrl-label">Center Z</span>
          <input type="range" id="slider_cz" class="ctrl-slider" min="{coord_ranges['min_z']}" max="{coord_ranges['max_z']}" step="0.1" value="{initial_center[2]}" oninput="syncFromSlider('cz')"/>
          <input type="number" id="input_cz" class="ctrl-input" step="0.1" value="{initial_center[2]:.3f}" onchange="syncFromInput('cz')"/>
          <div class="step-btn-group">
            <button type="button" class="step-btn" onclick="adjustValue('cz', -1.0)">-1</button>
            <button type="button" class="step-btn" onclick="adjustValue('cz', 1.0)">+1</button>
          </div>
        </div>
      </div>

      <!-- Size (X, Y, Z) Controls -->
      <div class="card">
        <div class="card-title">Box Size (Å) — Max {max_box_dimension:.0f} Å</div>
        <div class="ctrl-row">
          <span class="ctrl-label">Size X</span>
          <input type="range" id="slider_sx" class="ctrl-slider" min="5.0" max="45.0" step="0.5" value="{initial_size[0]}" oninput="syncFromSlider('sx')"/>
          <input type="number" id="input_sx" class="ctrl-input" step="0.5" min="5.0" max="50.0" value="{initial_size[0]:.1f}" onchange="syncFromInput('sx')"/>
          <div class="step-btn-group">
            <button type="button" class="step-btn" onclick="adjustValue('sx', -2.0)">-2</button>
            <button type="button" class="step-btn" onclick="adjustValue('sx', 2.0)">+2</button>
          </div>
        </div>

        <div class="ctrl-row">
          <span class="ctrl-label">Size Y</span>
          <input type="range" id="slider_sy" class="ctrl-slider" min="5.0" max="45.0" step="0.5" value="{initial_size[1]}" oninput="syncFromSlider('sy')"/>
          <input type="number" id="input_sy" class="ctrl-input" step="0.5" min="5.0" max="50.0" value="{initial_size[1]:.1f}" onchange="syncFromInput('sy')"/>
          <div class="step-btn-group">
            <button type="button" class="step-btn" onclick="adjustValue('sy', -2.0)">-2</button>
            <button type="button" class="step-btn" onclick="adjustValue('sy', 2.0)">+2</button>
          </div>
        </div>

        <div class="ctrl-row">
          <span class="ctrl-label">Size Z</span>
          <input type="range" id="slider_sz" class="ctrl-slider" min="5.0" max="45.0" step="0.5" value="{initial_size[2]}" oninput="syncFromSlider('sz')"/>
          <input type="number" id="input_sz" class="ctrl-input" step="0.5" min="5.0" max="50.0" value="{initial_size[2]:.1f}" onchange="syncFromInput('sz')"/>
          <div class="step-btn-group">
            <button type="button" class="step-btn" onclick="adjustValue('sz', -2.0)">-2</button>
            <button type="button" class="step-btn" onclick="adjustValue('sz', 2.0)">+2</button>
          </div>
        </div>
      </div>

      <!-- Submit / Confirm Button -->
      <div>
        <button type="button" id="btn_submit" class="btn btn-primary" onclick="submitCoordinates()">
          Select Coordinates (Enter)
        </button>
      </div>
    </div>
  </div>

  <!-- Confirmation Modal -->
  <div id="modal_confirmed" class="overlay-modal">
    <div class="modal-box">
      <h3 style="color:#38bdf8; margin-bottom: 12px; font-size: 18px;">Coordinates Selected</h3>
      <p style="color:#cbd5e1; font-size: 14px; margin-bottom: 16px;">
        Docking box saved successfully. Returning to the DrugLab pipeline in terminal...
      </p>
      <div id="modal_coords" class="coords-display" style="text-align: left; margin-bottom: 16px;"></div>
      <p style="font-size: 12px; color:#94a3b8;">You can now close this tab.</p>
    </div>
  </div>

  <script>
    let glviewer = null;
    let protModel = null;
    let showPocketResidues = true;
    const proteinData = {protein_json};
    const detectedPockets = {pockets_json};
    const initialCenter = [{initial_center[0]}, {initial_center[1]}, {initial_center[2]}];
    const initialSize = [{initial_size[0]}, {initial_size[1]}, {initial_size[2]}];

    let cx = initialCenter[0];
    let cy = initialCenter[1];
    let cz = initialCenter[2];
    let sx = initialSize[0];
    let sy = initialSize[1];
    let sz = initialSize[2];

    document.addEventListener("DOMContentLoaded", function() {{
      initViewer();
      initPresets();
      updateUI();
    }});

    window.addEventListener("resize", function() {{
      if (glviewer) {{
        glviewer.resize();
        glviewer.render();
      }}
    }});

    // Global Enter key listener
    window.addEventListener("keydown", function(e) {{
      if (e.key === "Enter") {{
        e.preventDefault();
        submitCoordinates();
      }}
    }});

    function getPocketSelection() {{
      return {{
        predicate: function(atom) {{
          let dx = atom.x - cx;
          let dy = atom.y - cy;
          let dz = atom.z - cz;
          let maxD = Math.max(sx, sy, sz) * 0.75 + 4.0;
          return (dx * dx + dy * dy + dz * dz) <= (maxD * maxD);
        }}
      }};
    }}

    function updateProteinStyles() {{
      if (!glviewer || !protModel) return;

      // Base: cartoon representation of entire protein
      glviewer.setStyle({{}}, {{ cartoon: {{ color: "spectrum", opacity: 0.82 }} }});

      // Highlight pocket lining residues in warm amber sticks
      if (showPocketResidues) {{
        let sel = getPocketSelection();
        let atoms = glviewer.getAtomsFromSel(sel);
        if (atoms && atoms.length > 0) {{
          glviewer.addStyle(sel, {{
            stick: {{ color: "#f59e0b", radius: 0.22, opacity: 0.95 }}
          }});
        }}
      }}
      glviewer.render();
    }}

    function togglePocketResidues() {{
      showPocketResidues = !showPocketResidues;
      let btn = document.getElementById("btn_toggle_residues");
      if (btn) {{
        btn.textContent = showPocketResidues ? "Residues: ON" : "Residues: OFF";
      }}
      updateProteinStyles();
    }}

    function initViewer() {{
      let container = document.getElementById("glviewer_canvas");
      if (!container) return;

      glviewer = $3Dmol.createViewer(container, {{ backgroundColor: "#090d16" }});

      if (proteinData && proteinData.length > 20) {{
        protModel = glviewer.addModel(proteinData, "pdb");
        updateProteinStyles();
      }}

      update3DBox();
      glviewer.zoomTo();
      glviewer.render();
    }}

    function initPresets() {{
      let container = document.getElementById("preset_buttons");
      if (!container) return;
      container.innerHTML = "";

      if (detectedPockets && detectedPockets.length > 0) {{
        detectedPockets.slice(0, 5).forEach(function(p, idx) {{
          let btn = document.createElement("button");
          btn.type = "button";
          btn.className = "btn btn-pocket" + (idx === 0 ? " active" : "");
          let scoreTxt = p.score ? (" · " + Math.round(p.score)) : "";
          let label = (idx === 0) ? ("Pocket 1 ★ Best" + scoreTxt) : ("Pocket " + (idx + 1) + scoreTxt);
          btn.textContent = label;
          btn.onclick = function() {{
            document.querySelectorAll(".btn-pocket").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            applyPocketPreset(p, idx);
          }};
          container.appendChild(btn);
        }});
      }} else {{
        let span = document.createElement("span");
        span.style.fontSize = "12px";
        span.style.color = "#94a3b8";
        span.textContent = "No automated pockets found. Using manual mode.";
        container.appendChild(span);
      }}
    }}

    function applyPocketPreset(p, idx) {{
      if (!p || !p.center || !p.size) return;
      cx = p.center[0];
      cy = p.center[1];
      cz = p.center[2];
      sx = p.size[0];
      sy = p.size[1];
      sz = p.size[2];

      document.getElementById("slider_cx").value = cx;
      document.getElementById("input_cx").value = cx.toFixed(3);
      document.getElementById("slider_cy").value = cy;
      document.getElementById("input_cy").value = cy.toFixed(3);
      document.getElementById("slider_cz").value = cz;
      document.getElementById("input_cz").value = cz.toFixed(3);

      document.getElementById("slider_sx").value = sx;
      document.getElementById("input_sx").value = sx.toFixed(1);
      document.getElementById("slider_sy").value = sy;
      document.getElementById("input_sy").value = sy.toFixed(1);
      document.getElementById("slider_sz").value = sz;
      document.getElementById("input_sz").value = sz.toFixed(1);

      let badge = document.getElementById("pocket_badge");
      if (badge) {{
        badge.textContent = (idx === 0) ? "Pocket 1 (Best)" : ("Pocket " + (idx + 1));
        badge.style.background = "#0284c7";
      }}

      updateUI();
      focusBox();
      notifyCoordinates();
    }}

    let previewDebounce = null;
    function notifyCoordinates() {{
      if (previewDebounce) clearTimeout(previewDebounce);
      previewDebounce = setTimeout(function() {{
        fetch("/update_preview", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{
            center: [parseFloat(cx.toFixed(3)), parseFloat(cy.toFixed(3)), parseFloat(cz.toFixed(3))],
            size: [parseFloat(sx.toFixed(1)), parseFloat(sy.toFixed(1)), parseFloat(sz.toFixed(1))]
          }})
        }}).catch(function() {{}});
      }}, 120);
    }}

    function syncFromSlider(prop) {{
      let val = parseFloat(document.getElementById("slider_" + prop).value);
      if (prop === "cx") cx = val;
      if (prop === "cy") cy = val;
      if (prop === "cz") cz = val;
      if (prop === "sx") sx = val;
      if (prop === "sy") sy = val;
      if (prop === "sz") sz = val;

      let input = document.getElementById("input_" + prop);
      if (input) {{
        input.value = (prop.startsWith("c") ? val.toFixed(3) : val.toFixed(1));
      }}

      let badge = document.getElementById("pocket_badge");
      if (badge) {{
        badge.textContent = "Custom Tuned";
        badge.style.background = "#475569";
      }}
      document.querySelectorAll(".btn-pocket").forEach(b => b.classList.remove("active"));

      updateUI();
      notifyCoordinates();
    }}

    function syncFromInput(prop) {{
      let val = parseFloat(document.getElementById("input_" + prop).value);
      if (isNaN(val)) return;
      if (prop === "cx") cx = val;
      if (prop === "cy") cy = val;
      if (prop === "cz") cz = val;
      if (prop === "sx") sx = Math.max(2.0, val);
      if (prop === "sy") sy = Math.max(2.0, val);
      if (prop === "sz") sz = Math.max(2.0, val);

      let slider = document.getElementById("slider_" + prop);
      if (slider) {{
        if (val < parseFloat(slider.min)) slider.min = Math.floor(val - 10);
        if (val > parseFloat(slider.max)) slider.max = Math.ceil(val + 10);
        slider.value = val;
      }}

      let badge = document.getElementById("pocket_badge");
      if (badge) {{
        badge.textContent = "Custom Tuned";
        badge.style.background = "#475569";
      }}
      document.querySelectorAll(".btn-pocket").forEach(b => b.classList.remove("active"));

      updateUI();
      notifyCoordinates();
    }}

    function adjustValue(prop, delta) {{
      let input = document.getElementById("input_" + prop);
      if (!input) return;
      let cur = parseFloat(input.value) || 0.0;
      let next = cur + delta;
      input.value = (prop.startsWith("c") ? next.toFixed(3) : Math.max(2.0, next).toFixed(1));
      syncFromInput(prop);
    }}

    function updateUI() {{
      // Update the live formatted coordinates panel
      let panel = document.getElementById("live_coords_panel");
      if (panel) {{
        panel.innerHTML =
"    Center:\\n" +
"      X = " + cx.toFixed(3) + "\\n" +
"      Y = " + cy.toFixed(3) + "\\n" +
"      Z = " + cz.toFixed(3) + "\\n\\n" +
"    Size:\\n" +
"      X = " + sx.toFixed(1) + "\\n" +
"      Y = " + sy.toFixed(1) + "\\n" +
"      Z = " + sz.toFixed(1);
      }}

      update3DBox();
      updateProteinStyles();
    }}

    function update3DBox() {{
      if (!glviewer) return;
      glviewer.removeAllShapes();

      // 1. Semi-transparent docking box volume
      glviewer.addBox({{
        center: {{ x: cx, y: cy, z: cz }},
        dimensions: {{ w: sx, h: sy, d: sz }},
        color: "#0284c7",
        opacity: 0.32,
        wireframe: false
      }});

      // 2. Wireframe border outline
      glviewer.addBox({{
        center: {{ x: cx, y: cy, z: cz }},
        dimensions: {{ w: sx, h: sy, d: sz }},
        color: "#38bdf8",
        wireframe: true
      }});

      // 3. Red center point
      glviewer.addSphere({{
        center: {{ x: cx, y: cy, z: cz }},
        radius: 1.1,
        color: "#ef4444"
      }});

      glviewer.render();
    }}

    function focusBox() {{
      if (!glviewer) return;
      let sel = getPocketSelection();
      let atoms = glviewer.getAtomsFromSel(sel);
      if (atoms && atoms.length > 0) {{
        glviewer.zoomTo(sel, 400);
      }} else {{
        glviewer.zoomTo({{}}, 400);
      }}
    }}

    function zoomFullProtein() {{
      if (!glviewer) return;
      glviewer.zoomTo({{}}, 400);
    }}

    function resetToInitial() {{
      cx = initialCenter[0];
      cy = initialCenter[1];
      cz = initialCenter[2];
      sx = initialSize[0];
      sy = initialSize[1];
      sz = initialSize[2];

      document.getElementById("slider_cx").value = cx;
      document.getElementById("input_cx").value = cx.toFixed(3);
      document.getElementById("slider_cy").value = cy;
      document.getElementById("input_cy").value = cy.toFixed(3);
      document.getElementById("slider_cz").value = cz;
      document.getElementById("input_cz").value = cz.toFixed(3);

      document.getElementById("slider_sx").value = sx;
      document.getElementById("input_sx").value = sx.toFixed(1);
      document.getElementById("slider_sy").value = sy;
      document.getElementById("input_sy").value = sy.toFixed(1);
      document.getElementById("slider_sz").value = sz;
      document.getElementById("input_sz").value = sz.toFixed(1);

      let badge = document.getElementById("pocket_badge");
      if (badge) {{
        badge.textContent = "Pocket 1 (Best)";
        badge.style.background = "#0284c7";
      }}
      document.querySelectorAll(".btn-pocket").forEach((b, idx) => {{
        if (idx === 0) b.classList.add("active");
        else b.classList.remove("active");
      }});

      updateUI();
      focusBox();
      notifyCoordinates();
    }}

    function submitCoordinates() {{
      let payload = {{
        center: [parseFloat(cx.toFixed(3)), parseFloat(cy.toFixed(3)), parseFloat(cz.toFixed(3))],
        size: [parseFloat(sx.toFixed(1)), parseFloat(sy.toFixed(1)), parseFloat(sz.toFixed(1))]
      }};

      let btn = document.getElementById("btn_submit");
      if (btn) {{
        btn.textContent = "Selecting Coordinates...";
        btn.disabled = true;
      }}

      fetch("/select_pocket", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify(payload)
      }}).then(response => {{
        if (response.ok) {{
          let modal = document.getElementById("modal_confirmed");
          let modalCoords = document.getElementById("modal_coords");
          if (modalCoords) {{
            modalCoords.textContent =
"Center: X=" + payload.center[0] + ", Y=" + payload.center[1] + ", Z=" + payload.center[2] + "\\n" +
"Size:   X=" + payload.size[0] + ", Y=" + payload.size[1] + ", Z=" + payload.size[2];
          }}
          if (modal) modal.style.display = "flex";
          setTimeout(function() {{
            window.close();
          }}, 1200);
        }} else {{
          alert("Could not select coordinates: server error");
          if (btn) {{
            btn.textContent = "Select Coordinates (Enter)";
            btn.disabled = false;
          }}
        }}
      }}).catch(err => {{
        alert("Failed to communicate with DrugLab server: " + err);
        if (btn) {{
          btn.textContent = "Select Coordinates (Enter)";
          btn.disabled = false;
        }}
      }});
    }}
  </script>
</body>
</html>
"""


def launch_pocket_gui(
    pockets: list[dict],
    doc=None,
    protein_pdb: str = None,
    max_box_dimension: float = 30.0,
    timeout: Optional[float] = None,
) -> Tuple[dict, str]:
    """
    Launch the interactive graphical interface (3D Web Browser GUI) for docking box selection.

    Allows the user to interactively inspect the protein cavity in 3D, drag/tune sliders
    for Center (X, Y, Z) and Size (X, Y, Z), see live formatted updates, and press Enter
    to confirm coordinates.

    Args:
        pockets: List of detected pocket dictionaries.
        doc: Gemmi Structure object (optional).
        protein_pdb: Path to the protein PDB file.
        max_box_dimension: Maximum allowed box dimension per axis (default 30.0).
        timeout: Optional timeout in seconds to wait before auto-selecting current coordinates.

    Returns:
        Tuple of (selected_pocket_dict, "graphical_gui").
    """
    # 1. Determine initial center and size
    if pockets and len(pockets) > 0:
        initial_center = [round(float(c), 3) for c in pockets[0].get("center", [0.0, 0.0, 0.0])]
        initial_size = [round(float(s), 1) for s in pockets[0].get("size", [30.0, 30.0, 30.0])]
        top_score = pockets[0].get("score", 0.0)
    else:
        initial_center = [0.0, 0.0, 0.0]
        initial_size = [30.0, 30.0, 30.0]
        top_score = 0.0

    # 2. Extract protein PDB string and coordinate ranges
    protein_pdb_text = ""
    coord_ranges = {
        "min_x": initial_center[0] - 60.0,
        "max_x": initial_center[0] + 60.0,
        "min_y": initial_center[1] - 60.0,
        "max_y": initial_center[1] + 60.0,
        "min_z": initial_center[2] - 60.0,
        "max_z": initial_center[2] + 60.0,
    }

    if protein_pdb and os.path.isfile(protein_pdb):
        try:
            with open(protein_pdb, "r", encoding="utf-8", errors="replace") as f:
                protein_pdb_text = f.read()
        except Exception:
            protein_pdb_text = ""

    if not protein_pdb_text and doc is not None:
        try:
            protein_pdb_text = doc.make_pdb_string()
        except Exception:
            pass

    if doc is not None:
        try:
            from core.pocket_detection import extract_protein_atoms
            atoms = extract_protein_atoms(doc)
            if len(atoms) > 0:
                mins = atoms.min(axis=0)
                maxs = atoms.max(axis=0)
                coord_ranges = {
                    "min_x": round(float(mins[0]) - 15.0, 1),
                    "max_x": round(float(maxs[0]) + 15.0, 1),
                    "min_y": round(float(mins[1]) - 15.0, 1),
                    "max_y": round(float(maxs[1]) + 15.0, 1),
                    "min_z": round(float(mins[2]) - 15.0, 1),
                    "max_z": round(float(maxs[2]) + 15.0, 1),
                }
                if initial_center == [0.0, 0.0, 0.0]:
                    initial_center = [round(float(c), 3) for c in atoms.mean(axis=0)]
        except Exception:
            pass

    # 3. Build HTML
    html_content = _build_pocket_gui_html(
        protein_pdb_text=protein_pdb_text,
        pockets=pockets,
        initial_center=initial_center,
        initial_size=initial_size,
        max_box_dimension=max_box_dimension,
        coord_ranges=coord_ranges,
    )

    # 4. Start local HTTP server
    port = find_free_port()
    server = http.server.HTTPServer(("127.0.0.1", port), _PocketGUIHandler)
    server.html_content = html_content
    server.selected_coordinates = (initial_center, initial_size)
    server.done_event = threading.Event()

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    url = f"http://127.0.0.1:{port}"

    print()
    print("=" * 66)
    print("  DrugLab Graphical Pocket & Docking Box Tuner")
    print("=" * 66)
    print(f"  URL: {url}")
    print()
    print("  Current Coordinates:")
    print(f"    Center: X = {initial_center[0]:.3f}, Y = {initial_center[1]:.3f}, Z = {initial_center[2]:.3f}")
    print(f"    Size:   X = {initial_size[0]:.1f},   Y = {initial_size[1]:.1f},   Z = {initial_size[2]:.1f}")
    print()
    print("  * Adjust coordinates in the 3D window and press [Enter] to confirm.")
    print("  * Or press [Enter] here in terminal to accept current coordinates.")
    print("=" * 66)
    print()

    # Try opening browser automatically
    try:
        webbrowser.open(url)
    except Exception:
        pass

    # 5. Wait for browser submission or terminal Enter
    terminal_choice_done = False
    start_wait = time.time()
    try:
        while not server.done_event.is_set():
            if timeout is not None and (time.time() - start_wait) >= timeout:
                break

            # Check if stdin has input (if running in interactive terminal)
            try:
                if hasattr(sys.stdin, "fileno"):
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.3)
                    if sys.stdin in rlist:
                        line = sys.stdin.readline()
                        # User pressed enter in terminal!
                        terminal_choice_done = True
                        break
                else:
                    time.sleep(0.3)
            except (io.UnsupportedOperation, OSError, ValueError, AttributeError):
                time.sleep(0.3)
    except (KeyboardInterrupt, EOFError):
        print("\nGUI cancelled. Using default pocket.")
    finally:
        server.shutdown()
        server.server_close()

    final_center, final_size = server.selected_coordinates

    # Clamp box size
    from core.pocket_detection import _clamp_box_size, extract_pocket_residues
    clamped_size, _ = _clamp_box_size(final_size, max_box_dimension, warn_large_boxes=False)

    residues = []
    if doc is not None:
        try:
            residues = extract_pocket_residues(doc, np.array(final_center), radius=12.0)
        except Exception:
            residues = pockets[0].get("residues", []) if pockets else []

    selected_pocket = {
        "pocket_id": -1,
        "center": [round(float(c), 3) for c in final_center],
        "size": clamped_size,
        "volume": float(round(np.prod(clamped_size), 2)),
        "residues": residues,
        "score": top_score,
    }

    print()
    print("Confirmed Coordinates from Graphical Interface:")
    print("  Center:")
    print(f"    X = {selected_pocket['center'][0]:.3f}")
    print(f"    Y = {selected_pocket['center'][1]:.3f}")
    print(f"    Z = {selected_pocket['center'][2]:.3f}")
    print("  Size:")
    print(f"    X = {selected_pocket['size'][0]:.1f}")
    print(f"    Y = {selected_pocket['size'][1]:.1f}")
    print(f"    Z = {selected_pocket['size'][2]:.1f}")
    print()

    return selected_pocket, "graphical_gui"
