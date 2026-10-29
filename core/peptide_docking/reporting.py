"""
Interactive HTML Report Generator for DrugLab 2.0 Peptide Docking.
Reuses the visual design, KPI cards, and 3Dmol.js viewer infrastructure from DrugLab.
"""

from __future__ import annotations

import os
import json
import html
from pathlib import Path
from datetime import datetime
from typing import Any
import pandas as pd

from utils.io import ensure_dir
from utils.logger import setup_logger

logger = setup_logger("peptide_reporting", level="INFO")


def generate_peptide_html_report(
    output_dir: str | Path,
    manifest: dict[str, Any],
    candidates_df: pd.DataFrame | None = None,
    poses_df: pd.DataFrame | None = None,
    filtering_df: pd.DataFrame | None = None,
    admet_df: pd.DataFrame | None = None,
    leads_df: pd.DataFrame | None = None,
    output_filename: str = "report.html",
) -> str:
    """
    Generate a standalone interactive HTML report for protein–peptide docking.
    """
    out_dir = Path(output_dir)
    report_path = out_dir / output_filename

    # Read dataframes from disk if not supplied
    if candidates_df is None and (out_dir / "docking_results.tsv").is_file():
        candidates_df = pd.read_csv(out_dir / "docking_results.tsv", sep="\t")
    if poses_df is None and (out_dir / "pose_metrics.tsv").is_file():
        poses_df = pd.read_csv(out_dir / "pose_metrics.tsv", sep="\t")
    if filtering_df is None and (out_dir / "peptide_filtering.tsv").is_file():
        filtering_df = pd.read_csv(out_dir / "peptide_filtering.tsv", sep="\t")
    if admet_df is None and (out_dir / "admet_predictions.tsv").is_file():
        admet_df = pd.read_csv(out_dir / "admet_predictions.tsv", sep="\t")
    if leads_df is None and (out_dir / "lead_peptides.csv").is_file():
        try:
            leads_df = pd.read_csv(out_dir / "lead_peptides.csv")
        except Exception:
            pass

    # KPIs
    total_peptides = manifest.get("total_peptides", 0)
    passed_filter = manifest.get("passed_filtering", 0)
    docked_count = manifest.get("completed_docking", 0)
    warn_count = manifest.get("length_warnings_count", 0)
    best_score = "N/A"
    best_cand = "N/A"

    if candidates_df is not None and not candidates_df.empty:
        comp = candidates_df[candidates_df["status"] == "completed"]
        if not comp.empty and "best_lightdock_score" in comp.columns:
            top_row = comp.iloc[0]
            val = top_row["best_lightdock_score"]
            if pd.notna(val):
                best_score = f"{float(val):.2f}"
            best_cand = str(top_row.get("candidate_id", "N/A"))

    rec_pdb_path = manifest.get("prepared_receptor_path") or manifest.get("original_receptor_path")
    rec_name = Path(rec_pdb_path).name if rec_pdb_path else "Receptor Target"

    filter_sub = f"{passed_filter} passed"
    if warn_count > 0:
        filter_sub += f" ({warn_count} with length warnings)"
    else:
        filter_sub += f" ({total_peptides - passed_filter} filtered)"

    admet_provider_name = manifest.get("admet_provider", "local")
    prov_title = "PEPLife 2.0" if "peplife" in str(admet_provider_name).lower() else "Local Peptide ADMET Engine"

    # Prepare pose data for 3Dmol.js viewer
    # Embed up to 10 best poses as inline PDB strings
    poses_for_viewer: list[dict[str, Any]] = []
    if poses_df is not None and not poses_df.empty:
        for _, row in poses_df.head(10).iterrows():
            pdb_path = str(row.get("pose_pdb_path", ""))
            if pdb_path and os.path.isfile(pdb_path):
                try:
                    with open(pdb_path, "r", encoding="utf-8") as pf:
                        pdb_content = pf.read()
                    poses_for_viewer.append({
                        "label": f"{row.get('candidate_id')} (Pose {row.get('pose_rank')}, Score: {row.get('lightdock_score')})",
                        "candidate_id": str(row.get("candidate_id")),
                        "rank": int(row.get("pose_rank", 1)),
                        "score": row.get("lightdock_score"),
                        "pdb_data": pdb_content,
                    })
                except Exception as e:
                    logger.warning(f"Failed reading pose file {pdb_path}: {e}")

    # Build HTML sections
    kpi_cards = f"""
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Receptor Target</div>
        <div class="kpi-val" style="font-size: 18px;">{html.escape(rec_name)}</div>
        <div class="kpi-sub">Chains: {html.escape(str(manifest.get('receptor_chains', [])))}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Peptides Screened</div>
        <div class="kpi-val">{total_peptides}</div>
        <div class="kpi-sub">{filter_sub}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Successfully Docked</div>
        <div class="kpi-val">{docked_count}</div>
        <div class="kpi-sub">Engine: LightDock ({html.escape(manifest.get('lightdock_version', '0.9.4'))})</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Top LightDock Score</div>
        <div class="kpi-val" style="color: var(--primary);">{best_score}</div>
        <div class="kpi-sub">Lead Candidate: {html.escape(best_cand)}</div>
      </div>
    </div>
    """

    # Table of Docking Results
    docking_table_rows = ""
    if candidates_df is not None and not candidates_df.empty:
        for _, r in candidates_df.iterrows():
            cid = html.escape(str(r.get("candidate_id", "")))
            seq = html.escape(str(r.get("sequence", "")))
            length = r.get("length", "")
            score = f"{float(r['best_lightdock_score']):.2f}" if pd.notna(r.get("best_lightdock_score")) else "-"
            sc_func = html.escape(str(r.get("scoring_function", "-")))
            contacts = r.get("contacts", "-")
            hbonds = r.get("hbonds", "-")
            clashes = r.get("clashes", "-")
            status = str(r.get("status", ""))
            badge_class = "badge-success" if status == "completed" else "badge-warning"

            docking_table_rows += f"""
            <tr>
              <td><strong>{cid}</strong></td>
              <td style="font-family: monospace; font-size: 12px;">{seq}</td>
              <td>{length}</td>
              <td style="font-weight: 700; color: #2b6cb0;">{score}</td>
              <td>{sc_func}</td>
              <td>{contacts}</td>
              <td>{hbonds}</td>
              <td>{clashes}</td>
              <td><span class="badge {badge_class}">{status}</span></td>
            </tr>
            """
    else:
        docking_table_rows = "<tr><td colspan='9' style='text-align:center;'>No docking results recorded.</td></tr>"

    # Tables for ADMET Descriptors, Permeability, and Toxicity
    admet_descriptors_rows = ""
    admet_permeability_rows = ""
    admet_toxicity_rows = ""

    if admet_df is not None and not admet_df.empty:
        for _, r in admet_df.iterrows():
            cid = html.escape(str(r.get("candidate_id", "")))
            seq = html.escape(str(r.get("sequence", "")))
            mw = f"{float(r['molecular_weight_da']):.1f}" if pd.notna(r.get("molecular_weight_da")) and str(r.get("molecular_weight_da")) != "N/A" else "-"
            logp_val = str(r.get("logp", "N/A"))
            logp_str = f"{float(logp_val):.2f}" if logp_val != "N/A" and pd.notna(r.get("logp")) else "N/A"
            hbd_str = html.escape(str(r.get("hbd", "N/A")))
            hba_str = html.escape(str(r.get("hba", "N/A")))
            tpsa_val = str(r.get("tpsa_angstrom2", "N/A"))
            tpsa_str = f"{float(tpsa_val):.1f}" if tpsa_val != "N/A" and pd.notna(r.get("tpsa_angstrom2")) else "N/A"
            charge = f"{float(r['net_charge_ph74']):+.2f}" if pd.notna(r.get("net_charge_ph74")) else "-"
            pi = f"{float(r['isoelectric_point_pi']):.2f}" if pd.notna(r.get("isoelectric_point_pi")) else "-"
            boman = f"{float(r['boman_index_kcal_mol']):.2f}" if pd.notna(r.get("boman_index_kcal_mol")) else "-"

            admet_descriptors_rows += f"""
            <tr>
              <td><strong>{cid}</strong></td>
              <td style="font-family: monospace; font-size: 12px;">{seq}</td>
              <td>{mw}</td>
              <td>{logp_str}</td>
              <td>{hbd_str}</td>
              <td>{hba_str}</td>
              <td>{tpsa_str}</td>
              <td>{charge}</td>
              <td>{pi}</td>
              <td>{boman}</td>
            </tr>
            """

            # Permeability & MACCS
            hl = html.escape(str(r.get("half_life", "-")))
            caco2_lin = html.escape(str(r.get("caco2_linear_papp", "N/A")))
            caco2_cyc = html.escape(str(r.get("caco2_cyclic_papp", "N/A")))
            caco2_all = html.escape(str(r.get("caco2_all_peptide", "N/A")))
            pampa = html.escape(str(r.get("pampa_cyclic_papp", "N/A")))
            rrck = html.escape(str(r.get("rrck_cyclic_papp", "N/A")))
            bbb = html.escape(str(r.get("bbb_penetration", "N/A")))
            bbb_badge = "badge-success" if "BBB+" in bbb else "badge-neutral"

            ref_match = str(r.get("maccs_ref_match", "N/A"))
            sim_val = r.get("maccs_similarity")
            if ref_match != "N/A" and pd.notna(sim_val) and str(sim_val) != "N/A":
                maccs_display = f"{html.escape(ref_match)} (Tanimoto: {float(sim_val):.3f})"
            else:
                maccs_display = "N/A"

            admet_permeability_rows += f"""
            <tr>
              <td><strong>{cid}</strong></td>
              <td style="font-weight: 600; color: #2c5282;">{hl}</td>
              <td>{caco2_lin}</td>
              <td>{caco2_cyc}</td>
              <td><span class="badge {'badge-success' if 'Moderate' in caco2_all or 'High' in caco2_all else 'badge-neutral'}">{caco2_all}</span></td>
              <td>{pampa}</td>
              <td>{rrck}</td>
              <td><span class="badge {bbb_badge}">{bbb}</span></td>
              <td style="font-size: 12px;">{maccs_display}</td>
            </tr>
            """

            # Toxicity Risk
            tox_risk = str(r.get("peptide_toxicity_risk", "toxicity undetermined"))
            if "High" in tox_risk:
                tox_badge = '<span class="badge badge-danger">High Risk (Hemolytic)</span>'
            elif "Moderate" in tox_risk:
                tox_badge = '<span class="badge badge-warning">Moderate Liability</span>'
            elif "Low" in tox_risk:
                tox_badge = '<span class="badge badge-success">Low Risk (Safe)</span>'
            else:
                tox_badge = '<span class="badge badge-neutral">Toxicity Undetermined</span>'

            tox_notes = html.escape(str(r.get("toxicity_notes", "-")))
            tox_src = html.escape(str(r.get("toxicity_source", "Undetermined")))

            admet_toxicity_rows += f"""
            <tr>
              <td><strong>{cid}</strong></td>
              <td style="font-family: monospace; font-size: 12px;">{seq}</td>
              <td>{tox_badge}</td>
              <td style="font-size: 12px; color: #4a5568;">{tox_notes}</td>
              <td style="font-size: 12px; color: #2b6cb0;">{tox_src}</td>
            </tr>
            """
    else:
        admet_descriptors_rows = "<tr><td colspan='10' style='text-align:center;'>No descriptors recorded.</td></tr>"
        admet_permeability_rows = "<tr><td colspan='9' style='text-align:center;'>No permeability predictions recorded.</td></tr>"
        admet_toxicity_rows = "<tr><td colspan='5' style='text-align:center;'>No toxicity assessment recorded.</td></tr>"

    # Filtering Table
    filtering_rows = ""
    if filtering_df is not None and not filtering_df.empty:
        for _, r in filtering_df.iterrows():
            cid = html.escape(str(r.get("candidate_id", "")))
            seq = html.escape(str(r.get("sequence", "")))
            length = r.get("length", "")
            mw = f"{float(r['molecular_weight_da']):.1f}" if pd.notna(r.get("molecular_weight_da")) else "-"
            st = str(r.get("status", ""))
            reason = html.escape(str(r.get("filter_reason", "-")))
            if st == "pass":
                badge_class = "badge-success"
            elif "warn" in st.lower():
                badge_class = "badge-warning"
            else:
                badge_class = "badge-danger"

            filtering_rows += f"""
            <tr>
              <td><strong>{cid}</strong></td>
              <td style="font-family: monospace; font-size: 12px;">{seq}</td>
              <td>{length}</td>
              <td>{mw}</td>
              <td><span class="badge {badge_class}">{st}</span></td>
              <td style="color: #718096; font-size: 13px;">{reason}</td>
            </tr>
            """

    # Top Ranked Lead Candidates Table
    lead_table_section = ""
    if leads_df is not None and not leads_df.empty:
        lead_rows = ""
        for _, r in leads_df.iterrows():
            rk = r.get("rank", "-")
            cid = html.escape(str(r.get("candidate_id", "")))
            seq = html.escape(str(r.get("sequence", "")))
            cscore = f"{float(r['composite_score']):.2f}" if pd.notna(r.get("composite_score")) else "-"
            lscore = f"{float(r['lightdock_score']):.2f}" if pd.notna(r.get("lightdock_score")) else "-"
            contacts = r.get("contacts", "-")
            hbonds = r.get("hbonds", "-")
            clashes = r.get("clashes", "-")
            hrisk = html.escape(str(r.get("hemolytic_risk", "-")))
            notes = html.escape(str(r.get("selection_notes", "-")))

            badge_c = "badge-success" if "Low" in hrisk else ("badge-danger" if "High" in hrisk else "badge-warning")
            highlight_bg = "background: #f0fff4;" if str(rk) == "1" else ""

            lead_rows += f"""
            <tr style="{highlight_bg}">
              <td style="font-weight: 700; text-align: center;">#{rk}</td>
              <td><strong>{cid}</strong></td>
              <td style="font-family: monospace; font-size: 12px;">{seq}</td>
              <td style="font-weight: 700; color: #2b6cb0;">{cscore}</td>
              <td>{lscore}</td>
              <td>{contacts}</td>
              <td>{hbonds}</td>
              <td>{clashes}</td>
              <td><span class="badge {badge_c}">{hrisk}</span></td>
              <td style="font-size: 12px; color: #4a5568;">{notes}</td>
            </tr>
            """

        lead_table_section = f"""
  <!-- Top Ranked Lead Candidates -->
  <div class="section-card" style="border-left: 4px solid #38a169;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
      <h3>🎯 Top Lead Candidates (Multi-Objective Selection)</h3>
      <span class="badge badge-success">Ranked & Selected</span>
    </div>
    <p style="color: #718096; font-size: 13px; margin-bottom: 16px;">
      Peptide candidates ranked by integrated composite score balancing LightDock binding score, interface contacts (&lt;4.5Å), hydrogen bonds (&lt;3.5Å), steric clash penalties, and safety/toxicity profile.
    </p>
    <div style="overflow-x: auto;">
      <table>
        <thead>
          <tr>
            <th style="width: 50px;">Rank</th>
            <th>Candidate ID</th>
            <th>Peptide Sequence</th>
            <th>Composite Score</th>
            <th>LightDock Score</th>
            <th>Contacts</th>
            <th>H-Bonds</th>
            <th>Clashes</th>
            <th>Toxicity Risk</th>
            <th>Selection Rationales</th>
          </tr>
        </thead>
        <tbody>
          {lead_rows}
        </tbody>
      </table>
    </div>
  </div>
        """

    # 3D Viewer JSON data
    viewer_json = json.dumps(poses_for_viewer)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{html.escape(manifest.get('title', 'DrugLab Peptide Docking Report'))}</title>
  <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    :root {{
      --primary: #2b6cb0;
      --primary-light: #ebf8ff;
      --success: #2f855a;
      --warning: #c05621;
      --danger: #c53030;
      --bg: #f7fafc;
      --card-bg: #ffffff;
      --text: #2d3748;
      --border: #e2e8f0;
    }}
    *, *::before, *::after {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
    }}
    .container {{
      max-width: 1320px;
      margin: 0 auto;
    }}
    header {{
      background: var(--card-bg);
      padding: 24px 32px;
      border: 1px solid var(--border);
      box-shadow: 0 1px 3px rgba(0,0,0,0.06);
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }}
    h1 {{ font-size: 24px; font-weight: 700; color: #1a202c; }}
    h2 {{ font-size: 20px; font-weight: 600; margin: 28px 0 14px 0; color: #2d3748; }}
    .badge {{
      display: inline-block;
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      border: 1px solid transparent;
    }}
    .badge-success {{ background: #c6f6d5; color: #22543d; border-color: #9ae6b4; }}
    .badge-warning {{ background: #feebc8; color: #744210; border-color: #fbd38d; }}
    .badge-danger  {{ background: #fed7d7; color: #742a2a; border-color: #feb2b2; }}
    .badge-neutral {{ background: #edf2f7; color: #4a5568; border-color: #cbd5e0; }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .kpi-card {{
      background: var(--card-bg);
      padding: 20px;
      border: 1px solid var(--border);
      box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    .kpi-label {{
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: #718096;
      font-weight: 600;
    }}
    .kpi-val {{
      font-size: 26px;
      font-weight: 700;
      color: #1a202c;
      margin: 8px 0;
    }}
    .kpi-sub {{
      font-size: 13px;
      color: #718096;
    }}
    .section-card {{
      background: var(--card-bg);
      padding: 24px;
      border: 1px solid var(--border);
      box-shadow: 0 1px 3px rgba(0,0,0,0.04);
      margin-bottom: 24px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 12px 14px;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }}
    th {{
      background: #edf2f7;
      color: #4a5568;
      font-weight: 600;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    tr:hover td {{
      background: #f7fafc;
    }}
    .viewer-container {{
      height: 540px;
      width: 100%;
      position: relative;
      background: #1a202c;
      border: 1px solid var(--border);
      margin-top: 14px;
    }}
    .btn {{
      padding: 8px 14px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      border: 1px solid var(--border);
      background: #edf2f7;
      color: #2d3748;
      margin-right: 6px;
    }}
    .btn:hover {{ background: #e2e8f0; }}
    .btn-primary {{
      background: var(--primary);
      color: #ffffff;
      border-color: var(--primary);
    }}
    .btn-primary:hover {{ background: #2b5f9e; }}
  </style>
</head>
<body>
<div class="container">
  <header>
    <div>
      <h1>🧬 DrugLab 2.0 — Protein–Peptide Docking Report</h1>
      <p style="color: #718096; font-size: 14px; margin-top: 4px;">
        Pipeline: LightDock Engine | Run ID: {html.escape(str(manifest.get('run_id', 'N/A')))} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
      </p>
    </div>
    <div>
      <span class="badge badge-success">Workflow: Peptide</span>
    </div>
  </header>

  {kpi_cards}

  <!-- Interactive 3D Molecular Complex Viewer -->
  <div class="section-card">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
      <div>
        <h3>Interactive 3D Protein–Peptide Complex</h3>
        <p style="color: #718096; font-size: 13px;">View docked peptide poses on the receptor surface (3Dmol.js)</p>
      </div>
      <div>
        <label for="poseSelect" style="font-size: 13px; font-weight: 600; margin-right: 6px;">Select Pose:</label>
        <select id="poseSelect" class="btn" style="padding: 6px 12px;">
        </select>
        <button class="btn btn-primary" onclick="focusPeptide()" title="Zoom and center camera directly on the docked peptide">🎯 Focus Peptide</button>
        <button class="btn" onclick="resetViewerCamera()">🔄 Reset View</button>
      </div>
    </div>

    <!-- Viewer Toolbar Controls -->
    <div style="display: flex; flex-wrap: wrap; gap: 16px; margin-top: 14px; padding: 12px 16px; background: #f7fafc; border: 1px solid var(--border); border-radius: 4px; font-size: 13px; align-items: center;">
      <div>
        <strong>Protein View:</strong>
        <select id="proteinStyle" class="btn" style="padding: 4px 8px;" onchange="updateComplexStyles()">
          <option value="cartoon">Cartoon</option>
          <option value="surface">Surface</option>
          <option value="ribbon">Ribbon</option>
          <option value="stick">Sticks</option>
          <option value="line">Wireframe</option>
        </select>
        <select id="proteinColor" class="btn" style="padding: 4px 8px;" onchange="updateComplexStyles()">
          <option value="cyan">Cyan</option>
          <option value="spectrum">Spectrum</option>
          <option value="ss">Sec. Structure</option>
          <option value="b">B-Factor</option>
        </select>
      </div>
      <div>
        <strong>Peptide View:</strong>
        <select id="peptideStyle" class="btn" style="padding: 4px 8px;" onchange="updateComplexStyles()">
          <option value="stick">Sticks</option>
          <option value="ballAndStick">Ball & Stick</option>
          <option value="cartoon">Cartoon</option>
          <option value="sphere">Spacefill</option>
          <option value="surface">Surface</option>
        </select>
        <select id="peptideColor" class="btn" style="padding: 4px 8px;" onchange="updateComplexStyles()">
          <option value="orangeCarbon">Orange Carbon</option>
          <option value="greenCarbon">Green Carbon</option>
          <option value="polarity">Residue Polarity</option>
          <option value="spectrum">Spectrum</option>
        </select>
      </div>
      <div style="display: flex; gap: 12px; align-items: center; margin-left: auto;">
        <label><input type="checkbox" id="chkProtein" checked onchange="updateComplexStyles()"> Show Protein</label>
        <label><input type="checkbox" id="chkPeptide" checked onchange="updateComplexStyles()"> Show Peptide</label>
        <label><input type="checkbox" id="chkContacts" onchange="toggleContacts(this.checked)"> Interface (<4.5Å)</label>
        <label><input type="checkbox" id="chkHbonds" onchange="toggleHbonds(this.checked)"> H-Bonds</label>
      </div>
    </div>

    <!-- 3Dmol Canvas Container with Interactive HUD -->
    <div style="position: relative; width: 100%; height: 540px; margin-top: 14px; border: 1px solid var(--border); background: #1a202c; overflow: hidden;">
      <div id="complex_3d_viewer" style="width: 100%; height: 100%;"></div>
      <div id="interaction_hud" style="display: none; position: absolute; bottom: 12px; left: 16px; right: 16px; background: rgba(15, 23, 42, 0.92); backdrop-filter: blur(6px); border: 1px solid #334155; padding: 10px 16px; border-radius: 4px; color: #f8fafc; font-size: 13px; pointer-events: none; z-index: 50; transition: all 0.15s ease; box-shadow: 0 4px 12px rgba(0,0,0,0.35);">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
          <div id="hud_main" style="display: flex; align-items: center; gap: 10px;"></div>
          <div id="hud_badge"></div>
        </div>
      </div>
    </div>
  </div>

  {lead_table_section}

  <!-- Docking Results Table -->
  <div class="section-card">
    <h3>Ranked Docking Candidates</h3>
    <p style="color: #718096; font-size: 13px; margin-bottom: 16px;">
      Native LightDock scoring potentials and supplementary structural metrics (contacts, hydrogen bonds, steric clashes).
    </p>
    <div style="overflow-x: auto;">
      <table>
        <thead>
          <tr>
            <th>Candidate ID</th>
            <th>Peptide Sequence</th>
            <th>Length</th>
            <th>LightDock Score</th>
            <th>Scoring Function</th>
            <th>Contacts (<4.5Å)</th>
            <th>H-Bonds (<3.5Å)</th>
            <th>Clashes (<2.0Å)</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {docking_table_rows}
        </tbody>
      </table>
    </div>
  </div>

  <!-- ADMET, Permeability & Safety Section -->
  <div class="section-card">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
      <h3>Peptide ADMET, Permeability & Safety Profile</h3>
      <span class="badge badge-success">Provider: {prov_title}</span>
    </div>
    <p style="color: #718096; font-size: 13px; margin-bottom: 16px;">
      Comprehensive peptide profiling incorporating exact RDKit & BioPython calculated descriptors, MACCS reference library matching, multi-barrier permeability models (Caco-2, PAMPA, RRCK), blood-brain barrier transport, and peptide-specific hemolytic toxicity risk.
    </p>

    <!-- Table 1: Calculated Descriptors -->
    <h4 style="margin: 16px 0 8px; color: #2d3748;">1. Calculated Physicochemical Descriptors <span style="font-size: 12px; font-weight: normal; color: #718096;">(RDKit & BioPython ProtParam)</span></h4>
    <div style="overflow-x: auto; margin-bottom: 24px;">
      <table>
        <thead>
          <tr>
            <th>Candidate ID</th>
            <th>Sequence</th>
            <th>MW (Da)</th>
            <th>LogP</th>
            <th>HBD</th>
            <th>HBA</th>
            <th>TPSA (Å²)</th>
            <th>Net Charge (pH 7.4)</th>
            <th>pI</th>
            <th>Boman Index (kcal/mol)</th>
          </tr>
        </thead>
        <tbody>
          {admet_descriptors_rows}
        </tbody>
      </table>
    </div>

    <!-- Table 2: Permeability, BBB & Reference Similarity -->
    <h4 style="margin: 16px 0 8px; color: #2d3748;">2. Pharmacokinetics, Barrier Permeability & Reference Similarity <span style="font-size: 12px; font-weight: normal; color: #718096;">(Literature-Grounded Models)</span></h4>
    <div style="overflow-x: auto; margin-bottom: 24px;">
      <table>
        <thead>
          <tr>
            <th>Candidate ID</th>
            <th>Half-Life</th>
            <th>Caco-2 (Linear)</th>
            <th>Caco-2 (Cyclic)</th>
            <th>Caco-2 (All-Peptide)</th>
            <th>PAMPA (Cyclic)</th>
            <th>RRCK (Cyclic)</th>
            <th>BBB Penetration</th>
            <th>Top Reference Peptide Match (MACCS Tanimoto)</th>
          </tr>
        </thead>
        <tbody>
          {admet_permeability_rows}
        </tbody>
      </table>
    </div>

    <!-- Table 3: Peptide Toxicity Assessment -->
    <h4 style="margin: 16px 0 8px; color: #2d3748;">3. Peptide-Specific Toxicity Risk Assessment <span style="font-size: 12px; font-weight: normal; color: #718096;">(Hemolytic & Membrane Disruption Models)</span></h4>
    <div style="overflow-x: auto; margin-bottom: 16px;">
      <table>
        <thead>
          <tr>
            <th>Candidate ID</th>
            <th>Sequence</th>
            <th>Hemolytic Toxicity Risk</th>
            <th>Biophysical Driving Factors / Assessment Notes</th>
            <th>Validated Model Evidence</th>
          </tr>
        </thead>
        <tbody>
          {admet_toxicity_rows}
        </tbody>
      </table>
    </div>

    <!-- Scientific References Box -->
    <div style="background-color: #f7fafc; border-left: 4px solid #2b6cb0; padding: 12px 16px; border-radius: 4px; font-size: 12px; color: #4a5568; margin-top: 16px;">
      <strong>Scientific Validation & PubMed Citations:</strong>
      <ul style="margin: 6px 0 0 18px; line-height: 1.6;">
        <li><strong>Caco-2 & RRCK Permeability:</strong> Tan et al., <em>Mol Pharmaceut</em> 2024 (<a href="https://pubmed.ncbi.nlm.nih.gov/39031123/" target="_blank">PMID 39031123</a>); Okumu et al., <em>Pharm Res</em> 1997 (<a href="https://pubmed.ncbi.nlm.nih.gov/9090704/" target="_blank">PMID 9090704</a>).</li>
        <li><strong>Cyclic PAMPA & RRCK:</strong> Jiang et al., <em>Front Bioinf</em> 2025 (<a href="https://pubmed.ncbi.nlm.nih.gov/40134508/" target="_blank">PMID 40134508</a>); Xiong et al., <em>IEEE TCBB</em> 2026 (<a href="https://pubmed.ncbi.nlm.nih.gov/41379908/" target="_blank">PMID 41379908</a>).</li>
        <li><strong>Blood-Brain Barrier (BBB) Transport:</strong> Charoenkwan et al., <em>J Comput Aided Mol Des</em> 2022 (<a href="https://pubmed.ncbi.nlm.nih.gov/36284036/" target="_blank">PMID 36284036</a>); Gu et al., <em>BMC Biol</em> 2024 (<a href="https://pubmed.ncbi.nlm.nih.gov/38637801/" target="_blank">PMID 38637801</a>).</li>
        <li><strong>Peptide Hemolytic Toxicity:</strong> Hasan et al., <em>Bioinformatics</em> 2020 (<a href="https://pubmed.ncbi.nlm.nih.gov/32145017/" target="_blank">PMID 32145017</a>); Rathore et al., <em>Comm Biol</em> 2025 (<a href="https://pubmed.ncbi.nlm.nih.gov/39905233/" target="_blank">PMID 39905233</a>); Plisson et al., <em>Sci Rep</em> 2020 (<a href="https://pubmed.ncbi.nlm.nih.gov/33024236/" target="_blank">PMID 33024236</a>).</li>
        <li><strong>Biological Half-Life:</strong> Bachmair et al., <em>Science</em> 1986 (<a href="https://pubmed.ncbi.nlm.nih.gov/3016897/" target="_blank">PMID 3016897</a>); Varshavsky, <em>PNAS</em> 1996.</li>
      </ul>
    </div>
  </div>

  <!-- Filtering Details -->
  <div class="section-card">
    <h3>Peptide Size & Length Filtering Diagnostics</h3>
    <p style="color: #718096; font-size: 13px; margin-bottom: 16px;">
      Pre-docking filtering audit log. Peptide length filter is removed so all candidates proceed to docking; advisory warnings are shown for candidate review.
    </p>
    <div style="overflow-x: auto;">
      <table>
        <thead>
          <tr>
            <th>Candidate ID</th>
            <th>Sequence</th>
            <th>Length (AA)</th>
            <th>MW (Da)</th>
            <th>Filter Status</th>
            <th>Filter Reason</th>
          </tr>
        </thead>
        <tbody>
          {filtering_rows}
        </tbody>
      </table>
    </div>
  </div>

  <!-- Run Manifest & Reproducibility -->
  <div class="section-card">
    <h3>Run Manifest & Reproducibility</h3>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; font-size: 13px; margin-top: 12px;">
      <div>
        <strong>Run ID:</strong> {html.escape(str(manifest.get('run_id', 'N/A')))}<br>
        <strong>Start Time:</strong> {html.escape(str(manifest.get('start_time', 'N/A')))}<br>
        <strong>End Time:</strong> {html.escape(str(manifest.get('end_time', 'N/A')))}<br>
        <strong>Runtime:</strong> {manifest.get('runtime_sec', 0):.2f}s
      </div>
      <div>
        <strong>LightDock Version:</strong> {html.escape(str(manifest.get('lightdock_version', '0.9.4')))}<br>
        <strong>Scoring Function:</strong> {html.escape(str(manifest.get('scoring_function', 'fastdfire')))}<br>
        <strong>Preset:</strong> {html.escape(str(manifest.get('preset', 'fast')))}<br>
        <strong>Binding Site Mode:</strong> {html.escape(str(manifest.get('binding_site_mode', 'auto')))}
      </div>
      <div>
        <strong>Receptor Checksum:</strong> <span style="font-family: monospace; font-size: 11px;">{html.escape(str(manifest.get('receptor_checksum', 'N/A'))[:24])}...</span><br>
        <strong>Random Seed:</strong> {manifest.get('seed', 42)}<br>
        <strong>Restraints Count:</strong> {manifest.get('restraints_count', 0)}
      </div>
    </div>
  </div>
</div>

<script>
  let posesData = {viewer_json};
  let glviewer = null;
  let currentModel = null;
  let receptorSurface = null;
  let peptideSurface = null;

  const AA_NAMES = {{
    ALA: "Alanine", ARG: "Arginine", ASN: "Asparagine", ASP: "Aspartate",
    CYS: "Cysteine", GLN: "Glutamine", GLU: "Glutamate", GLY: "Glycine",
    HIS: "Histidine", ILE: "Isoleucine", LEU: "Leucine", LYS: "Lysine",
    MET: "Methionine", PHE: "Phenylalanine", PRO: "Proline", SER: "Serine",
    THR: "Threonine", TRP: "Tryptophan", TYR: "Tyrosine", VAL: "Valine",
    SEC: "Selenocysteine", PYL: "Pyrrolysine", MSE: "Selenomethionine"
  }};

  let hoverLabels = [];
  let hoverShapes = [];
  let lastHoveredAtom = null;

  function clearHoverHighlights() {{
    if (!glviewer) return;
    for (let i = 0; i < hoverLabels.length; i++) {{
      try {{ glviewer.removeLabel(hoverLabels[i]); }} catch (e) {{}}
    }}
    hoverLabels = [];
    for (let i = 0; i < hoverShapes.length; i++) {{
      try {{ glviewer.removeShape(hoverShapes[i]); }} catch (e) {{}}
    }}
    hoverShapes = [];
  }}

  function classifyInteraction(resAtoms, partnerAtoms) {{
    if (!resAtoms || resAtoms.length === 0 || !partnerAtoms || partnerAtoms.length === 0) {{
      return {{
        type: "none",
        label: "No direct contact",
        color: "#718096",
        dist: 999.0,
        resAtom: resAtoms ? resAtoms[0] : null,
        partnerAtom: null,
        detail: "No partner atoms nearby"
      }};
    }}

    const donAccElements = new Set(['N', 'O', 'S', 'F']);
    const basicRes = new Set(['ARG', 'LYS', 'HIS']);
    const acidicRes = new Set(['ASP', 'GLU']);
    const aromaticRes = new Set(['PHE', 'TYR', 'TRP', 'HIS']);
    const hydrophobicRes = new Set(['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PRO', 'PHE', 'TRP', 'TYR', 'CYS']);

    let minDist = 999.0;
    let bestResAtom = resAtoms[0];
    let bestPartnerAtom = partnerAtoms[0];
    let interactionsFound = [];

    for (let i = 0; i < resAtoms.length; i++) {{
      let ra = resAtoms[i];
      let raElem = (ra.elem || (ra.atom ? ra.atom.replace(/[^a-zA-Z]/g, '').slice(0, 1) : '')).toUpperCase();
      if (raElem === 'H') continue;

      for (let j = 0; j < partnerAtoms.length; j++) {{
        let pa = partnerAtoms[j];
        let paElem = (pa.elem || (pa.atom ? pa.atom.replace(/[^a-zA-Z]/g, '').slice(0, 1) : '')).toUpperCase();
        if (paElem === 'H') continue;

        let dx = ra.x - pa.x;
        let dy = ra.y - pa.y;
        let dz = ra.z - pa.z;
        let d = Math.sqrt(dx*dx + dy*dy + dz*dz);

        if (d < minDist) {{
          minDist = d;
          bestResAtom = ra;
          bestPartnerAtom = pa;
        }}

        // 1. Steric Clash (< 2.0 A)
        if (d < 2.0) {{
          interactionsFound.push({{
            type: "clash",
            priority: 10,
            label: "⚠️ Steric Clash",
            color: "#e53e3e",
            dist: d,
            resAtom: ra,
            partnerAtom: pa,
            detail: `Heavy atom steric overlap (${{d.toFixed(2)}} Å)`
          }});
        }}

        // 2. Salt Bridge (< 4.2 A)
        let resIsBasic = basicRes.has(ra.resn) && (ra.atom && (ra.atom.includes('N') || raElem === 'N'));
        let resIsAcidic = acidicRes.has(ra.resn) && (ra.atom && (ra.atom.includes('O') || raElem === 'O'));
        let partnerIsAcidic = (acidicRes.has(pa.resn) && (pa.atom && pa.atom.includes('O'))) || (paElem === 'O' && pa.resn !== 'HOH');
        let partnerIsBasic = (basicRes.has(pa.resn) && (pa.atom && pa.atom.includes('N'))) || (paElem === 'N');

        if ((resIsBasic && partnerIsAcidic) || (resIsAcidic && partnerIsBasic)) {{
          if (d <= 4.2) {{
            interactionsFound.push({{
              type: "salt_bridge",
              priority: 9,
              label: "⚡ Salt Bridge / Ionic",
              color: "#9f7aea",
              dist: d,
              resAtom: ra,
              partnerAtom: pa,
              detail: `Ionic attraction between oppositely charged groups (${{d.toFixed(2)}} Å)`
            }});
          }}
        }}

        // 3. Hydrogen Bond (2.4 - 3.5 A)
        if (donAccElements.has(raElem) && donAccElements.has(paElem)) {{
          if (d >= 2.4 && d <= 3.5) {{
            interactionsFound.push({{
              type: "hbond",
              priority: 8,
              label: "⚡ Hydrogen Bond",
              color: "#48bb78",
              dist: d,
              resAtom: ra,
              partnerAtom: pa,
              detail: `Heteroatom donor–acceptor pair: ${{ra.atom || raElem}} ··· ${{pa.atom || paElem}} (${{d.toFixed(2)}} Å)`
            }});
          }}
        }}

        // 4. Cation-pi (< 4.5 A)
        let resIsArom = aromaticRes.has(ra.resn);
        let partnerIsArom = aromaticRes.has(pa.resn) || (paElem === 'C' && partnerAtoms.length < 60);
        if ((resIsArom && partnerIsBasic) || (resIsBasic && partnerIsArom)) {{
          if (d <= 4.5) {{
            interactionsFound.push({{
              type: "cation_pi",
              priority: 7,
              label: "⚡ Cation–π Interaction",
              color: "#ed64a6",
              dist: d,
              resAtom: ra,
              partnerAtom: pa,
              detail: `Electrostatic π-system interaction with basic group (${{d.toFixed(2)}} Å)`
            }});
          }}
        }}

        // 5. pi-pi Stacking (3.2 - 4.8 A)
        if (resIsArom && partnerIsArom) {{
          if (d >= 3.2 && d <= 4.8) {{
            interactionsFound.push({{
              type: "pi_stacking",
              priority: 6,
              label: "⚡ π–π Stacking",
              color: "#4299e1",
              dist: d,
              resAtom: ra,
              partnerAtom: pa,
              detail: `Aromatic ring stacking / quadrupole interaction (${{d.toFixed(2)}} Å)`
            }});
          }}
        }}

        // 6. Hydrophobic Contact (< 4.5 A)
        let resIsHydro = hydrophobicRes.has(ra.resn) || raElem === 'C';
        let partnerIsHydro = hydrophobicRes.has(pa.resn) || paElem === 'C';
        if (resIsHydro && partnerIsHydro && raElem === 'C' && paElem === 'C') {{
          if (d <= 4.5) {{
            interactionsFound.push({{
              type: "hydrophobic",
              priority: 5,
              label: "Hydrophobic Contact",
              color: "#ed8936",
              dist: d,
              resAtom: ra,
              partnerAtom: pa,
              detail: `Non-polar carbon dispersion / desolvation contact (${{d.toFixed(2)}} Å)`
            }});
          }}
        }}

        // 7. Van der Waals Contact (< 4.2 A)
        if (d <= 4.2) {{
          interactionsFound.push({{
            type: "vdw",
            priority: 4,
            label: "Van der Waals Contact",
            color: "#38b2ac",
            dist: d,
            resAtom: ra,
            partnerAtom: pa,
            detail: `Close steric contact (${{d.toFixed(2)}} Å)`
          }});
        }}
      }}
    }}

    if (interactionsFound.length > 0) {{
      interactionsFound.sort((a, b) => (b.priority - a.priority) || (a.dist - b.dist));
      let best = interactionsFound[0];
      return {{
        type: best.type,
        label: best.label,
        color: best.color,
        dist: best.dist,
        resAtom: best.resAtom,
        partnerAtom: best.partnerAtom,
        detail: best.detail
      }};
    }}

    if (minDist <= 5.5) {{
      return {{
        type: "pocket_lining",
        label: "Pocket Lining Residue",
        color: "#a0aec0",
        dist: minDist,
        resAtom: bestResAtom,
        partnerAtom: bestPartnerAtom,
        detail: `Cavity residue in proximity to partner (${{minDist.toFixed(2)}} Å)`
      }};
    }}

    return {{
      type: "surface",
      label: "No Direct Contact",
      color: "#718096",
      dist: minDist,
      resAtom: bestResAtom,
      partnerAtom: bestPartnerAtom,
      detail: `Solvent-exposed surface residue (${{minDist.toFixed(2)}} Å from partner)`
    }};
  }}

  function onHoverAtom(atom, viewer) {{
    if (!glviewer || !atom) return;
    if (lastHoveredAtom && lastHoveredAtom === atom) return;

    clearHoverHighlights();
    lastHoveredAtom = atom;

    let chain = atom.chain || 'A';
    let isPeptide = (chain === 'B');
    let resi = atom.resi;
    let resn = atom.resn || 'UNK';
    let aaFullName = AA_NAMES[resn] ? `${{resn}} (${{AA_NAMES[resn]}})` : resn;
    let resStr = `${{isPeptide ? 'Peptide' : 'Receptor'}} ${{chain}}:${{resn}}${{resi}}`;

    let resSel = {{chain: chain, resi: resi}};
    let resAtoms = glviewer.getAtomsFromSel(resSel);
    if (!resAtoms || resAtoms.length === 0) resAtoms = [atom];

    let partnerSel = isPeptide ? {{chain: 'A'}} : {{chain: 'B'}};
    let partnerAtoms = glviewer.getAtomsFromSel(partnerSel) || [];

    // Render prominent 3D ball-and-stick highlight directly on the hovered residue
    for (let i = 0; i < resAtoms.length; i++) {{
      let a = resAtoms[i];
      let aElem = (a.elem || (a.atom ? a.atom.replace(/[^a-zA-Z]/g, '').slice(0, 1) : '')).toUpperCase();
      if (aElem === 'H') continue;

      let sph = glviewer.addSphere({{
        center: {{x: a.x, y: a.y, z: a.z}},
        radius: isPeptide ? 0.32 : 0.28,
        color: '#f6e05e',
        opacity: 0.95
      }});
      if (sph) hoverShapes.push(sph);

      for (let j = i + 1; j < resAtoms.length; j++) {{
        let b = resAtoms[j];
        let bElem = (b.elem || (b.atom ? b.atom.replace(/[^a-zA-Z]/g, '').slice(0, 1) : '')).toUpperCase();
        if (bElem === 'H') continue;
        let dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
        let d2 = dx*dx + dy*dy + dz*dz;
        if (d2 >= 1.30 && d2 <= 3.50) {{
          let cyl = glviewer.addCylinder({{
            start: {{x: a.x, y: a.y, z: a.z}},
            end: {{x: b.x, y: b.y, z: b.z}},
            radius: isPeptide ? 0.20 : 0.16,
            color: '#f6e05e',
            opacity: 0.95,
            fromCap: 1,
            toCap: 1
          }});
          if (cyl) hoverShapes.push(cyl);
        }}
      }}
    }}

    let inter = classifyInteraction(resAtoms, partnerAtoms);

    // Highlight partner atom and draw connector cylinder
    if (inter.partnerAtom && inter.dist <= 5.0) {{
      let sph = glviewer.addSphere({{
        center: {{x: inter.partnerAtom.x, y: inter.partnerAtom.y, z: inter.partnerAtom.z}},
        radius: 0.40,
        color: inter.color,
        opacity: 0.90
      }});
      if (sph) hoverShapes.push(sph);

      let cyl = glviewer.addCylinder({{
        start: {{x: inter.resAtom.x, y: inter.resAtom.y, z: inter.resAtom.z}},
        end: {{x: inter.partnerAtom.x, y: inter.partnerAtom.y, z: inter.partnerAtom.z}},
        radius: 0.10,
        dashed: true,
        color: inter.color,
        fromCap: 1,
        toCap: 1
      }});
      if (cyl) hoverShapes.push(cyl);
    }}

    // In-canvas 3D label
    let distTxt = (inter.dist < 100) ? `${{inter.dist.toFixed(2)}} Å` : "No contact";
    let labelContent = `${{resStr}}\n${{inter.label}} (${{distTxt}})`;
    let lbl = glviewer.addLabel(labelContent, {{
      position: {{x: atom.x, y: atom.y, z: atom.z}},
      backgroundColor: "#0f172a",
      backgroundOpacity: 0.92,
      fontColor: "#ffffff",
      fontSize: 12,
      borderColor: inter.color,
      borderThickness: 1.5,
      inFront: true
    }});
    if (lbl) hoverLabels.push(lbl);

    // Update bottom HUD
    let hudElem = document.getElementById("interaction_hud");
    if (hudElem) hudElem.style.display = "block";

    let hudMain = document.getElementById("hud_main");
    let hudBadge = document.getElementById("hud_badge");
    if (hudMain) {{
      let partnerDesc = inter.partnerAtom 
        ? `${{isPeptide ? 'Receptor' : 'Peptide'}} ${{inter.partnerAtom.chain || ''}}:${{inter.partnerAtom.resn || ''}}${{inter.partnerAtom.resi || ''}} (${{inter.partnerAtom.atom || inter.partnerAtom.elem}})`
        : "None";

      hudMain.innerHTML = `
        <span style="font-weight: 700; color: #f6e05e;">[${{resStr}} ${{aaFullName}}]</span>
        <span style="color: #cbd5e1;">↔</span>
        <span style="color: #ffffff; font-weight: 600;">${{partnerDesc}}</span>
        <span style="color: #94a3b8; font-size: 12px;">(${{inter.detail}})</span>
      `;
    }}
    if (hudBadge) {{
      hudBadge.innerHTML = `
        <span class="badge" style="background: ${{inter.color}}; color: #ffffff; padding: 4px 10px; border-radius: 4px; font-weight: 700;">
          ${{inter.label}}
        </span>
      `;
    }}

    glviewer.render();
  }}

  function onUnhoverAtom(atom, viewer) {{
    clearHoverHighlights();
    lastHoveredAtom = null;

    let hudElem = document.getElementById("interaction_hud");
    if (hudElem) hudElem.style.display = "none";
    let hudMain = document.getElementById("hud_main");
    let hudBadge = document.getElementById("hud_badge");
    if (hudMain) {{
      hudMain.innerHTML = "";
    }}
    if (hudBadge) {{
      hudBadge.innerHTML = "";
    }}

    if (glviewer) glviewer.render();
  }}

  function initViewer() {{
    let container = document.getElementById("complex_3d_viewer");
    if (!container || !window.$3Dmol) return;

    glviewer = $3Dmol.createViewer(container, {{ defaultcolors: $3Dmol.rasmolElementColors }});
    glviewer.setBackgroundColor(0x1a202c);
    glviewer.setHoverDuration(20);

    // Continuous mousemove raycaster listener on container
    if (!container._hasHoverMoveListener) {{
      container._hasHoverMoveListener = true;
      let lastMoveTime = 0;
      container.addEventListener("mousemove", function(e) {{
        let now = Date.now();
        if (now - lastMoveTime > 30 && glviewer && glviewer.mouseXY && glviewer.hoverables && glviewer.hoverables.length > 0) {{
          lastMoveTime = now;
          let s = glviewer.mouseXY(e.pageX, e.pageY);
          if (s && s.x >= -1 && s.x <= 1 && s.y >= -1 && s.y <= 1) {{
            try {{
              glviewer.handleHoverSelection(s.x, s.y, e);
            }} catch (err) {{}}
          }}
        }}
      }}, {{passive: true}});
      container.addEventListener("mouseleave", function(e) {{
        if (glviewer) {{
          try {{ glviewer.setHover(null); }} catch (err) {{}}
          onUnhoverAtom(null, glviewer);
        }}
      }});
    }}

    let selectElem = document.getElementById("poseSelect");
    if (!posesData || posesData.length === 0) {{
      selectElem.innerHTML = "<option>No 3D poses available</option>";
      container.innerHTML = "<div style='color: #a0aec0; padding: 40px; text-align: center;'>No 3D PDB poses available for viewer.</div>";
      return;
    }}

    selectElem.innerHTML = "";
    posesData.forEach((p, idx) => {{
      let opt = document.createElement("option");
      opt.value = idx;
      opt.textContent = p.label;
      selectElem.appendChild(opt);
    }});

    selectElem.addEventListener("change", function() {{
      loadPose(parseInt(this.value));
    }});

    loadPose(0);
  }}

  function loadPose(index) {{
    if (!glviewer || !posesData[index]) return;
    clearHoverHighlights();
    glviewer.clear();
    glviewer.removeAllSurfaces();
    glviewer.removeAllShapes();
    window._hbondCylinders = [];

    let p = posesData[index];
    currentModel = glviewer.addModel(p.pdb_data, "pdb");
    updateComplexStyles();
    glviewer.zoomTo();
  }}

  function getProteinStyleObj(style, color) {{
    let styleObj = {{}};
    let colOpt = {{}};
    if (color === "spectrum") {{
      colOpt = {{ colorscheme: "spectrum" }};
    }} else if (color === "ss") {{
      colOpt = {{ colorscheme: "ssJmol" }};
    }} else if (color === "b") {{
      colOpt = {{ colorscheme: "bFactor" }};
    }} else {{
      colOpt = {{ color: color || "cyan" }};
    }}

    if (style === "cartoon") {{
      styleObj.cartoon = colOpt;
    }} else if (style === "ribbon") {{
      styleObj.cartoon = Object.assign({{ ribbon: true, thickness: 0.35, style: "oval" }}, colOpt);
    }} else if (style === "stick") {{
      styleObj.stick = colOpt;
    }} else if (style === "line") {{
      styleObj.line = colOpt;
    }} else if (style === "surface") {{
      styleObj.cartoon = Object.assign({{ opacity: 0.25 }}, colOpt);
    }}
    styleObj.clicksphere = {{ radius: 1.2 }};
    return styleObj;
  }}

  function getPeptideStyleObj(style, color) {{
    let styleObj = {{}};
    let colOpt = {{}};
    if (color === "polarity") {{
      colOpt = {{ colorscheme: "amino" }};
    }} else if (color === "spectrum") {{
      colOpt = {{ colorscheme: "spectrum" }};
    }} else if (color === "greenCarbon") {{
      colOpt = {{ colorscheme: "greenCarbon" }};
    }} else {{
      colOpt = {{ colorscheme: "orangeCarbon" }};
    }}

    if (style === "stick") {{
      styleObj.stick = Object.assign({{ radius: 0.25 }}, colOpt);
    }} else if (style === "ballAndStick") {{
      styleObj.stick = Object.assign({{ radius: 0.18 }}, colOpt);
      styleObj.sphere = Object.assign({{ scale: 0.28 }}, colOpt);
    }} else if (style === "cartoon") {{
      styleObj.cartoon = colOpt;
    }} else if (style === "sphere") {{
      styleObj.sphere = Object.assign({{ scale: 0.85 }}, colOpt);
    }} else if (style === "surface") {{
      styleObj.stick = Object.assign({{ radius: 0.2 }}, colOpt);
    }}
    styleObj.clicksphere = {{ radius: 1.2 }};
    return styleObj;
  }}

  function updateComplexStyles() {{
    if (!glviewer || !currentModel) return;

    let protStyleElem = document.getElementById("proteinStyle");
    let protColorElem = document.getElementById("proteinColor");
    let pepStyleElem = document.getElementById("peptideStyle");
    let pepColorElem = document.getElementById("peptideColor");
    let chkProtElem = document.getElementById("chkProtein");
    let chkPepElem = document.getElementById("chkPeptide");
    let chkContElem = document.getElementById("chkContacts");
    let chkHbondElem = document.getElementById("chkHbonds");

    let protStyle = protStyleElem ? protStyleElem.value : "cartoon";
    let protColor = protColorElem ? protColorElem.value : "cyan";
    let pepStyle = pepStyleElem ? pepStyleElem.value : "stick";
    let pepColor = pepColorElem ? pepColorElem.value : "orangeCarbon";
    let showProt = chkProtElem ? chkProtElem.checked : true;
    let showPep = chkPepElem ? chkPepElem.checked : true;
    let showContacts = chkContElem ? chkContElem.checked : false;
    let showHbonds = chkHbondElem ? chkHbondElem.checked : false;

    clearHoverHighlights();
    glviewer.removeAllSurfaces();
    glviewer.removeAllShapes();

    glviewer.setStyle({{}}, {{}});

    // Receptor (chain A)
    if (showProt) {{
      let pObj = getProteinStyleObj(protStyle, protColor);
      glviewer.setStyle({{chain: 'A'}}, pObj);
      if (protStyle === "surface") {{
        let sColor = protColor === "cyan" ? "cyan" : "white";
        glviewer.addSurface($3Dmol.SurfaceType.VDW, {{opacity: 0.45, color: sColor}}, {{chain: 'A'}}).then(function() {{
          glviewer.render();
        }}).catch(function(e) {{
          console.warn("Receptor surface error:", e);
        }});
      }}
    }} else {{
      glviewer.setStyle({{chain: 'A'}}, {{}});
    }}

    // Peptide (chain B)
    let pepSel = {{chain: 'B'}};
    if (showPep) {{
      let lObj = getPeptideStyleObj(pepStyle, pepColor);
      glviewer.setStyle(pepSel, lObj);
      if (pepStyle === "surface") {{
        let pScheme = pepColor === "greenCarbon" ? "greenCarbon" : "orangeCarbon";
        glviewer.addSurface($3Dmol.SurfaceType.MS, {{opacity: 0.75, colorscheme: pScheme}}, pepSel).then(function() {{
          glviewer.render();
        }}).catch(function(e) {{
          console.warn("Peptide surface error:", e);
        }});
      }}
    }} else {{
      glviewer.setStyle(pepSel, {{}});
    }}

    // Contact residues (<4.5A)
    if (showContacts && showProt && showPep) {{
      glviewer.addStyle({{chain: 'A', within: {{distance: 4.5, sel: {{chain: 'B'}}}}}}, {{
        stick: {{colorscheme: 'magentaCarbon', radius: 0.18}}
      }});
      glviewer.addStyle({{chain: 'B', within: {{distance: 4.5, sel: {{chain: 'A'}}}}}}, {{
        stick: {{colorscheme: 'yellowCarbon', radius: 0.28}}
      }});
    }}

    // Hydrogen bonds
    window._hbondCylinders = [];
    if (showHbonds && showProt && showPep) {{
      let recAtoms = glviewer.getAtomsFromSel({{chain: 'A'}});
      let pepAtoms = glviewer.getAtomsFromSel({{chain: 'B'}});
      if (recAtoms && pepAtoms) {{
        let donAcc = ['N', 'O', 'S'];
        for (let i = 0; i < recAtoms.length; i++) {{
          let r = recAtoms[i];
          if (!donAcc.includes(r.elem)) continue;
          for (let j = 0; j < pepAtoms.length; j++) {{
            let p = pepAtoms[j];
            if (!donAcc.includes(p.elem)) continue;
            let dx = r.x - p.x;
            let dy = r.y - p.y;
            let dz = r.z - p.z;
            let d2 = dx*dx + dy*dy + dz*dz;
            if (d2 >= 5.76 && d2 <= 12.25) {{
              let cyl = glviewer.addCylinder({{
                start: {{x: r.x, y: r.y, z: r.z}},
                end: {{x: p.x, y: p.y, z: p.z}},
                radius: 0.08,
                dashed: true,
                color: "#f6e05e",
                fromCap: 1,
                toCap: 1
              }});
              window._hbondCylinders.push(cyl);
            }}
          }}
        }}
      }}
    }}

    // Register hover and click handlers across all atoms in loaded structure
    glviewer.setHoverDuration(20);
    glviewer.setHoverable({{}}, true, onHoverAtom, onUnhoverAtom);
    glviewer.setClickable({{}}, true, onHoverAtom);

    glviewer.render();
  }}

  function focusPeptide() {{
    if (!glviewer) return;
    glviewer.zoomTo({{chain: 'B'}}, 1000);
    glviewer.render();
  }}

  function resetViewerCamera() {{
    if (!glviewer) return;
    glviewer.zoomTo();
    glviewer.render();
  }}

  function toggleContacts(val) {{
    updateComplexStyles();
  }}

  function toggleHbonds(val) {{
    updateComplexStyles();
  }}

  document.addEventListener("DOMContentLoaded", initViewer);
</script>
</body>
</html>
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Peptide docking HTML report successfully generated at: {report_path}")
    return str(report_path.resolve())
