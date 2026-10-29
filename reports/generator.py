"""
DrugLab Standalone Interactive Report Generator.

Generates comprehensive, publication-grade HTML reports containing:
  - Executive KPI scorecards
  - Interactive sortable data tables with embedded 2D vector (SVG) structures
  - Interactive 3D molecular pose viewer powered by 3Dmol.js (py3Dmol)
  - Interactive Plotly charts (screening funnel, affinity vs risk, property distributions, radar charts)
  - Quarantined failure diagnostics table
  - Execution metadata and config snapshots
"""

import os
import sys
import re
import json
import base64
import html
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Any
import pandas as pd
import numpy as np

from utils.logger import setup_logger, log_step
from utils.io import ensure_dir, read_csv, read_json, list_files
from utils.quarantine import load_quarantine

logger = setup_logger("report_generator", level="INFO")

# Try importing RDKit drawing
_HAS_RDKIT_DRAW = False
try:
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    _HAS_RDKIT_DRAW = True
except Exception:
    _HAS_RDKIT_DRAW = False


def _mol_to_svg(mol, width: int = 240, height: int = 180) -> str:
    """Render an RDKit Mol to inline SVG string."""
    if not _HAS_RDKIT_DRAW or mol is None:
        return "<div style='color:#888;font-size:12px;padding:20px 0;'>No 2D Preview</div>"
    try:
        mol_copy = Chem.Mol(mol)
        try:
            from rdkit.Chem import AllChem
            AllChem.Compute2DCoords(mol_copy)
        except Exception:
            pass

        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.drawOptions().clearBackground = False
        drawer.drawOptions().padding = 0.05
        drawer.DrawMolecule(mol_copy)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        # Clean SVG header for direct inline embedding
        svg = svg.replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")
        svg = svg.replace("<!DOCTYPE svg PUBLIC '-//W3C//DTD SVG 1.1//EN'\n 'http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd'>", "")
        return svg
    except Exception as e:
        return f"<div style='color:#888;font-size:11px;'>2D Render error: {e}</div>"


def _smiles_to_svg(smiles: str, width: int = 240, height: int = 180) -> str:
    """Render a SMILES string to inline SVG."""
    if not _HAS_RDKIT_DRAW or not smiles or pd.isna(smiles):
        return "<div style='color:#888;font-size:12px;padding:20px 0;'>No Structure</div>"
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return _mol_to_svg(mol, width, height)
    except Exception:
        return "<div style='color:#888;font-size:12px;padding:20px 0;'>Invalid Structure</div>"


def _read_df_safe(path: str) -> Optional[pd.DataFrame]:
    """Safely read CSV into DataFrame if file exists."""
    if os.path.isfile(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def _extract_docked_ligands_and_poses(output_dir: str, ranking_df=None, docking_df=None) -> dict[str, dict]:
    """
    Extract all docked ligands and all their individual poses with affinities, RMSD, and 3D coordinates.
    Supports PDBQT pose files, multi-pose SDF files, and best pose fallbacks.
    """
    docked_data: dict[str, dict] = {}
    dock_dir = os.path.join(output_dir, "docking")
    poses_dir = os.path.join(dock_dir, "poses")

    # 1. Collect metadata from ranking/docking CSVs
    meta_by_name = {}
    if ranking_df is not None and not ranking_df.empty:
        for _, row in ranking_df.iterrows():
            nm = str(row.get("name", row.get("Compound", "")))
            if nm:
                le_val = row.get("ligand_efficiency")
                lle_val = row.get("lipophilic_efficiency")
                cs_val = row.get("composite_score", row.get("FinalScore"))
                meta_by_name[nm] = {
                    "smiles": str(row.get("smiles", "")),
                    "best_affinity": row.get("best_affinity", row.get("Docking")),
                    "le": f"{float(le_val):.3f}" if pd.notnull(le_val) and isinstance(le_val, (int, float)) else None,
                    "lle": f"{float(lle_val):.2f}" if pd.notnull(lle_val) and isinstance(lle_val, (int, float)) else None,
                    "composite_score": f"{float(cs_val):.3f}" if pd.notnull(cs_val) and isinstance(cs_val, (int, float)) else None,
                }

    if docking_df is not None and not docking_df.empty:
        for _, row in docking_df.iterrows():
            nm = str(row.get("name", row.get("Compound", "")))
            if nm and nm not in meta_by_name:
                meta_by_name[nm] = {
                    "smiles": str(row.get("smiles", "")),
                    "best_affinity": row.get("best_affinity"),
                    "le": None,
                    "lle": None,
                    "composite_score": None,
                }

    # 2. Check docking_report.json for affinities list
    report_json_path = os.path.join(dock_dir, "docking_report.json")
    affinities_by_name = {}
    if os.path.exists(report_json_path):
        try:
            with open(report_json_path, "r", encoding="utf-8") as f:
                r_data = json.load(f)
                for item in r_data.get("docking_results", []):
                    nm = item.get("name")
                    if nm and "all_affinities" in item:
                        affinities_by_name[nm] = item["all_affinities"]
        except Exception:
            pass

    # 3. Check for PDBQT pose files in poses_dir and dock_dir
    candidate_pdbqt_files = []
    if os.path.isdir(poses_dir):
        candidate_pdbqt_files.extend(list(Path(poses_dir).glob("*_poses.pdbqt")))
        candidate_pdbqt_files.extend(list(Path(poses_dir).glob("*.pdbqt")))
    if os.path.isdir(dock_dir):
        candidate_pdbqt_files.extend(list(Path(dock_dir).glob("*_poses.pdbqt")))

    seen_pdbqt = set()
    for pq_path in candidate_pdbqt_files:
        pq_str = str(pq_path)
        if pq_str in seen_pdbqt or pq_path.name.startswith("protein") or "temp" in pq_str:
            continue
        seen_pdbqt.add(pq_str)

        lig_name = pq_path.stem.replace("_poses", "").replace(".pdbqt", "")
        try:
            with open(pq_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            continue

        raw_models = [m for m in re.split(r"^MODEL\s+\d+", content, flags=re.MULTILINE) if m.strip()]
        poses = []
        known_affs = affinities_by_name.get(lig_name, [])

        # Try obabel conversion to get clean SDF format with complete bond connectivity
        sdf_blocks = []
        obabel_bin = shutil.which("obabel")
        if not obabel_bin:
            for cand in [
                os.path.join(os.path.dirname(sys.executable), "obabel"),
                "/home/faiyaz/miniconda3/envs/druglab/bin/obabel",
                os.path.expanduser("~/miniconda3/envs/druglab/bin/obabel"),
            ]:
                if os.path.isfile(cand) and os.access(cand, os.X_OK):
                    obabel_bin = cand
                    break

        if obabel_bin:
            try:
                res = subprocess.run(
                    [obabel_bin, str(pq_path), "-osdf"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if res.returncode == 0 and "$$$$" in res.stdout:
                    raw_sdfs = [b.strip() for b in res.stdout.split("$$$$") if b.strip()]
                    sdf_blocks = [b + "\n$$$$\n" for b in raw_sdfs]
            except Exception:
                sdf_blocks = []

        for rm in raw_models:
            aff = None
            rmsd_lb = 0.0
            rmsd_ub = 0.0
            vm = re.search(r"REMARK VINA RESULT:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", rm)
            if vm:
                aff = float(vm.group(1))
                rmsd_lb = float(vm.group(2))
                rmsd_ub = float(vm.group(3))
            elif len(poses) < len(known_affs):
                aff = float(known_affs[len(poses)])

            pose_idx = len(poses)
            if pose_idx < len(sdf_blocks):
                poses.append({
                    "pose_id": pose_idx + 1,
                    "affinity": aff,
                    "rmsd_lb": rmsd_lb,
                    "rmsd_ub": rmsd_ub,
                    "format": "sdf",
                    "data": sdf_blocks[pose_idx]
                })
            else:
                # Fallback to clean PDB with HETATM and distance-based CONECT records
                pdb_lines = []
                atom_coords = []
                for line in rm.splitlines():
                    if line.startswith(("ATOM", "HETATM")):
                        idx = len(atom_coords) + 1
                        try:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            atom_name = line[12:16].strip()
                            elem = atom_name[0] if atom_name else "C"
                            atom_coords.append((idx, x, y, z, elem))
                        except Exception:
                            elem = "C"
                        base = f"HETATM{line[6:66].ljust(60)}"
                        pdb_lines.append(f"{base}          {elem:>2}")

                conect_map = {}
                for i, (i_idx, x1, y1, z1, el1) in enumerate(atom_coords):
                    for j in range(i + 1, len(atom_coords)):
                        j_idx, x2, y2, z2, el2 = atom_coords[j]
                        d2 = (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
                        cutoff = 1.25 if (el1 == "H" or el2 == "H") else 1.85
                        if 0.16 < d2 <= (cutoff * cutoff):
                            conect_map.setdefault(i_idx, []).append(j_idx)
                            conect_map.setdefault(j_idx, []).append(i_idx)

                for c_idx in sorted(conect_map.keys()):
                    b_str = "".join(f"{b:>5}" for b in conect_map[c_idx])
                    pdb_lines.append(f"CONECT{c_idx:>5}{b_str}")

                pdb_lines.append("END\n")
                if pdb_lines:
                    poses.append({
                        "pose_id": pose_idx + 1,
                        "affinity": aff,
                        "rmsd_lb": rmsd_lb,
                        "rmsd_ub": rmsd_ub,
                        "format": "pdb",
                        "data": "\n".join(pdb_lines)
                    })

        if poses:
            meta = meta_by_name.get(lig_name, {})
            best_aff = poses[0]["affinity"] if poses[0]["affinity"] is not None else meta.get("best_affinity")
            docked_data[lig_name] = {
                "name": lig_name,
                "smiles": meta.get("smiles", ""),
                "best_affinity": float(best_aff) if best_aff is not None else -7.0,
                "le": meta.get("le"),
                "lle": meta.get("lle"),
                "composite_score": meta.get("composite_score"),
                "poses": poses
            }

    # 4. Check SDF files (docking/best_poses.sdf, lead_selection/leads.sdf, etc.)
    sdf_candidates = [
        os.path.join(output_dir, "lead_selection", "leads.sdf"),
        os.path.join(output_dir, "docking", "best_poses.sdf"),
        os.path.join(output_dir, "lead_selection", "best_lead.sdf"),
        os.path.join(output_dir, "ranking", "top_candidates.sdf"),
    ]
    for sc in sdf_candidates:
        if not os.path.isfile(sc):
            continue
        try:
            with open(sc, "r", encoding="utf-8", errors="replace") as f:
                sdf_text = f.read()
            mols = sdf_text.split("$$$$")
            for m in mols:
                if not m.strip():
                    continue
                lines = m.strip().splitlines()
                m_name = lines[0].strip() if lines else "Compound"
                if m_name.endswith(".pdbqt") or "/" in m_name:
                    m_name = Path(m_name).stem.replace("_poses", "")

                aff = None
                aff_match = re.search(r">\s+<.*?affinity.*?>\s*\n\s*([-\d.]+)", m, re.IGNORECASE)
                if aff_match:
                    aff = float(aff_match.group(1))
                elif m_name in meta_by_name and meta_by_name[m_name].get("best_affinity") is not None:
                    aff = float(meta_by_name[m_name]["best_affinity"])

                if m_name not in docked_data:
                    meta = meta_by_name.get(m_name, {})
                    docked_data[m_name] = {
                        "name": m_name,
                        "smiles": meta.get("smiles", ""),
                        "best_affinity": aff if aff is not None else -7.0,
                        "le": meta.get("le"),
                        "lle": meta.get("lle"),
                        "composite_score": meta.get("composite_score"),
                        "poses": [{
                            "pose_id": 1,
                            "affinity": aff,
                            "rmsd_lb": 0.0,
                            "rmsd_ub": 0.0,
                            "format": "sdf",
                            "data": m.strip() + "\n$$$$\n"
                        }]
                    }
        except Exception:
            pass

    # Sort ligands by best affinity (most negative first)
    sorted_data = dict(sorted(
        docked_data.items(),
        key=lambda item: float(item[1].get("best_affinity", 0.0) if item[1].get("best_affinity") is not None else 0.0)
    ))
    return sorted_data


def generate_report(
    output_dir: str,
    output_file: Optional[str] = None,
    report_title: str = "DrugLab Screening & Optimization Report",
) -> str:
    """
    Generate a standalone interactive HTML report from pipeline output directories.

    Args:
        output_dir: The DrugLab output directory containing results.
        output_file: Destination HTML file path (default: {output_dir}/reports/summary.html).
        report_title: Title heading for the report.

    Returns:
        Path to the generated HTML file.
    """
    start_time = datetime.utcnow()
    log_step(logger, "Report Generation", "started", {"output_dir": output_dir})

    reports_dir = os.path.join(output_dir, "reports")
    ensure_dir(reports_dir)
    if output_file is None:
        output_file = os.path.join(reports_dir, "summary.html")

    # 1. Discover available pipeline outputs
    docking_csv = _read_df_safe(os.path.join(output_dir, "docking", "docking_results.csv"))
    ranking_csv = _read_df_safe(os.path.join(output_dir, "lead_selection", "ranking.csv"))
    top_hits_csv = _read_df_safe(os.path.join(output_dir, "lead_selection", "top_hits.csv"))
    opt_ranking_csv = _read_df_safe(os.path.join(output_dir, "ranking", "optimized_candidates.csv"))
    admet_csv = _read_df_safe(os.path.join(output_dir, "admet", "admet_results.csv"))
    dl_csv = _read_df_safe(os.path.join(output_dir, "druglikeness", "descriptors.csv"))
    
    quarantine_records = load_quarantine(output_dir)

    # Determine primary leads dataframe
    primary_df = None
    workflow_type = "Virtual Screening"
    if opt_ranking_csv is not None and not opt_ranking_csv.empty:
        primary_df = opt_ranking_csv
        workflow_type = "Lead Optimization"
    elif top_hits_csv is not None and not top_hits_csv.empty:
        primary_df = top_hits_csv
    elif ranking_csv is not None and not ranking_csv.empty:
        primary_df = ranking_csv.head(10)
    elif docking_csv is not None and not docking_csv.empty:
        primary_df = docking_csv.sort_values("best_affinity", ascending=True).head(10)

    # 2. Compute KPI Metrics
    total_screened = 0
    passed_dl = 0
    passed_admet = 0
    total_docked = len(docking_csv) if docking_csv is not None else 0
    best_affinity = "N/A"
    top_lead_name = "N/A"
    top_lead_score = "N/A"

    if dl_csv is not None:
        total_screened = len(dl_csv)
        passed_dl = len(dl_csv[dl_csv["status"] == "PASS"]) if "status" in dl_csv.columns else len(dl_csv)
    elif docking_csv is not None:
        total_screened = len(docking_csv)

    if admet_csv is not None:
        passed_admet = len(admet_csv[admet_csv["status"] == "PASS"]) if "status" in admet_csv.columns else len(admet_csv)

    if docking_csv is not None and "best_affinity" in docking_csv.columns and not docking_csv.empty:
        min_aff = docking_csv["best_affinity"].min()
        best_affinity = f"{min_aff:.2f} kcal/mol"

    if primary_df is not None and not primary_df.empty:
        first_row = primary_df.iloc[0]
        top_lead_name = str(first_row.get("name", first_row.get("Compound", "Lead 1")))
        if "composite_score" in first_row:
            top_lead_score = f"{float(first_row['composite_score']):.3f}"
        elif "FinalScore" in first_row:
            top_lead_score = f"{float(first_row['FinalScore']):.1f}"
        elif "best_affinity" in first_row:
            top_lead_score = f"{float(first_row['best_affinity']):.2f} kcal/mol"

    # 3. Read 3D Structures for py3Dmol Viewer
    # Locate protein structure
    protein_pdb_content = ""
    protein_candidates = [
        os.path.join(output_dir, "protein_preparation", "protein_clean.pdb"),
        os.path.join(output_dir, "protein_preparation", "protein.pdb"),
    ]
    for pc in protein_candidates:
        if os.path.isfile(pc):
            try:
                with open(pc, "r", encoding="utf-8", errors="replace") as pf:
                    protein_pdb_content = pf.read()
                break
            except Exception:
                pass

    # Extract all docked ligands and poses for 3D Viewer
    docked_ligands_data = _extract_docked_ligands_and_poses(
        output_dir=output_dir,
        ranking_df=primary_df,
        docking_df=docking_csv,
    )

    # 4. Generate Plotly Charts (HTML snippet format)
    charts_html = _build_plotly_charts(
        dl_df=dl_csv,
        admet_df=admet_csv,
        docking_df=docking_csv,
        ranking_df=primary_df,
        total_screened=total_screened,
        passed_dl=passed_dl,
        passed_admet=passed_admet,
        total_docked=total_docked,
    )

    # 5. Build Top Hits Table
    table_rows_html = ""
    if primary_df is not None and not primary_df.empty:
        for idx, row in primary_df.head(15).iterrows():
            rank = row.get("rank", idx + 1)
            name = row.get("name", row.get("Compound", f"Mol_{idx+1}"))
            smiles = row.get("smiles", "")
            affinity = row.get("best_affinity", row.get("Docking", "N/A"))
            if isinstance(affinity, (int, float)):
                affinity = f"{affinity:.2f}"
            
            le = row.get("ligand_efficiency", "N/A")
            if isinstance(le, (int, float)) and not np.isnan(le):
                le = f"{le:.3f}"
            else:
                le = "—"

            lle = row.get("lipophilic_efficiency", "N/A")
            if isinstance(lle, (int, float)) and not np.isnan(lle):
                lle = f"{lle:.2f}"
            else:
                lle = "—"

            comp_score = row.get("composite_score", row.get("FinalScore", "N/A"))
            if isinstance(comp_score, (int, float)) and not np.isnan(comp_score):
                comp_score = f"{comp_score:.3f}"

            mw = row.get("MW", row.get("mw", "—"))
            if isinstance(mw, (int, float)) and not np.isnan(mw):
                mw = f"{mw:.1f}"

            logp = row.get("LogP", row.get("logp", "—"))
            if isinstance(logp, (int, float)) and not np.isnan(logp):
                logp = f"{logp:.2f}"

            svg_str = _smiles_to_svg(smiles, width=200, height=140)

            table_rows_html += f"""
            <tr>
              <td style="font-weight:600; text-align:center;">#{rank}</td>
              <td style="text-align:center; padding: 4px;">{svg_str}</td>
              <td style="font-weight:500;"><strong>{html.escape(str(name))}</strong><br><span style="font-family:monospace; font-size:11px; color:#718096;">{html.escape(str(smiles)[:30])}{'...' if len(str(smiles)) > 30 else ''}</span></td>
              <td style="color:#2b6cb0; font-weight:600; text-align:center;">{affinity}</td>
              <td style="text-align:center;">{le}</td>
              <td style="text-align:center;">{lle}</td>
              <td style="text-align:center;">{mw}</td>
              <td style="text-align:center;">{logp}</td>
              <td style="color:#2f855a; font-weight:700; text-align:center;">{comp_score}</td>
            </tr>
            """

    # 6. Build Quarantine Audit Section
    quarantine_html = ""
    if quarantine_records:
        quarantine_rows = "".join(f"""
        <tr>
          <td><span class="badge badge-warning">{html.escape(r.get('stage', 'pipeline'))}</span></td>
          <td style="font-weight:600;">{html.escape(str(r.get('mol_id', 'unknown')))}</td>
          <td style="color:#c53030;">{html.escape(str(r.get('reason', 'Unknown error')))}</td>
          <td style="font-family:monospace; font-size:11px;">{html.escape(str(r.get('smiles', '—'))[:40])}</td>
          <td style="font-size:12px; color:#718096;">{html.escape(str(r.get('timestamp', '')))}</td>
        </tr>
        """ for r in quarantine_records)

        quarantine_html = f"""
        <div class="card" style="margin-top: 30px;">
          <h3 style="color:#c53030; display:flex; align-items:center; gap:8px;">
            Quarantined Molecule Diagnostics ({len(quarantine_records)})
          </h3>
          <p style="color:#718096; font-size:14px;">The following compounds encountered formatting or calculation exceptions and were quarantined to prevent workflow termination.</p>
          <div style="overflow-x:auto;">
            <table>
              <thead>
                <tr>
                  <th>Stage</th>
                  <th>Molecule ID</th>
                  <th>Error / Failure Cause</th>
                  <th>SMILES</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {quarantine_rows}
              </tbody>
            </table>
          </div>
        </div>
        """

    # 6b. Build Interactive Collapsible Execution Logs Section
    logs_html = ""
    logs_dir = os.path.join(output_dir, "logs")
    log_files = []
    if os.path.exists(logs_dir):
        log_files = sorted(Path(logs_dir).glob("*.log"))
    elif os.path.exists("output/logs"):
        log_files = sorted(Path("output/logs").glob("*.log"))

    if log_files:
        # Group by module and select the latest log file for each
        latest_by_module: dict[str, Path] = {}
        for lf in log_files:
            mod_key = lf.stem.split("_202")[0]
            if mod_key not in latest_by_module or lf.stat().st_mtime > latest_by_module[mod_key].stat().st_mtime:
                latest_by_module[mod_key] = lf

        stage_order = [
            "protein_preparation", "ligand_preparation", "pocket_detection",
            "druglikeness", "admet", "docking", "lead_selection",
            "analogue_generation", "optimization_screening", "optimization_ranking"
        ]
        sorted_modules = sorted(
            latest_by_module.keys(),
            key=lambda k: stage_order.index(k) if k in stage_order else 999
        )

        log_cards = []
        for mod_key in sorted_modules:
            lf = latest_by_module[mod_key]
            mod_title = mod_key.replace("_", " ").title()
            try:
                with open(lf, "r", encoding="utf-8") as f:
                    log_text = f.read()
            except Exception:
                log_text = "(Could not read log file)"

            log_cards.append(f"""
            <details class="log-card" style="margin-bottom: 12px; border: 1px solid var(--border); border-radius: 0; overflow: hidden; background: #ffffff;">
              <summary style="padding: 12px 16px; background: #f8fafc; font-weight: 600; cursor: pointer; display: flex; justify-content: space-between; align-items: center; user-select: none;">
                <span style="display:flex; align-items:center; gap:8px;"><b>{html.escape(mod_title)} Log</b></span>
                <span style="font-size: 12px; color: var(--primary); font-weight: 500;">Click to view log</span>
              </summary>
              <div style="padding: 14px; background: #0f172a; color: #f1f5f9; font-family: 'Consolas', 'Courier New', monospace; font-size: 12px; max-height: 400px; overflow-y: auto;">
                <pre style="margin: 0; white-space: pre-wrap; word-break: break-all;">{html.escape(log_text)}</pre>
              </div>
            </details>
            """)

        logs_html = f"""
        <div class="card" style="margin-top: 30px;">
          <h3 style="display:flex; align-items:center; gap:8px;">
            Module Execution Logs ({len(log_cards)})
          </h3>
          <p style="color:#718096; font-size:14px; margin-bottom: 16px;">
            Detailed execution logs are minimized by default. Click any module header below to expand and review full technical outputs.
          </p>
          {''.join(log_cards)}
        </div>
        """

    # 7. Assemble Full HTML Document
    html_content = _build_html_template(
        report_title=report_title,
        workflow_type=workflow_type,
        timestamp=start_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        total_screened=total_screened,
        passed_dl=passed_dl,
        passed_admet=passed_admet,
        total_docked=total_docked,
        best_affinity=best_affinity,
        top_lead_name=top_lead_name,
        top_lead_score=top_lead_score,
        table_rows_html=table_rows_html,
        charts_html=charts_html,
        protein_pdb=protein_pdb_content,
        docked_ligands_data=docked_ligands_data,
        quarantine_html=quarantine_html,
        logs_html=logs_html,
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    # 8. Export structured JSON summary
    summary_json_path = os.path.join(reports_dir, "summary_report.json")
    summary_data = {
        "title": report_title,
        "workflow": workflow_type,
        "generated_at": start_time.isoformat() + "Z",
        "kpis": {
            "total_screened": total_screened,
            "passed_druglikeness": passed_dl,
            "passed_admet": passed_admet,
            "total_docked": total_docked,
            "best_affinity_kcal_mol": best_affinity,
            "top_lead_name": top_lead_name,
            "top_lead_score": top_lead_score,
        },
        "quarantined_count": len(quarantine_records),
        "report_html": output_file,
    }
    with open(summary_json_path, "w", encoding="utf-8") as jf:
        json.dump(summary_data, jf, indent=2)

    log_step(logger, "Report Generation", "completed", {"report": output_file})
    return output_file


# ──────────────────────────────────────────────────────────────
# Module Figure Generators (Called by druglikeness & admet)
# ──────────────────────────────────────────────────────────────

def generate_druglikeness_figures(df: pd.DataFrame, output_dir: str) -> list[str]:
    """Generate static/interactive figures for drug-likeness screening."""
    figs = []
    dl_dir = os.path.join(output_dir, "druglikeness")
    ensure_dir(dl_dir)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not df.empty and "MW" in df.columns and "LogP" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            status_colors = {"PASS": "#2f855a", "FAIL": "#c53030"}
            colors = [status_colors.get(s, "#718096") for s in df.get("Status", ["PASS"] * len(df))]
            ax.scatter(df["MW"], df["LogP"], c=colors, alpha=0.7, edgecolors="none", s=40)
            ax.set_xlabel("Molecular Weight (Da)")
            ax.set_ylabel("cLogP")
            ax.set_title("Drug-Likeness: MW vs cLogP")
            ax.grid(True, linestyle="--", alpha=0.5)

            out_path = os.path.join(dl_dir, "druglikeness_scatter.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
            figs.append(out_path)

    except Exception as e:
        logger.warning(f"Could not generate druglikeness plot: {e}")

    return figs


def generate_admet_figures(df: pd.DataFrame, output_dir: str) -> list[str]:
    """Generate static/interactive figures for ADMET screening."""
    figs = []
    admet_dir = os.path.join(output_dir, "admet")
    ensure_dir(admet_dir)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not df.empty and "Risk_Score" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            scores = [float(s) for s in df["Risk_Score"] if pd.notnull(s)]
            if scores:
                ax.hist(scores, bins=10, color="#805ad5", edgecolor="black", alpha=0.7)
                ax.set_xlabel("ADMET Risk Score (0-100)")
                ax.set_ylabel("Count")
                ax.set_title("ADMET Risk Score Distribution")
                ax.grid(True, linestyle="--", alpha=0.5)

                out_path = os.path.join(admet_dir, "admet_risk_histogram.png")
                plt.tight_layout()
                plt.savefig(out_path, dpi=150)
                plt.close(fig)
                figs.append(out_path)

    except Exception as e:
        logger.warning(f"Could not generate ADMET plot: {e}")

    return figs


def generate_summary_figures(counts: dict[str, int], output_dir: str) -> list[str]:
    """Generate pipeline summary attrition / funnel figures."""
    figs = []
    summary_dir = os.path.join(output_dir, "figures", "summary")
    ensure_dir(summary_dir)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if counts:
            stages = list(counts.keys())
            values = [int(v) for v in counts.values()]

            fig, ax = plt.subplots(figsize=(8, 4.5))
            colors = ["#3182ce", "#319795", "#38a169", "#d69e2e", "#e53e3e", "#805ad5"]
            bar_colors = [colors[i % len(colors)] for i in range(len(stages))]

            bars = ax.bar(stages, values, color=bar_colors, edgecolor="#2d3748", width=0.55)
            ax.set_ylabel("Molecule Count", fontsize=11, fontweight="bold")
            ax.set_title("Virtual Screening Pipeline Funnel / Stage Counts", fontsize=12, fontweight="bold")
            ax.grid(axis="y", linestyle="--", alpha=0.5)

            # Add value labels above bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=10,
                )

            plt.xticks(rotation=15, ha="right", fontsize=10)
            plt.tight_layout()

            out_path = os.path.join(summary_dir, "pipeline_funnel.png")
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
            figs.append(out_path)

    except Exception as e:
        logger.warning(f"Could not generate summary figures: {e}")

    return figs


def generate_docking_figures(df: pd.DataFrame, output_dir: str) -> list[str]:
    """Generate binding affinity distribution and correlation figures for molecular docking."""
    figs = []
    dock_dir = os.path.join(output_dir, "figures", "docking")
    ensure_dir(dock_dir)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not df.empty and "best_affinity" in df.columns:
            affinities = [float(a) for a in df["best_affinity"] if pd.notnull(a)]
            if affinities:
                # 1. Binding Affinity Histogram
                fig, ax = plt.subplots(figsize=(6.5, 4.2))
                ax.hist(affinities, bins=max(5, min(15, len(affinities))), color="#2b6cb0", edgecolor="#1a365d", alpha=0.75)
                mean_aff = float(np.mean(affinities))
                ax.axvline(mean_aff, color="#e53e3e", linestyle="--", label=f"Mean: {mean_aff:.2f}")
                ax.set_xlabel("Vina Binding Affinity (kcal/mol)", fontsize=11, fontweight="bold")
                ax.set_ylabel("Ligand Count", fontsize=11, fontweight="bold")
                ax.set_title("Docking Binding Affinity Distribution", fontsize=12, fontweight="bold")
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend(frameon=True)

                out_path = os.path.join(dock_dir, "affinity_distribution.png")
                plt.tight_layout()
                plt.savefig(out_path, dpi=150)
                plt.close(fig)
                figs.append(out_path)

                # 2. Affinity vs MW or Risk Score scatter if available
                if "MW" in df.columns or "Risk_Score" in df.columns:
                    fig, ax = plt.subplots(figsize=(6.5, 4.2))
                    x_col = "MW" if "MW" in df.columns else "Risk_Score"
                    x_label = "Molecular Weight (Da)" if x_col == "MW" else "ADMET Risk Score"
                    ax.scatter(df[x_col], df["best_affinity"], c="#319795", s=50, alpha=0.8, edgecolors="#234e52")
                    ax.set_xlabel(x_label, fontsize=11, fontweight="bold")
                    ax.set_ylabel("Binding Affinity (kcal/mol)", fontsize=11, fontweight="bold")
                    ax.set_title(f"Docking Affinity vs {x_col}", fontsize=12, fontweight="bold")
                    ax.grid(True, linestyle="--", alpha=0.5)

                    out_path2 = os.path.join(dock_dir, "affinity_scatter.png")
                    plt.tight_layout()
                    plt.savefig(out_path2, dpi=150)
                    plt.close(fig)
                    figs.append(out_path2)

    except Exception as e:
        logger.warning(f"Could not generate docking figures: {e}")

    return figs


def generate_lead_selection_figures(df: pd.DataFrame, output_dir: str) -> list[str]:
    """Generate prioritized lead ranking scores and property comparison figures."""
    figs = []
    lead_dir = os.path.join(output_dir, "figures", "lead_selection")
    ensure_dir(lead_dir)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not df.empty and "composite_score" in df.columns:
            top_df = df.head(10)
            names = [str(n)[:18] for n in top_df.get("name", [f"Lead_{i+1}" for i in range(len(top_df))])]
            scores = [float(s) for s in top_df["composite_score"]]

            # 1. Top Leads Composite Score Bar Chart
            fig, ax = plt.subplots(figsize=(7, 4.2))
            bars = ax.barh(range(len(names)), scores, color="#2f855a", edgecolor="#22543d", height=0.6)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=10)
            ax.invert_yaxis()
            ax.set_xlabel("Composite Priority Score (0-1)", fontsize=11, fontweight="bold")
            ax.set_title("Top Selected Lead Candidates", fontsize=12, fontweight="bold")
            ax.grid(axis="x", linestyle="--", alpha=0.5)

            for bar in bars:
                width = bar.get_width()
                ax.annotate(f"{width:.3f}", xy=(width, bar.get_y() + bar.get_height() / 2),
                            xytext=(4, 0), textcoords="offset points", ha="left", va="center", fontsize=9, fontweight="bold")

            out_path = os.path.join(lead_dir, "lead_ranking_scores.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
            figs.append(out_path)

    except Exception as e:
        logger.warning(f"Could not generate lead selection figures: {e}")

    return figs


def generate_optimization_figures(df: pd.DataFrame, parent_df: pd.DataFrame, output_dir: str) -> list[str]:
    """Generate lead optimization candidate ranking and improvement figures."""
    figs = []
    opt_dir = os.path.join(output_dir, "figures", "optimization")
    ensure_dir(opt_dir)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        score_col = "Total_Score" if "Total_Score" in df.columns else ("composite_score" if "composite_score" in df.columns else None)
        if not df.empty and score_col:
            top_df = df.head(10)
            names = [str(n)[:18] for n in top_df.get("name", [f"Opt_{i+1}" for i in range(len(top_df))])]
            scores = [float(s) for s in top_df[score_col]]

            fig, ax = plt.subplots(figsize=(7, 4.2))
            bars = ax.barh(range(len(names)), scores, color="#805ad5", edgecolor="#44337a", height=0.6)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=10)
            ax.invert_yaxis()
            ax.set_xlabel("Optimization Score", fontsize=11, fontweight="bold")
            ax.set_title("Top Optimized Analogues", fontsize=12, fontweight="bold")
            ax.grid(axis="x", linestyle="--", alpha=0.5)

            out_path = os.path.join(opt_dir, "optimization_scores.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
            figs.append(out_path)

    except Exception as e:
        logger.warning(f"Could not generate optimization figures: {e}")

    return figs


def _build_plotly_charts(dl_df, admet_df, docking_df, ranking_df, total_screened, passed_dl, passed_admet, total_docked) -> str:
    """Build responsive Plotly interactive chart HTML blocks with clear gridlines and no overlap."""
    charts = []
    try:
        import plotly.graph_objects as go

        # Chart 1: Screening Funnel
        funnel_stages = ["Input Library", "Drug-Likeness", "ADMET Pass", "Docked Hits"]
        funnel_counts = [
            max(total_screened, total_docked),
            passed_dl if passed_dl > 0 else total_docked,
            passed_admet if passed_admet > 0 else total_docked,
            total_docked,
        ]
        fig_funnel = go.Figure(go.Funnel(
            y=funnel_stages,
            x=funnel_counts,
            textinfo="value+percent initial",
            opacity=0.9,
            marker=dict(
                color=["#2b6cb0", "#3182ce", "#319795", "#38a169"],
                line=dict(width=1, color="#1a365d")
            ),
            connector=dict(line=dict(color="#cbd5e1", width=1.5))
        ))
        fig_funnel.update_layout(
            title=dict(text="Virtual Screening Attrition Funnel", font=dict(size=14, color="#1a202c", family="sans-serif")),
            autosize=True,
            margin=dict(l=110, r=25, t=50, b=25),
            height=340,
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        charts.append(fig_funnel.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True, "displayModeBar": False}))

        # Chart 2: Affinity Distribution
        if docking_df is not None and "best_affinity" in docking_df.columns and not docking_df.empty:
            fig_hist = go.Figure()
            aff_values = [float(a) for a in docking_df["best_affinity"] if pd.notnull(a)]
            fig_hist.add_trace(go.Histogram(
                x=aff_values,
                nbinsx=max(6, min(15, len(aff_values) * 2)),
                marker=dict(color="#2b6cb0", line=dict(color="#1a365d", width=1)),
                opacity=0.85,
                hovertemplate="Affinity: %{x:.2f} kcal/mol<br>Count: %{y}<extra></extra>"
            ))
            if aff_values:
                mean_aff = float(np.mean(aff_values))
                fig_hist.add_vline(
                    x=mean_aff,
                    line_width=2,
                    line_dash="dash",
                    line_color="#e53e3e",
                    annotation_text=f"Mean: {mean_aff:.2f}",
                    annotation_position="top right",
                    annotation_font=dict(color="#e53e3e", size=11)
                )
            fig_hist.update_layout(
                title=dict(text="Binding Affinity Distribution (AutoDock Vina)", font=dict(size=14, color="#1a202c", family="sans-serif")),
                xaxis_title="Binding Affinity (kcal/mol)",
                yaxis_title="Molecule Count",
                autosize=True,
                margin=dict(l=55, r=25, t=50, b=45),
                height=340,
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            fig_hist.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor="#e2e8f0",
                showline=True,
                linewidth=1,
                linecolor="#cbd5e1",
                mirror=True,
                zeroline=False,
            )
            fig_hist.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor="#e2e8f0",
                showline=True,
                linewidth=1,
                linecolor="#cbd5e1",
                mirror=True,
                zeroline=False,
            )
            charts.append(fig_hist.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True, "displayModeBar": False}))

        # Chart 3: Physicochemical Distributions (MW vs LogP) - Chemical Space with crisp grid & Lipinski rules
        if dl_df is not None and "MW" in dl_df.columns and "LogP" in dl_df.columns:
            fig_scatter = go.Figure()
            names = dl_df["name"] if "name" in dl_df.columns else [f"Mol_{i+1}" for i in range(len(dl_df))]
            fig_scatter.add_trace(go.Scatter(
                x=dl_df["MW"],
                y=dl_df["LogP"],
                mode="markers+text" if len(dl_df) <= 8 else "markers",
                text=names,
                textposition="top center",
                textfont=dict(size=10, color="#4a5568"),
                marker=dict(size=11, color="#805ad5", opacity=0.85, line=dict(width=1.5, color="#44337a")),
                hovertemplate="<b>%{text}</b><br>MW: %{x:.1f} Da<br>cLogP: %{y:.2f}<extra></extra>"
            ))

            # Lipinski Rule-of-5 threshold boundary lines
            fig_scatter.add_vline(
                x=500,
                line_width=1.5,
                line_dash="dot",
                line_color="#e53e3e",
                annotation_text="Lipinski MW ≤ 500",
                annotation_position="bottom right",
                annotation_font=dict(color="#e53e3e", size=10)
            )
            fig_scatter.add_hline(
                y=5.0,
                line_width=1.5,
                line_dash="dot",
                line_color="#e53e3e",
                annotation_text="Lipinski cLogP ≤ 5.0",
                annotation_position="top left",
                annotation_font=dict(color="#e53e3e", size=10)
            )

            fig_scatter.update_layout(
                title=dict(text="Chemical Space (MW vs cLogP)", font=dict(size=14, color="#1a202c", family="sans-serif")),
                xaxis_title="Molecular Weight (Da)",
                yaxis_title="cLogP",
                autosize=True,
                margin=dict(l=55, r=25, t=50, b=45),
                height=340,
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            fig_scatter.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor="#e2e8f0",
                showline=True,
                linewidth=1,
                linecolor="#cbd5e1",
                mirror=True,
                zeroline=True,
                zerolinecolor="#cbd5e1",
            )
            fig_scatter.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor="#e2e8f0",
                showline=True,
                linewidth=1,
                linecolor="#cbd5e1",
                mirror=True,
                zeroline=True,
                zerolinecolor="#cbd5e1",
            )
            charts.append(fig_scatter.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True, "displayModeBar": False}))

    except Exception as e:
        logger.warning(f"Could not build Plotly charts: {e}")

    if not charts:
        return "<p style='color:#718096;'>No chart data available.</p>"

    # Wrap in responsive grid with CSS containment to prevent any SVG overflow
    grid_html = "<div class='chart-grid'>"
    for c in charts:
        grid_html += f"<div class='chart-box'>{c}</div>"
    grid_html += "</div>"
    return grid_html


def _build_html_template(
    report_title, workflow_type, timestamp,
    total_screened, passed_dl, passed_admet, total_docked,
    best_affinity, top_lead_name, top_lead_score,
    table_rows_html, charts_html, protein_pdb, docked_ligands_data: dict, quarantine_html,
    logs_html: str = ""
) -> str:
    """Combine components into a modern responsive HTML report with advanced 3D visualizer and charts."""
    
    clean_protein = json.dumps(protein_pdb or "")
    docked_data_json = json.dumps(docked_ligands_data or {})
    has_3d = bool(protein_pdb or docked_ligands_data)

    viewer_section = ""
    if has_3d:
        viewer_section = f"""
        <div class="card" style="margin-top: 30px; border-radius: 0;">
          <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 12px;">
            <div>
              <h3 style="margin-bottom: 4px;">Interactive 3D Protein–Ligand Complex</h3>
              <p style="color:#718096; font-size:13px; margin: 0;">
                Rotate (Left Click + Drag), Pan (Right Click + Drag), Zoom (Scroll). Inspect docked ligand poses in the binding cavity (3Dmol.js)
              </p>
            </div>
            <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
              <label for="ligand_select" style="font-size: 13px; font-weight: 600; color: #4a5568;">Ligand:</label>
              <select id="ligand_select" onchange="onLigandChange()" style="padding: 5px 10px; border-radius: 0; border: 1px solid var(--border); font-size: 13px; font-weight: 600; background: #ffffff; color: #1a202c; cursor: pointer;">
              </select>
              <label for="pose_select" style="font-size: 13px; font-weight: 600; color: #4a5568;">Pose:</label>
              <select id="pose_select" onchange="onPoseChange()" style="padding: 5px 10px; border-radius: 0; border: 1px solid var(--border); font-size: 13px; font-weight: 600; background: #ffffff; color: #1a202c; cursor: pointer;">
              </select>
              <button class="btn btn-primary" onclick="focusLigand()" title="Focus camera directly on the docked ligand pose">🎯 Focus Ligand</button>
              <button class="btn" onclick="reset3DView()" title="Reset 3D camera orientation">🔄 Reset View</button>
              <button id="btn_bg_mode" class="btn" onclick="toggleBackground()" title="Toggle Light / Dark Background">Light BG</button>
            </div>
          </div>

          <!-- Active Pose Metrics Badges -->
          <div id="pose_metrics_container" style="display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 14px;">
            <span id="badge_affinity" class="badge" style="background: #ebf8ff; color: #2b6cb0; font-size: 13px; font-weight: 700; padding: 6px 12px; border: 1px solid #bee3f8; border-radius: 0;">
              ΔG: -- kcal/mol
            </span>
            <span id="badge_pose_num" class="badge" style="background: #edf2f7; color: #4a5568; font-size: 12px; padding: 6px 10px; border-radius: 0;">
              Pose 1 / 1
            </span>
            <span id="badge_rmsd" class="badge" style="background: #edf2f7; color: #4a5568; font-size: 12px; padding: 6px 10px; border-radius: 0;">
              RMSD: 0.00 Å
            </span>
            <span id="badge_le" class="badge" style="background: #def7ec; color: #2f855a; font-size: 12px; padding: 6px 10px; display: none; border-radius: 0;">
              LE: --
            </span>
          </div>

          <!-- Viewer Toolbar Controls -->
          <div style="display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 14px; padding: 12px 16px; background: #f7fafc; border: 1px solid var(--border); font-size: 13px; align-items: center;">
            <div>
              <strong>Protein View:</strong>
              <select id="proteinStyle" class="btn" style="padding: 4px 8px;" onchange="applyStyles()">
                <option value="cartoon">Cartoon</option>
                <option value="surface">Surface</option>
                <option value="cartoon_surface">Cartoon + Surface</option>
                <option value="ribbon">Ribbon</option>
                <option value="stick">Sticks</option>
                <option value="line">Wireframe</option>
              </select>
              <select id="proteinColor" class="btn" style="padding: 4px 8px;" onchange="applyStyles()">
                <option value="spectrum">Spectrum</option>
                <option value="cyan">Cyan</option>
                <option value="ss">Sec. Structure</option>
                <option value="b">B-Factor</option>
              </select>
            </div>
            <div>
              <strong>Ligand View:</strong>
              <select id="ligandStyle" class="btn" style="padding: 4px 8px;" onchange="applyStyles()">
                <option value="stick">Sticks</option>
                <option value="ballAndStick">Ball &amp; Stick</option>
                <option value="sphere">Spacefill</option>
                <option value="surface">Surface</option>
              </select>
              <select id="ligandColor" class="btn" style="padding: 4px 8px;" onchange="applyStyles()">
                <option value="greenCarbon">Green Carbon</option>
                <option value="orangeCarbon">Orange Carbon</option>
                <option value="element">Element / CPK</option>
                <option value="spectrum">Spectrum</option>
              </select>
            </div>
            <div style="display: flex; gap: 12px; align-items: center; margin-left: auto; flex-wrap: wrap;">
              <label style="cursor: pointer; display: flex; align-items: center; gap: 4px;"><input type="checkbox" id="chkProtein" checked onchange="applyStyles()"> Show Protein</label>
              <label style="cursor: pointer; display: flex; align-items: center; gap: 4px;"><input type="checkbox" id="chkLigand" checked onchange="applyStyles()"> Show Ligand</label>
              <label style="cursor: pointer; display: flex; align-items: center; gap: 4px;"><input type="checkbox" id="chk_pocket_residues" checked onchange="togglePocketResidues()"> Pocket (&lt;5Å)</label>
              <label style="cursor: pointer; display: flex; align-items: center; gap: 4px;"><input type="checkbox" id="chkHbonds" onchange="applyStyles()"> H-Bonds</label>
            </div>
          </div>

          <!-- 3Dmol Canvas Container with Interactive HUD -->
          <div style="position: relative; width: 100%; height: 520px; overflow: hidden; background: #1a202c; border: 1px solid var(--border);">
            <div id="pose3d_viewer" style="width: 100%; height: 100%;"></div>
            <div id="sm_interaction_hud" style="display: none; position: absolute; bottom: 12px; left: 16px; right: 16px; background: rgba(15, 23, 42, 0.92); backdrop-filter: blur(6px); border: 1px solid #334155; padding: 10px 16px; border-radius: 4px; color: #f8fafc; font-size: 13px; pointer-events: none; z-index: 50; transition: all 0.15s ease; box-shadow: 0 4px 12px rgba(0,0,0,0.35);">
              <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                <div id="sm_hud_main" style="display: flex; align-items: center; gap: 10px;"></div>
                <div id="sm_hud_badge"></div>
              </div>
            </div>
          </div>
        </div>

        <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"></script>
        <script>
          let glviewer = null;
          let protModel = null;
          let ligModel = null;
          let isDarkMode = true;
          let activePocketResis = new Set();

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

            let isReceptor = false;
            if (protModel && atom.model === protModel.getID()) {{
              isReceptor = true;
            }} else if (ligModel && atom.model === ligModel.getID()) {{
              isReceptor = false;
            }} else if (atom.resn && AA_NAMES[atom.resn]) {{
              isReceptor = true;
            }}

            let resAtoms = [];
            let partnerAtoms = [];
            let resStr = "";
            let aaFullName = "";

            if (isReceptor) {{
              let resi = atom.resi;
              let resn = atom.resn || 'UNK';
              let chain = atom.chain || '';
              aaFullName = AA_NAMES[resn] ? `${{resn}} (${{AA_NAMES[resn]}})` : resn;
              resStr = `Receptor ${{chain ? chain + ':' : ''}}${{resn}}${{resi}}`;

              resAtoms = protModel ? (protModel.selectedAtoms({{chain: chain, resi: resi}}) || []) : [];
              if (!resAtoms.length && protModel) {{
                resAtoms = protModel.selectedAtoms({{resi: resi}}) || [];
              }}
              if (!resAtoms.length) resAtoms = [atom];

              partnerAtoms = (ligModel !== null) ? (ligModel.selectedAtoms({{}}) || []) : [];

              for (let i = 0; i < resAtoms.length; i++) {{
                let a = resAtoms[i];
                let aElem = (a.elem || (a.atom ? a.atom.replace(/[^a-zA-Z]/g, '').slice(0, 1) : '')).toUpperCase();
                if (aElem === 'H') continue;

                let sph = glviewer.addSphere({{
                  center: {{x: a.x, y: a.y, z: a.z}},
                  radius: 0.28,
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
                      radius: 0.16,
                      color: '#f6e05e',
                      opacity: 0.95,
                      fromCap: 1,
                      toCap: 1
                    }});
                    if (cyl) hoverShapes.push(cyl);
                  }}
                }}
              }}
            }} else {{
              let atomName = atom.atom || atom.elem || 'LIG';
              let ser = atom.serial ? `#${{atom.serial}}` : '';
              resStr = `Ligand Atom ${{atomName}} ${{ser}}`.trim();
              aaFullName = atom.elem || "Ligand";
              resAtoms = [atom];
              partnerAtoms = (protModel !== null) ? (protModel.selectedAtoms({{}}) || []) : [];

              let sph = glviewer.addSphere({{
                center: {{x: atom.x, y: atom.y, z: atom.z}},
                radius: 0.44,
                color: '#f6e05e',
                opacity: 0.95
              }});
              if (sph) hoverShapes.push(sph);
            }}

            let inter = classifyInteraction(resAtoms, partnerAtoms);

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

            let hudElem = document.getElementById("sm_interaction_hud");
            if (hudElem) hudElem.style.display = "block";

            let hudMain = document.getElementById("sm_hud_main");
            let hudBadge = document.getElementById("sm_hud_badge");
            if (hudMain) {{
              let partnerDesc = inter.partnerAtom 
                ? `${{isReceptor ? 'Ligand' : 'Receptor'}} ${{inter.partnerAtom.resn || ''}}${{inter.partnerAtom.resi ? ':' + inter.partnerAtom.resi : ''}} (${{inter.partnerAtom.atom || inter.partnerAtom.elem}})`
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

            let hudElem = document.getElementById("sm_interaction_hud");
            if (hudElem) hudElem.style.display = "none";
            let hudMain = document.getElementById("sm_hud_main");
            let hudBadge = document.getElementById("sm_hud_badge");
            if (hudMain) {{
              hudMain.innerHTML = "";
            }}
            if (hudBadge) {{
              hudBadge.innerHTML = "";
            }}

            if (glviewer) glviewer.render();
          }}

          const allDockedData = {docked_data_json};
          const protData = {clean_protein};

          document.addEventListener("DOMContentLoaded", function() {{
            init3DViewer();
          }});

          function init3DViewer() {{
            let container = document.getElementById("pose3d_viewer");
            if (!container || !window.$3Dmol) return;

            let config = {{ backgroundColor: "#1a202c" }};
            glviewer = $3Dmol.createViewer(container, config);
            glviewer.setHoverDuration(20);

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

            if (protData && protData.length > 20) {{
              protModel = glviewer.addModel(protData, "pdb");
            }}

            let ligSelect = document.getElementById("ligand_select");
            if (ligSelect) {{
              ligSelect.innerHTML = "";
              let ligNames = Object.keys(allDockedData);
              if (ligNames.length > 0) {{
                ligNames.forEach(function(name) {{
                  let ldata = allDockedData[name];
                  let affStr = (ldata.best_affinity !== undefined && ldata.best_affinity !== null) 
                    ? (" (" + Number(ldata.best_affinity).toFixed(2) + " kcal/mol)") 
                    : "";
                  let opt = document.createElement("option");
                  opt.value = name;
                  opt.textContent = name + affStr;
                  ligSelect.appendChild(opt);
                }});
                populatePoseSelect(ligNames[0]);
              }} else {{
                let opt = document.createElement("option");
                opt.textContent = "No docked poses found";
                ligSelect.appendChild(opt);
                applyStyles();
                glviewer.zoomTo();
                glviewer.render();
              }}
            }}
          }}

          function populatePoseSelect(ligName) {{
            let poseSelect = document.getElementById("pose_select");
            if (!poseSelect) return;
            poseSelect.innerHTML = "";

            let ldata = allDockedData[ligName];
            if (!ldata || !ldata.poses || ldata.poses.length === 0) {{
              let opt = document.createElement("option");
              opt.textContent = "Pose 1";
              opt.value = "0";
              poseSelect.appendChild(opt);
            }} else {{
              ldata.poses.forEach(function(p, idx) {{
                let opt = document.createElement("option");
                opt.value = idx.toString();
                let affTxt = (p.affinity !== null && p.affinity !== undefined) 
                  ? (" (ΔG: " + Number(p.affinity).toFixed(2) + " kcal/mol)") 
                  : "";
                opt.textContent = "Pose " + (idx + 1) + affTxt;
                poseSelect.appendChild(opt);
              }});
            }}

            loadActivePose();
          }}

          function onLigandChange() {{
            let ligSelect = document.getElementById("ligand_select");
            if (!ligSelect) return;
            populatePoseSelect(ligSelect.value);
          }}

          function onPoseChange() {{
            loadActivePose();
          }}

          function loadActivePose() {{
            if (!glviewer) return;
            clearHoverHighlights();
            let ligSelect = document.getElementById("ligand_select");
            let poseSelect = document.getElementById("pose_select");
            if (!ligSelect || !poseSelect) return;

            let ligName = ligSelect.value;
            let poseIdx = parseInt(poseSelect.value, 10);
            let ldata = allDockedData[ligName];
            if (!ldata || !ldata.poses || !ldata.poses[poseIdx]) return;

            let pose = ldata.poses[poseIdx];

            if (ligModel !== null) {{
              glviewer.removeModel(ligModel);
              ligModel = null;
            }}

            if (pose.data && pose.data.length > 20) {{
              let fmt = pose.format || "sdf";
              ligModel = glviewer.addModel(pose.data, fmt);
            }}

            let affBadge = document.getElementById("badge_affinity");
            if (affBadge) {{
              let affVal = (pose.affinity !== null && pose.affinity !== undefined) 
                ? ("ΔG: " + Number(pose.affinity).toFixed(2) + " kcal/mol") 
                : "ΔG: N/A";
              affBadge.innerText = affVal;
            }}

            let poseBadge = document.getElementById("badge_pose_num");
            if (poseBadge) {{
              poseBadge.innerText = "Pose " + (poseIdx + 1) + " / " + ldata.poses.length;
            }}

            let rmsdBadge = document.getElementById("badge_rmsd");
            if (rmsdBadge) {{
              let rmsdVal = (pose.rmsd_lb !== undefined && pose.rmsd_lb !== null) 
                ? Number(pose.rmsd_lb).toFixed(2) 
                : "0.00";
              rmsdBadge.innerText = "RMSD: " + rmsdVal + " Å";
            }}

            let leBadge = document.getElementById("badge_le");
            if (leBadge) {{
              if (ldata.le) {{
                leBadge.innerText = "LE: " + ldata.le;
                leBadge.style.display = "inline-block";
              }} else {{
                leBadge.style.display = "none";
              }}
            }}

            applyStyles();
            focusLigand();
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
            }} else if (color === "cyan") {{
              colOpt = {{ color: "cyan" }};
            }} else {{
              colOpt = {{ colorscheme: "spectrum" }};
            }}

            if (style === "cartoon") {{
              styleObj.cartoon = colOpt;
            }} else if (style === "ribbon") {{
              styleObj.cartoon = Object.assign({{ ribbon: true, thickness: 0.35, style: "oval" }}, colOpt);
            }} else if (style === "stick" || style === "sticks") {{
              styleObj.stick = colOpt;
            }} else if (style === "line" || style === "wireframe") {{
              styleObj.line = colOpt;
            }} else if (style === "surface" || style === "cartoon_surface") {{
              styleObj.cartoon = Object.assign({{ opacity: 0.25 }}, colOpt);
            }}
            styleObj.clicksphere = {{ radius: 1.2 }};
            return styleObj;
          }}

          function getLigandStyleObj(style, color) {{
            let styleObj = {{}};
            let colOpt = {{}};
            if (color === "orangeCarbon") {{
              colOpt = {{ colorscheme: "orangeCarbon" }};
            }} else if (color === "element") {{
              colOpt = {{ colorscheme: "rasmol" }};
            }} else if (color === "spectrum") {{
              colOpt = {{ colorscheme: "spectrum" }};
            }} else {{
              colOpt = {{ colorscheme: "greenCarbon" }};
            }}

            if (style === "stick" || style === "sticks") {{
              styleObj.stick = Object.assign({{ radius: 0.25 }}, colOpt);
            }} else if (style === "ballAndStick" || style === "ball_stick") {{
              styleObj.stick = Object.assign({{ radius: 0.18 }}, colOpt);
              styleObj.sphere = Object.assign({{ scale: 0.28 }}, colOpt);
            }} else if (style === "sphere" || style === "spheres") {{
              styleObj.sphere = Object.assign({{ scale: 0.85 }}, colOpt);
            }} else if (style === "surface") {{
              styleObj.stick = Object.assign({{ radius: 0.2 }}, colOpt);
            }}
            styleObj.clicksphere = {{ radius: 1.2 }};
            return styleObj;
          }}

          function applyStyles() {{
            if (!glviewer) return;

            clearHoverHighlights();
            // Synchronously remove and dispose all previous surfaces & shapes
            glviewer.removeAllSurfaces();
            glviewer.removeAllShapes();

            let protStyleElem = document.getElementById("proteinStyle");
            let protColorElem = document.getElementById("proteinColor");
            let ligStyleElem = document.getElementById("ligandStyle");
            let ligColorElem = document.getElementById("ligandColor");
            let chkProt = document.getElementById("chkProtein");
            let chkLig = document.getElementById("chkLigand");
            let chkPocket = document.getElementById("chk_pocket_residues");
            let chkHbonds = document.getElementById("chkHbonds");

            let protStyle = protStyleElem ? protStyleElem.value : "cartoon";
            let protColor = protColorElem ? protColorElem.value : "spectrum";
            let ligStyle = ligStyleElem ? ligStyleElem.value : "stick";
            let ligColor = ligColorElem ? ligColorElem.value : "greenCarbon";
            let showProt = chkProt ? chkProt.checked : true;
            let showLig = chkLig ? chkLig.checked : true;
            let showPocket = chkPocket ? chkPocket.checked : true;
            let showHbonds = chkHbonds ? chkHbonds.checked : false;

            // Compute pocket residues within 5Å of ligand
            activePocketResis = new Set();
            let pocketResis = activePocketResis;
            let protAtoms = (protModel !== null) ? (protModel.selectedAtoms({{}}) || []) : [];
            let ligAtoms = (ligModel !== null) ? (ligModel.selectedAtoms({{}}) || []) : [];

            if (ligAtoms.length > 0 && protAtoms.length > 0) {{
              for (let i = 0; i < ligAtoms.length; i++) {{
                let la = ligAtoms[i];
                for (let j = 0; j < protAtoms.length; j++) {{
                  let pa = protAtoms[j];
                  let dx = la.x - pa.x, dy = la.y - pa.y, dz = la.z - pa.z;
                  if (dx > 5.5 || dx < -5.5 || dy > 5.5 || dy < -5.5 || dz > 5.5 || dz < -5.5) continue;
                  if (dx * dx + dy * dy + dz * dz <= 25.0) {{
                    pocketResis.add(pa.resi);
                  }}
                }}
              }}
            }}

            // 1. Protein rendering
            if (protModel !== null) {{
              protModel.setStyle({{}}, {{}});
              if (showProt) {{
                let pObj = getProteinStyleObj(protStyle, protColor);
                protModel.setStyle({{}}, pObj);

                if (protStyle === "surface" || protStyle === "cartoon_surface") {{
                  let surfColor = protColor === "cyan" ? "cyan" : (isDarkMode ? "white" : "#cbd5e1");
                  let surfSel = (pocketResis.size > 0) ? {{model: protModel, resi: Array.from(pocketResis)}} : {{model: protModel}};
                  glviewer.addSurface($3Dmol.SurfaceType.VDW, {{opacity: protStyle === "surface" ? 0.65 : 0.35, color: surfColor}}, surfSel)
                    .then(function() {{ glviewer.render(); }})
                    .catch(function(e) {{ console.warn("Protein surface error:", e); }});
                }}

                if (showPocket && pocketResis.size > 0 && protStyle !== "stick" && protStyle !== "surface") {{
                  let resiList = Array.from(pocketResis);
                  let comboObj = Object.assign({{}}, pObj, {{
                    stick: {{ colorscheme: "amino", radius: 0.20 }}
                  }});
                  protModel.setStyle({{ resi: resiList }}, comboObj);
                }}
              }}
            }}

            // 2. Ligand rendering
            if (ligModel !== null) {{
              ligModel.setStyle({{}}, {{}});
              if (showLig) {{
                let lObj = getLigandStyleObj(ligStyle, ligColor);
                ligModel.setStyle({{}}, lObj);

                if (ligStyle === "surface") {{
                  let pScheme = (ligColor === "orangeCarbon") ? "orangeCarbon" : "greenCarbon";
                  glviewer.addSurface($3Dmol.SurfaceType.MS, {{opacity: 0.75, colorscheme: pScheme}}, {{model: ligModel}})
                    .then(function() {{ glviewer.render(); }})
                    .catch(function(e) {{ console.warn("Ligand surface error:", e); }});
                }}
              }}
            }}

            // 3. Hydrogen bonds
            if (showHbonds && showProt && showLig && protModel !== null && ligModel !== null) {{
              let donAcc = ['N', 'O', 'S', 'F'];
              for (let i = 0; i < protAtoms.length; i++) {{
                let pa = protAtoms[i];
                if (!donAcc.includes(pa.elem)) continue;
                for (let j = 0; j < ligAtoms.length; j++) {{
                  let la = ligAtoms[j];
                  if (!donAcc.includes(la.elem)) continue;
                  let dx = pa.x - la.x, dy = pa.y - la.y, dz = pa.z - la.z;
                  let d2 = dx*dx + dy*dy + dz*dz;
                  if (d2 >= 5.76 && d2 <= 12.25) {{
                    glviewer.addCylinder({{
                      start: {{x: pa.x, y: pa.y, z: pa.z}},
                      end: {{x: la.x, y: la.y, z: la.z}},
                      radius: 0.08,
                      dashed: true,
                      color: "#f6e05e",
                      fromCap: 1,
                      toCap: 1
                    }});
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

          function focusLigand() {{
            if (!glviewer) return;
            if (ligModel !== null) {{
              glviewer.zoomTo({{model: ligModel}}, 800);
            }} else if (protModel !== null) {{
              glviewer.zoomTo({{model: protModel}}, 800);
            }} else {{
              glviewer.zoomTo();
            }}
            glviewer.render();
          }}

          function zoomFullProtein() {{
            reset3DView();
          }}

          function reset3DView() {{
            if (!glviewer) return;
            glviewer.zoomTo();
            glviewer.render();
          }}

          function togglePocketResidues() {{
            applyStyles();
          }}

          function toggleBackground() {{
            if (!glviewer) return;
            isDarkMode = !isDarkMode;
            let bg = isDarkMode ? "#1a202c" : "#ffffff";
            glviewer.setBackgroundColor(bg);

            let btn = document.getElementById("btn_bg_mode");
            if (btn) {{
              btn.innerHTML = isDarkMode ? "Light BG" : "Dark BG";
            }}
            let container = document.getElementById("pose3d_viewer");
            if (container) {{
              container.style.backgroundColor = bg;
            }}
            applyStyles();
          }}

          function setProteinStyle(styleName, btn) {{
            let elem = document.getElementById("proteinStyle");
            if (elem) {{ elem.value = styleName; }}
            applyStyles();
          }}

          function setLigandStyle(styleName, btn) {{
            let elem = document.getElementById("ligandStyle");
            if (elem) {{ elem.value = styleName; }}
            applyStyles();
          }}

          function cleanupWebGL() {{
            if (glviewer) {{
              try {{
                glviewer.removeAllSurfaces();
                glviewer.removeAllModels();
                glviewer.removeAllShapes();
                glviewer.removeAllLabels();
                glviewer.clear();

                let container = document.getElementById("pose3d_viewer");
                if (container) {{
                  let canvases = container.querySelectorAll("canvas");
                  canvases.forEach(function(canvas) {{
                    let gl = canvas.getContext("webgl") || canvas.getContext("webgl2") || canvas.getContext("experimental-webgl");
                    if (gl) {{
                      let loseExt = gl.getExtension("WEBGL_lose_context");
                      if (loseExt) {{
                        loseExt.loseContext();
                      }}
                    }}
                    if (canvas.parentNode) {{
                      canvas.parentNode.removeChild(canvas);
                    }}
                  }});
                }}
                glviewer = null;
              }} catch (e) {{
                console.warn("WebGL cleanup error:", e);
              }}
            }}
          }}

          window.addEventListener("beforeunload", cleanupWebGL);
          window.addEventListener("pagehide", cleanupWebGL);
          window.addEventListener("unload", cleanupWebGL);
        </script>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{html.escape(report_title)}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
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
      border-radius: 0 !important;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
    }}
    .container {{
      max-width: 1280px;
      margin: 0 auto;
    }}
    header {{
      background: var(--card-bg);
      padding: 24px 32px;
      border-radius: 0;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06);
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }}
    h1 {{ font-size: 24px; font-weight: 700; color: #1a202c; }}
    h3 {{ font-size: 18px; font-weight: 600; margin-bottom: 12px; }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 0;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .badge-primary {{ background: var(--primary-light); color: var(--primary); }}
    .badge-success {{ background: #def7ec; color: var(--success); }}
    .badge-warning {{ background: #feecdc; color: var(--warning); }}

    /* KPI Cards */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .kpi-card {{
      background: var(--card-bg);
      padding: 20px;
      border-radius: 0;
      border: 1px solid var(--border);
      box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }}
    .kpi-title {{ font-size: 12px; font-weight: 600; color: #718096; text-transform: uppercase; margin-bottom: 6px; }}
    .kpi-value {{ font-size: 22px; font-weight: 700; color: #1a202c; }}

    /* Cards */
    .card {{
      background: var(--card-bg);
      padding: 24px;
      border-radius: 0;
      border: 1px solid var(--border);
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
      margin-bottom: 24px;
    }}

    /* Chart Grid & Containers (Prevents overlapping) */
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 20px;
      width: 100%;
      box-sizing: border-box;
    }}
    .chart-box {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 0;
      padding: 14px;
      min-width: 0;
      overflow: hidden;
      position: relative;
      box-sizing: border-box;
    }}
    .chart-box .plotly-graph-div {{
      width: 100% !important;
      height: 340px !important;
    }}

    /* Table */
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th {{
      background: #edf2f7;
      color: #4a5568;
      font-weight: 600;
      text-align: left;
      padding: 12px 14px;
      border-bottom: 2px solid var(--border);
    }}
    td {{
      padding: 10px 14px;
      border-bottom: 1px solid var(--border);
      vertical-align: middle;
    }}
    tr:hover {{ background-color: #f7fafc; }}

    .btn {{
      background: #edf2f7;
      color: #2d3748;
      border: 1px solid var(--border);
      padding: 8px 14px;
      border-radius: 0;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s ease;
    }}
    .btn:hover {{ background: #e2e8f0; }}
    .btn-sm {{
      padding: 5px 11px;
      font-size: 12px;
      font-weight: 600;
    }}
    .btn-active {{
      background: var(--primary) !important;
      color: #ffffff !important;
      border-color: var(--primary) !important;
    }}
    .btn-primary {{
      background: var(--primary) !important;
      color: #ffffff !important;
      border-color: var(--primary) !important;
    }}
    .btn-primary:hover {{
      background: #23568c !important;
    }}
    .style-btn-group {{
      display: inline-flex;
      border-radius: 0;
      overflow: hidden;
      border: 1px solid var(--border);
    }}
    .style-btn-group .btn {{
      border-radius: 0;
      border: none;
      border-right: 1px solid var(--border);
      margin: 0;
    }}
    .style-btn-group .btn:last-child {{
      border-right: none;
    }}

    footer {{
      text-align: center;
      color: #a0aec0;
      font-size: 13px;
      margin-top: 40px;
      padding-bottom: 20px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <span class="badge badge-primary">{html.escape(workflow_type)}</span>
        <h1 style="margin-top: 6px;">{html.escape(report_title)}</h1>
        <div style="font-size: 13px; color: #718096; margin-top: 4px;">Run timestamp: {timestamp}</div>
      </div>
      <div>
        <span class="badge badge-success">Completed Successfully</span>
      </div>
    </header>

    <!-- KPI Metric Cards -->
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-title">Screened Ligands</div>
        <div class="kpi-value">{total_screened}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Passed Drug-Likeness</div>
        <div class="kpi-value">{passed_dl}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Passed ADMET</div>
        <div class="kpi-value">{passed_admet}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Docked Poses</div>
        <div class="kpi-value">{total_docked}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Top Affinity</div>
        <div class="kpi-value" style="color: var(--primary);">{best_affinity}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Best Candidate</div>
        <div class="kpi-value" style="color: var(--success); font-size:18px;">{html.escape(str(top_lead_name))}</div>
      </div>
    </div>

    <!-- Charts Section -->
    <div class="card">
      <h3>Pipeline Analytics & Chemical Space</h3>
      {charts_html}
    </div>

    <!-- 3D Viewer Section -->
    {viewer_section}

    <!-- Top Candidates Table -->
    <div class="card">
      <h3>Top Identified Lead Candidates</h3>
      <p style="color:#718096; font-size:14px; margin-bottom:16px;">
        Ranked candidates prioritized by AutoDock Vina binding affinity, Ligand Efficiency (LE), Lipophilic Efficiency (LLE), and composite multi-parameter score.
      </p>
      <div style="overflow-x: auto;">
        <table>
          <thead>
            <tr>
              <th style="text-align:center;">Rank</th>
              <th style="text-align:center;">Structure (2D)</th>
              <th>Compound Name / SMILES</th>
              <th style="text-align:center;">Affinity (kcal/mol)</th>
              <th style="text-align:center;">LE (kcal/mol/atom)</th>
              <th style="text-align:center;">LLE (LipE)</th>
              <th style="text-align:center;">MW (Da)</th>
              <th style="text-align:center;">cLogP</th>
              <th style="text-align:center;">Score</th>
            </tr>
          </thead>
          <tbody>
            {table_rows_html if table_rows_html else "<tr><td colspan='9' style='text-align:center; padding:30px; color:#718096;'>No candidate records found in output directory.</td></tr>"}
          </tbody>
        </table>
      </div>
    </div>

    <!-- Quarantine Section -->
    {quarantine_html}

    <!-- Collapsible Execution Logs Section -->
    {logs_html}

    <footer>
      Generated automatically by <strong>DrugLab v0.2.0</strong> — Lightweight Computational Drug Discovery Platform.
    </footer>
  </div>

  <script>
    // Responsive Plotly resize handler
    window.addEventListener("resize", function() {{
      document.querySelectorAll(".plotly-graph-div").forEach(function(el) {{
        if (window.Plotly && el) {{
          Plotly.Plots.resize(el);
        }}
      }});
    }});
    setTimeout(function() {{
      window.dispatchEvent(new Event("resize"));
    }}, 250);
  </script>
</body>
</html>
"""
