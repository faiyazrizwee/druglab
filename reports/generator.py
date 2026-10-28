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
import json
import base64
import html
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

    # Locate top ligand pose
    ligand_pose_content = ""
    pose_candidates = [
        os.path.join(output_dir, "lead_selection", "best_lead.sdf"),
        os.path.join(output_dir, "ranking", "top_candidates.sdf"),
        os.path.join(output_dir, "docking", "best_poses.sdf"),
    ]
    for psc in pose_candidates:
        if os.path.isfile(psc):
            try:
                with open(psc, "r", encoding="utf-8", errors="replace") as lf:
                    # Take first record up to $$$$
                    all_text = lf.read()
                    first_mol = all_text.split("$$$$")[0] + "$$$$\n"
                    ligand_pose_content = first_mol
                break
            except Exception:
                pass

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
            ⚠️ Quarantined Molecule Diagnostics ({len(quarantine_records)})
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
            <details class="log-card" style="margin-bottom: 12px; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: #ffffff;">
              <summary style="padding: 12px 16px; background: #f8fafc; font-weight: 600; cursor: pointer; display: flex; justify-content: space-between; align-items: center; user-select: none;">
                <span style="display:flex; align-items:center; gap:8px;">📄 <b>{html.escape(mod_title)} Log</b></span>
                <span style="font-size: 12px; color: var(--primary); font-weight: 500;">Click to view log ▾</span>
              </summary>
              <div style="padding: 14px; background: #0f172a; color: #f1f5f9; font-family: 'Consolas', 'Courier New', monospace; font-size: 12px; max-height: 400px; overflow-y: auto;">
                <pre style="margin: 0; white-space: pre-wrap; word-break: break-all;">{html.escape(log_text)}</pre>
              </div>
            </details>
            """)

        logs_html = f"""
        <div class="card" style="margin-top: 30px;">
          <h3 style="display:flex; align-items:center; gap:8px;">
            📋 Module Execution Logs ({len(log_cards)})
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
        ligand_sdf=ligand_pose_content,
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
    """Build Plotly chart interactive HTML blocks."""
    charts = []
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

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
            marker=dict(color=["#3182ce", "#4299e1", "#48bb78", "#38a169"])
        ))
        fig_funnel.update_layout(
            title="Virtual Screening Attrition Funnel",
            margin=dict(l=20, r=20, t=40, b=20),
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        charts.append(fig_funnel.to_html(full_html=False, include_plotlyjs="cdn"))

        # Chart 2: Affinity Distribution or Affinity vs Score
        if docking_df is not None and "best_affinity" in docking_df.columns and not docking_df.empty:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=docking_df["best_affinity"],
                nbinsx=15,
                marker_color="#2b6cb0",
                opacity=0.85
            ))
            fig_hist.update_layout(
                title="Binding Affinity Distribution (AutoDock Vina)",
                xaxis_title="Binding Affinity (kcal/mol)",
                yaxis_title="Molecule Count",
                margin=dict(l=20, r=20, t=40, b=20),
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            charts.append(fig_hist.to_html(full_html=False, include_plotlyjs=False))

        # Chart 3: Physicochemical Distributions (MW vs LogP)
        if dl_df is not None and "MW" in dl_df.columns and "LogP" in dl_df.columns:
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=dl_df["MW"],
                y=dl_df["LogP"],
                mode="markers",
                marker=dict(size=8, color="#805ad5", opacity=0.7),
                text=dl_df["name"] if "name" in dl_df.columns else None,
                hoverinfo="text+x+y"
            ))
            fig_scatter.update_layout(
                title="Chemical Space (MW vs cLogP)",
                xaxis_title="Molecular Weight (Da)",
                yaxis_title="cLogP",
                margin=dict(l=20, r=20, t=40, b=20),
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            charts.append(fig_scatter.to_html(full_html=False, include_plotlyjs=False))

    except Exception as e:
        logger.warning(f"Could not build Plotly charts: {e}")

    if not charts:
        return "<p style='color:#718096;'>No chart data available.</p>"

    # Wrap in responsive grid
    grid_html = "<div style='display:grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px;'>"
    for c in charts:
        grid_html += f"<div class='chart-box'>{c}</div>"
    grid_html += "</div>"
    return grid_html


def _build_html_template(
    report_title, workflow_type, timestamp,
    total_screened, passed_dl, passed_admet, total_docked,
    best_affinity, top_lead_name, top_lead_score,
    table_rows_html, charts_html, protein_pdb, ligand_sdf, quarantine_html,
    logs_html: str = ""
) -> str:
    """Combine components into a modern responsive HTML report."""
    
    # Escape structures for safe JS injection
    clean_protein = json.dumps(protein_pdb)
    clean_ligand = json.dumps(ligand_sdf)
    has_3d = bool(protein_pdb or ligand_sdf)

    viewer_section = ""
    if has_3d:
        viewer_section = f"""
        <div class="card" style="margin-top: 30px;">
          <h3>🔬 3D Molecular Complex & Active Site Pose</h3>
          <p style="color:#718096; font-size:14px; margin-bottom:12px;">
            Interactive 3D visualization of target protein and top docking pose. Use left-click to rotate, right-click to translate, and scroll to zoom.
          </p>
          <div id="pose3d_viewer" style="height: 480px; width: 100%; position: relative; border-radius: 8px; overflow: hidden; background: #1a202c;"></div>
          <div style="display:flex; gap:10px; margin-top:10px;">
            <button class="btn" onclick="reset3DView()">Reset Orientation</button>
            <button class="btn" onclick="toggleProteinStyle()">Toggle Surface / Cartoon</button>
          </div>
        </div>

        <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"></script>
        <script>
          let glviewer = null;
          let showSurface = false;
          let surfId = null;

          document.addEventListener("DOMContentLoaded", function() {{
            let container = document.getElementById("pose3d_viewer");
            if (!container) return;

            let config = {{ backgroundColor: "#1a202c" }};
            glviewer = $3Dmol.createViewer(container, config);

            let protData = {clean_protein};
            let ligData = {clean_ligand};

            if (protData && protData.length > 20) {{
              glviewer.addModel(protData, "pdb");
              glviewer.setStyle({{}}, {{cartoon: {{color: "spectrum", opacity: 0.85}}}});
            }}

            if (ligData && ligData.length > 20) {{
              glviewer.addModel(ligData, "sdf");
              glviewer.setStyle({{model: -1}}, {{stick: {{colorscheme: "greenCarbon", radius: 0.25}}, sphere: {{radius: 0.45}}}});
            }}

            glviewer.zoomTo();
            glviewer.render();
          }});

          function reset3DView() {{
            if (glviewer) {{
              glviewer.zoomTo();
              glviewer.render();
            }}
          }}

          function toggleProteinStyle() {{
            if (!glviewer) return;
            showSurface = !showSurface;
            if (showSurface) {{
              surfId = glviewer.addSurface($3Dmol.SurfaceType.VDW, {{opacity: 0.5, color: "white"}}, {{model: 0}});
            }} else if (surfId) {{
              glviewer.removeSurface(surfId);
              surfId = null;
            }}
            glviewer.render();
          }}
        </script>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{html.escape(report_title)}</title>
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
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
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
      border-radius: 12px;
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
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
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
      border-radius: 10px;
      border: 1px solid var(--border);
      box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }}
    .kpi-title {{ font-size: 12px; font-weight: 600; color: #718096; text-transform: uppercase; margin-bottom: 6px; }}
    .kpi-value {{ font-size: 22px; font-weight: 700; color: #1a202c; }}

    /* Cards */
    .card {{
      background: var(--card-bg);
      padding: 24px;
      border-radius: 12px;
      border: 1px solid var(--border);
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
      margin-bottom: 24px;
    }}
    .chart-box {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
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
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s ease;
    }}
    .btn:hover {{ background: #e2e8f0; }}

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
      <h3>📊 Pipeline Analytics & Chemical Space</h3>
      {charts_html}
    </div>

    <!-- 3D Viewer Section -->
    {viewer_section}

    <!-- Top Candidates Table -->
    <div class="card">
      <h3>🏆 Top Identified Lead Candidates</h3>
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
</body>
</html>
"""
