"""
Peptide Lead Selection and Multi-Objective Complex Analysis.

Ranks docked peptide candidates across:
1. LightDock docking score (DFIRE / FastDFIRE / CPYDOCK).
2. Interface contacts (< 4.5 Å).
3. Hydrogen bonds (< 3.5 Å).
4. Steric clash penalties (< 2.0 Å).
5. ADMET & hemolytic safety profile.

Exports top hits to lead_peptides.csv, top_poses.tsv, and copies complex PDBs to results/leads/.
"""

from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path
from typing import Any
import pandas as pd

from utils.io import ensure_dir
from utils.logger import setup_logger

logger = setup_logger("peptide_leads", level="INFO")


def rank_and_select_peptide_leads(
    candidates_df: pd.DataFrame | list[dict[str, Any]] | None = None,
    poses_df: pd.DataFrame | list[dict[str, Any]] | None = None,
    output_dir: str | Path = "results",
    admet_df: pd.DataFrame | None = None,
    candidates: list[Any] | None = None,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Perform multi-objective lead selection on docked peptide candidates.

    Returns:
        pd.DataFrame containing top-ranked leads.
    """
    out_dir = Path(ensure_dir(output_dir))
    leads_dir = ensure_dir(out_dir / "leads")

    # Read from disk if not directly supplied
    if (candidates_df is None or (isinstance(candidates_df, (list, pd.DataFrame)) and len(candidates_df) == 0)) and (out_dir / "docking_results.tsv").is_file():
        try:
            candidates_df = pd.read_csv(out_dir / "docking_results.tsv", sep="\t")
        except Exception:
            pass

    if (poses_df is None or (isinstance(poses_df, (list, pd.DataFrame)) and len(poses_df) == 0)) and (out_dir / "pose_metrics.tsv").is_file():
        try:
            poses_df = pd.read_csv(out_dir / "pose_metrics.tsv", sep="\t")
        except Exception:
            pass

    if (admet_df is None or (isinstance(admet_df, pd.DataFrame) and admet_df.empty)) and (out_dir / "admet_predictions.tsv").is_file():
        try:
            admet_df = pd.read_csv(out_dir / "admet_predictions.tsv", sep="\t")
        except Exception:
            pass

    c_df = pd.DataFrame(candidates_df) if not isinstance(candidates_df, pd.DataFrame) else candidates_df.copy()
    p_df = pd.DataFrame(poses_df) if not isinstance(poses_df, pd.DataFrame) else poses_df.copy()

    if c_df.empty:
        logger.warning("No candidate records available for lead selection.")
        return pd.DataFrame()

    # Merge ADMET data if available
    if admet_df is not None and not admet_df.empty:
        admet_cols = [c for c in admet_df.columns if c not in ("sequence",) or c == "candidate_id"]
        c_df = c_df.merge(admet_df[admet_cols], on="candidate_id", how="left")

    # Merge best pose information
    if not p_df.empty and "candidate_id" in p_df.columns:
        best_poses = p_df.sort_values(by=["candidate_id", "pose_rank"]).groupby("candidate_id").first().reset_index()
        pose_cols = [c for c in best_poses.columns if c not in c_df.columns or c == "candidate_id"]
        c_df = c_df.merge(best_poses[pose_cols], on="candidate_id", how="left")

    # Compute Composite Lead Score
    # Lower/more negative LightDock score is better
    # High contacts, high H-bonds, low clashes, low toxicity risk
    composite_scores = []
    for _, row in c_df.iterrows():
        ld_score = float(row.get("best_lightdock_score", row.get("lightdock_score", 0.0)) or 0.0)
        contacts = int(row.get("contacts_count", 0) or 0)
        hbonds = int(row.get("hbonds_count", 0) or 0)
        clashes = int(row.get("clashes_count", 0) or 0)

        # ADMET toxicity bonus/penalty
        tox = str(row.get("peptide_toxicity_risk", "")).lower()
        tox_adj = 0.0
        if "low risk" in tox or "safe" in tox:
            tox_adj = -5.0  # Favorable
        elif "high risk" in tox:
            tox_adj = +15.0  # Penalty

        # Composite score (lower is better)
        # LightDock score + contacts/hbond reward - clash penalty + toxicity penalty
        comp_score = ld_score - (contacts * 0.15) - (hbonds * 0.8) + (clashes * 3.0) + tox_adj
        composite_scores.append(round(comp_score, 2))

    c_df["composite_lead_score"] = composite_scores
    c_df = c_df.sort_values(by="composite_lead_score", ascending=True).reset_index(drop=True)
    c_df["lead_rank"] = range(1, len(c_df) + 1)

    # Filter top leads
    top_leads_df = c_df.head(top_n).copy()

    # Add standard reporting aliases
    top_leads_df["rank"] = top_leads_df["lead_rank"]
    top_leads_df["composite_score"] = top_leads_df["composite_lead_score"]
    if "best_lightdock_score" in top_leads_df.columns:
        top_leads_df["lightdock_score"] = top_leads_df["best_lightdock_score"]
    if "contacts_count" in top_leads_df.columns:
        top_leads_df["contacts"] = top_leads_df["contacts_count"]
    if "hbonds_count" in top_leads_df.columns:
        top_leads_df["hbonds"] = top_leads_df["hbonds_count"]
    if "clashes_count" in top_leads_df.columns:
        top_leads_df["clashes"] = top_leads_df["clashes_count"]
    if "peptide_toxicity_risk" in top_leads_df.columns:
        top_leads_df["hemolytic_risk"] = top_leads_df["peptide_toxicity_risk"]
    if "selection_notes" not in top_leads_df.columns:
        top_leads_df["selection_notes"] = "Balanced binding energy and safety profile"

    # Copy top complex PDBs to leads/
    copied_leads = []
    for _, row in top_leads_df.iterrows():
        cid = str(row.get("candidate_id", ""))
        pose_path = row.get("pose_pdb_path")
        if pose_path and os.path.isfile(str(pose_path)):
            dest = Path(leads_dir) / f"lead_rank{row['lead_rank']}_{cid}.pdb"
            try:
                shutil.copy2(str(pose_path), dest)
                copied_leads.append(str(dest))
            except Exception as e:
                logger.debug(f"Could not copy lead PDB: {e}")

    # Write lead tables
    leads_csv_path = out_dir / "lead_peptides.csv"
    top_leads_df.to_csv(leads_csv_path, index=False)

    top_poses_tsv = out_dir / "top_poses.tsv"
    if not p_df.empty:
        p_df.to_csv(top_poses_tsv, sep="\t", index=False)

    logger.info(f"Selected {len(top_leads_df)} lead peptide candidate(s) -> {leads_csv_path}")

    # Display console table
    if sys.stdin.isatty():
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(
                title="DrugLab 2.0 — Top Selected Peptide Leads",
                border_style="magenta",
                header_style="bold magenta",
                show_header=True,
            )
            table.add_column("Lead Rank", style="bold yellow", width=11, justify="center")
            table.add_column("Candidate ID", style="bold white", width=14)
            table.add_column("Sequence", style="font-family monospace", width=15)
            table.add_column("Docking Score", justify="right", width=14)
            table.add_column("H-Bonds", justify="right", width=10)
            table.add_column("Contacts", justify="right", width=10)
            table.add_column("Clashes", justify="right", width=9)
            table.add_column("Toxicity Risk", justify="center", width=20)
            table.add_column("Composite Score", justify="right", style="bold cyan", width=16)

            for _, r in top_leads_df.iterrows():
                seq_str = str(r.get("sequence", ""))
                if len(seq_str) > 12:
                    seq_str = seq_str[:9] + "..."
                tox_str = str(r.get("peptide_toxicity_risk", "-"))
                if "Low" in tox_str:
                    tox_badge = "[green]Low (Safe)[/green]"
                elif "High" in tox_str:
                    tox_badge = "[red]High Risk[/red]"
                else:
                    tox_badge = "[dim]Moderate/Undet[/dim]"

                table.add_row(
                    f"#{r['lead_rank']}",
                    str(r.get("candidate_id", "")),
                    seq_str,
                    f"{float(r.get('best_lightdock_score', 0.0)):.2f}",
                    str(r.get("hbonds_count", 0)),
                    str(r.get("contacts_count", 0)),
                    str(r.get("clashes_count", 0)),
                    tox_badge,
                    f"{float(r.get('composite_lead_score', 0.0)):.2f}",
                )

            console.print()
            console.print(table)
            console.print()
        except Exception:
            pass

    return top_leads_df
