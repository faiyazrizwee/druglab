"""
DrugLab Lead Optimization — Candidate Ranking Module.

Ranks optimized analogues by comparing their properties to the original lead.
Computes improvement scores and produces final ranked output.
"""

import os
import time
import logging
import pandas as pd
import numpy as np
from rdkit import Chem

from utils.logger import setup_logger, log_step
from utils.io import (
    ensure_dir,
    write_json,
    read_sdf,
    write_sdf,
    write_csv,
    read_csv,
    validate_file_exists,
)
from utils.chemistry import (
    get_mol_name,
    compute_tanimoto_similarity,
    calculate_ligand_efficiency,
    calculate_lipophilic_efficiency,
)

logger = setup_logger("optimization_ranking", level="INFO")


def rank_candidates(
    docking_results: str | pd.DataFrame,
    output_dir: str,
    lead_sdf: str | None = None,
    admet_results: str | pd.DataFrame | None = None,
    druglikeness_results: str | pd.DataFrame | None = None,
    analogues_sdf: str | None = None,
    top_n: int = 10,
    weights: dict | None = None,
) -> dict:
    """
    Rank optimized analogue candidates.

    Args:
        docking_results: Path to docking_results.csv or DataFrame.
        output_dir: Output directory.
        lead_sdf: Path to the original lead SDF for comparison.
        admet_results: Path to admet_results.csv or DataFrame.
        druglikeness_results: Path to descriptors.csv or DataFrame.
        analogues_sdf: Path to analogues SDF file.
        top_n: Number of top candidates to select.
        weights: Component weights for scoring.

    Returns:
        Dict with ranking results and output file paths.
    """
    start_time = time.time()
    log_step(logger, "Candidate Ranking", "started", {"top_n": top_n})

    if weights is None:
        weights = {"affinity": 0.5, "admet": 0.3, "druglikeness": 0.2}

    # Output paths
    rank_dir = os.path.join(output_dir, "ranking")
    ensure_dir(rank_dir)

    optimized_csv_path = os.path.join(rank_dir, "analogue_ranking.csv")
    top_sdf_path = os.path.join(rank_dir, "top_optimized_leads.sdf")
    best_sdf_path = os.path.join(rank_dir, "best_optimized_lead.sdf")
    best_pdbqt_path = os.path.join(rank_dir, "best_optimized_lead.pdbqt")
    best_report_path = os.path.join(rank_dir, "best_optimized_lead_report.json")
    report_path = os.path.join(rank_dir, "lead_optimization_report.json")

    # Load docking results
    dock_df = _load_df(docking_results, "Docking results")
    logger.info(f"Loaded {len(dock_df)} docking results for ranking.")

    if len(dock_df) == 0:
        metrics = {
            "total_input": 0, "total_ranked": 0,
            "status": "completed", "duration": round(time.time() - start_time, 2),
        }
        write_json(metrics, report_path)
        return metrics

    # Load optional data
    admet_df = _load_df(admet_results, "ADMET results") if admet_results else None
    dl_df = _load_df(druglikeness_results, "DL results") if druglikeness_results else None

    # Merge
    merged = dock_df.copy()
    if admet_df is not None and "name" in admet_df.columns:
        admet_cols = ["name"] + [c for c in admet_df.columns
                                  if c not in merged.columns and c != "name"]
        merged = merged.merge(admet_df[admet_cols].drop_duplicates(subset=["name"]),
                              on="name", how="left")
    if dl_df is not None and "name" in dl_df.columns:
        dl_cols = ["name"] + [c for c in dl_df.columns
                               if c not in merged.columns and c != "name"]
        merged = merged.merge(dl_df[dl_cols].drop_duplicates(subset=["name"]),
                              on="name", how="left")

    # Compute scores
    merged = _compute_scores(merged, weights)

    # Compute Ligand Efficiency (LE) and Lipophilic Efficiency (LLE / LipE)
    if "best_affinity" in merged.columns:
        if "HeavyAtoms" in merged.columns:
            merged["ligand_efficiency"] = merged.apply(
                lambda r: calculate_ligand_efficiency(r["best_affinity"], int(r["HeavyAtoms"]))
                if pd.notnull(r.get("HeavyAtoms")) and r.get("HeavyAtoms") > 0 else np.nan,
                axis=1
            )
        elif "MW" in merged.columns:
            merged["ligand_efficiency"] = merged.apply(
                lambda r: calculate_ligand_efficiency(r["best_affinity"], max(1, int(r["MW"] / 13.5)))
                if pd.notnull(r.get("MW")) else np.nan,
                axis=1
            )
        if "LogP" in merged.columns:
            merged["lipophilic_efficiency"] = merged.apply(
                lambda r: calculate_lipophilic_efficiency(r["best_affinity"], float(r["LogP"]))
                if pd.notnull(r.get("LogP")) else np.nan,
                axis=1
            )

    # Compare to lead if available
    if lead_sdf and os.path.isfile(lead_sdf):
        merged = _compute_improvement(merged, lead_sdf)

    # Sort by composite score descending
    merged = merged.sort_values("composite_score", ascending=False).reset_index(drop=True)
    merged["rank"] = range(1, len(merged) + 1)

    # Select top N
    top_candidates = merged.head(top_n)

    # Format the requested columns
    merged["Compound"] = merged["name"]
    merged["Docking"] = merged["best_affinity"].fillna(0).round(3)
    merged["ADMET"] = merged["status_y"] if "status_y" in merged.columns else "PASS" # Assuming ADMET status is joined
    merged["DrugLike"] = merged["status_dl"] if "status_dl" in merged.columns else "PASS"
    
    # Scale FinalScore 0-100 based on composite_score
    merged["FinalScore"] = (merged["composite_score"] * 100).round(1)
    
    # Write outputs
    write_csv(merged, optimized_csv_path)

    # Write top candidates SDF
    best_candidate_name = None
    if analogues_sdf and os.path.isfile(analogues_sdf):
        top_names = {str(name) for name in top_candidates["name"].tolist()}
        mols = read_sdf(analogues_sdf)
        top_mols = [m for m in mols if m and get_mol_name(m) in top_names]
        if top_mols:
            write_sdf(top_mols, top_sdf_path)
            
        # Get the #1 best lead
        if len(top_mols) > 0:
            best_mol = top_mols[0] # Note: read_sdf might not preserve rank order, we should match by name
            best_candidate_name = str(top_candidates.iloc[0]["name"])
            for m in top_mols:
                if get_mol_name(m) == best_candidate_name:
                    write_sdf([m], best_sdf_path)
                    break
                    
    # Generate best_optimized_lead.pdbqt from the docking output
    if best_candidate_name:
        # We need to find the poses directory for the best candidate
        # The docking results contain "pose_file" which looks like: res/analogue_docking/poses/analogue_024_poses.pdbqt
        best_pose_file = top_candidates.iloc[0].get("pose_file")
        if best_pose_file and os.path.isfile(best_pose_file):
            base_pose_dir = os.path.splitext(best_pose_file)[0]
            best_pose_source = os.path.join(base_pose_dir, "model_1.pdbqt")
            if os.path.isfile(best_pose_source):
                import shutil
                shutil.copyfile(best_pose_source, best_pdbqt_path)
                
        # Generate best report
        best_report = top_candidates.iloc[0].to_dict()
        write_json(best_report, best_report_path)

    # ── Figure Generation ──
    try:
        from reports.generator import generate_optimization_figures
        plot_df = merged.copy()
        plot_df = plot_df.rename(columns={"composite_score": "Total_Score"})
        
        # We need a parent_df to compare improvement
        parent_df = None
        if "is_original" in plot_df.columns:
            # Maybe the lead is in the same dataframe? Usually it's not if it's only analogues.
            pass
        
        # We can construct parent_df from lead_sdf and its metrics if available
        # For now, generate_optimization_figures can handle parent_df=None gracefully
        generated_figs = generate_optimization_figures(plot_df, pd.DataFrame(), output_dir)
    except Exception as e:
        logger.error(f"Failed to generate optimization figures: {e}")
        generated_figs = []

    duration = time.time() - start_time
    metrics = {
        "total_input": len(dock_df),
        "total_ranked": len(merged),
        "top_n": len(top_candidates),
        "optimized_csv": optimized_csv_path,
        "top_sdf": top_sdf_path,
        "best_optimized_lead_sdf": best_sdf_path,
        "best_optimized_lead_pdbqt": best_pdbqt_path,
        "best_optimized_lead_report": best_report_path,
        "top_candidates": top_candidates.to_dict(orient="records"),
        "generated_figures": generated_figs,
        "duration": round(duration, 2),
        "status": "completed",
    }

    write_json(metrics, report_path)
    
    log_dict = {
        "ranked": len(merged),
        "top_n": len(top_candidates),
        "duration": f"{duration:.1f}s",
    }
    
    if generated_figs:
        fig_msg = "\nGenerated Figures:\n"
        for f in generated_figs:
            fig_msg += f"  ✓ {os.path.basename(f)}\n"
        log_dict["figures"] = fig_msg

    log_step(logger, "Candidate Ranking", "completed", log_dict)
    
    if generated_figs:
        print(f"\nGenerated {len(generated_figs)} figures:")
        print(f"{os.path.join(output_dir, 'figures', 'lead_optimization')}/")
        for f in generated_figs:
            print(f"  ✓ {os.path.basename(f)}")
        print()
    return metrics


def _compute_scores(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Compute composite scores for ranking."""
    df = df.copy()

    # Affinity score (more negative = better)
    if "best_affinity" in df.columns:
        aff = df["best_affinity"].fillna(0)
        aff_min, aff_max = aff.min(), aff.max()
        if aff_min != aff_max:
            df["affinity_score"] = (aff_max - aff) / (aff_max - aff_min)
        else:
            df["affinity_score"] = 1.0
    else:
        df["affinity_score"] = 0.0

    # ADMET score
    if "violation_count" in df.columns:
        vc = df["violation_count"].fillna(0)
        vc_max = max(vc.max(), 1)
        df["admet_score"] = 1.0 - (vc / vc_max)
    else:
        df["admet_score"] = 1.0

    # Drug-likeness score
    if "LogP" in df.columns and "MW" in df.columns:
        logp_s = 1.0 - (abs(df["LogP"].fillna(2.5) - 2.5) / 5.0).clip(0, 1)
        mw_s = 1.0 - (abs(df["MW"].fillna(350) - 350) / 350).clip(0, 1)
        df["dl_score"] = (logp_s + mw_s) / 2.0
    else:
        df["dl_score"] = 1.0

    # Weighted composite
    w_aff = weights.get("affinity", 0.5)
    w_admet = weights.get("admet", 0.3)
    w_dl = weights.get("druglikeness", 0.2)
    total = w_aff + w_admet + w_dl

    df["composite_score"] = (
        (df["affinity_score"] * w_aff +
         df["admet_score"] * w_admet +
         df["dl_score"] * w_dl) / total
    ).round(4)

    return df


def _compute_improvement(df: pd.DataFrame, lead_sdf: str) -> pd.DataFrame:
    """Compare analogues to the original lead and add improvement columns."""
    df = df.copy()

    try:
        mols = read_sdf(lead_sdf)
        if not mols:
            return df
        lead_mol = Chem.RemoveHs(mols[0])
        lead_props = {}
        if "best_affinity" in df.columns:
            # We don't have the lead's affinity in the SDF, use the best in the batch
            lead_props["affinity"] = None

        from utils.chemistry import calculate_basic_properties
        lp = calculate_basic_properties(lead_mol)

        # Add similarity to lead
        from utils.chemistry import compute_fingerprint
        from rdkit.Chem import DataStructs

        lead_fp = compute_fingerprint(lead_mol)
        similarities = []
        for _, row in df.iterrows():
            try:
                smi = row.get("smiles", "")
                if smi:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        fp = compute_fingerprint(mol)
                        sim = DataStructs.TanimotoSimilarity(lead_fp, fp)
                        similarities.append(round(sim, 3))
                        continue
            except Exception:
                pass
            similarities.append(None)

        df["similarity_to_lead"] = similarities

    except Exception as e:
        logger.warning(f"Could not compute improvement metrics: {e}")

    return df


def _load_df(source, description: str) -> pd.DataFrame:
    """Load a DataFrame from path or return if already a DataFrame."""
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, str):
        validate_file_exists(source, description)
        return pd.read_csv(source)
    raise ValueError(f"Expected str path or DataFrame for {description}")
