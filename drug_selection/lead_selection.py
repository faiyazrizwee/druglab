"""
DrugLab Lead Selection Module.

Implements multi-criteria ranking of docked ligands by combining binding
affinity, ADMET scores, and drug-likeness properties into a composite score.
Supports weighted-sum, TOPSIS, and Pareto-front scoring methods.
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
    deduplicate_mols,
    calculate_ligand_efficiency,
    calculate_lipophilic_efficiency,
)

logger = setup_logger("lead_selection", level="INFO")


def select_leads(
    docking_results: str | pd.DataFrame,
    output_dir: str,
    admet_results: str | pd.DataFrame | None = None,
    druglikeness_results: str | pd.DataFrame | None = None,
    ligands_sdf: str | None = None,
    top_n: int = 5,
    scoring_method: str = "weighted_sum",
    weights: dict | None = None,
    affinity_cutoff: float = -6.0,
    on_no_hits: str = "prompt",
    diversity_filter: bool = True,
    similarity_cutoff: float = 0.7,
) -> dict:
    """
    Select top lead compounds from docking and screening results.

    Args:
        docking_results: Path to docking_results.csv or DataFrame.
        output_dir: Path to the output directory.
        admet_results: Path to admet_results.csv or DataFrame (optional).
        druglikeness_results: Path to descriptors.csv or DataFrame (optional).
        ligands_sdf: Path to SDF file for extracting Mol objects (optional).
        top_n: Number of top leads to select.
        scoring_method: "weighted_sum", "topsis", or "pareto".
        weights: Dict of component weights (affinity, admet, druglikeness).
        affinity_cutoff: Minimum binding affinity (kcal/mol, more negative = better).
        diversity_filter: If True, apply diversity filtering.
        similarity_cutoff: Max Tanimoto similarity between leads (for diversity).

    Returns:
        Dict with selection results and output file paths.
    """
    start_time = time.time()
    log_step(logger, "Lead Selection", "started", {
        "scoring_method": scoring_method,
        "top_n": top_n,
    })

    if weights is None:
        weights = {"affinity": 0.5, "admet": 0.3, "druglikeness": 0.2}

    # Output paths
    lead_dir = os.path.join(output_dir, "lead_selection")
    ensure_dir(lead_dir)

    top_hits_csv_path = os.path.join(lead_dir, "top_hits.csv")
    ranking_csv_path = os.path.join(lead_dir, "ranking.csv")
    best_lead_sdf_path = os.path.join(lead_dir, "best_lead.sdf")
    leads_sdf_path = os.path.join(lead_dir, "leads.sdf")
    report_path = os.path.join(lead_dir, "lead_selection_report.json")

    # Load docking results
    dock_df = _load_df(docking_results, "Docking results")
    logger.info(f"Loaded {len(dock_df)} docking results.")

    # Filter by affinity cutoff
    dock_df_unfiltered = dock_df.copy()
    if "best_affinity" in dock_df.columns:
        dock_df = dock_df_unfiltered[dock_df_unfiltered["best_affinity"] <= affinity_cutoff].copy()
        logger.info(f"After affinity cutoff ({affinity_cutoff} kcal/mol): "
                     f"{len(dock_df)} remaining.")

    fallback_used = False
    fallback_mode = None
    cutoff_changed = False
    old_cutoff = affinity_cutoff
    best_affinity_observed = dock_df_unfiltered["best_affinity"].min() if "best_affinity" in dock_df_unfiltered.columns and not dock_df_unfiltered.empty else "N/A"

    while len(dock_df) == 0:
        if on_no_hits == "fallback":
            dock_df = dock_df_unfiltered
            fallback_used = True
            fallback_mode = "top_ranked"
            logger.warning("No molecules passed cutoff. Using top-ranked compounds instead (fallback mode).")
            break
        elif on_no_hits == "abort":
            logger.info("Workflow terminated by config (abort on no hits).")
            metrics = {
                "total_input": len(dock_df_unfiltered), "total_selected": 0, "status": "aborted",
                "top_hits_csv": top_hits_csv_path, "duration": round(time.time() - start_time, 2),
            }
            write_json(metrics, report_path)
            return metrics
        else:
            print("\n" + "="*60)
            print("Lead Selection Warning")
            print("="*22 + "\n")
            print(f"No molecules passed affinity cutoff ({affinity_cutoff} kcal/mol)\n")
            print(f"Best docking score observed:\n{best_affinity_observed} kcal/mol\n")
            print(f"Total docked molecules:\n{len(dock_df_unfiltered)}\n")
            print("Options:\n")
            print("  [1] Continue with top-ranked compounds")
            print("  [2] Change affinity cutoff")
            print("  [3] Abort workflow\n")
            print("Selection: ", end="", flush=True)
            
            try:
                choice = input().strip()
            except (EOFError, KeyboardInterrupt):
                choice = "3"
                print()
                
            if choice == "1":
                print("\nUsing top-ranked compounds.")
                dock_df = dock_df_unfiltered
                fallback_used = True
                fallback_mode = "top_ranked"
                break
            elif choice == "2":
                print("\nEnter new affinity cutoff: ", end="", flush=True)
                try:
                    new_cutoff_str = input().strip()
                    affinity_cutoff = float(new_cutoff_str)
                    cutoff_changed = True
                    print(f"\nNew cutoff accepted:\n{affinity_cutoff} kcal/mol\n")
                    
                    if "best_affinity" in dock_df_unfiltered.columns:
                        dock_df = dock_df_unfiltered[dock_df_unfiltered["best_affinity"] <= affinity_cutoff].copy()
                        print(f"Passed compounds:\n{len(dock_df)}\n")
                        if len(dock_df) > 0:
                            break
                        else:
                            continue
                except ValueError:
                    print("\nInvalid cutoff. Try again.")
                    continue
            else:
                print("\nWorkflow terminated by user.")
                metrics = {
                    "total_input": len(dock_df_unfiltered), "total_selected": 0, "status": "aborted",
                    "top_hits_csv": top_hits_csv_path, "duration": round(time.time() - start_time, 2),
                }
                write_json(metrics, report_path)
                return metrics

    # Load optional ADMET results
    admet_df = None
    if admet_results is not None:
        admet_df = _load_df(admet_results, "ADMET results")

    # Load optional druglikeness results
    dl_df = None
    if druglikeness_results is not None:
        dl_df = _load_df(druglikeness_results, "Druglikeness results")

    # Merge data on molecule name
    merged = dock_df.copy()
    if admet_df is not None and "name" in admet_df.columns:
        # Select relevant columns only
        admet_cols = ["name"] + [c for c in admet_df.columns
                                  if c not in merged.columns and c != "name"]
        admet_subset = admet_df[admet_cols].drop_duplicates(subset=["name"])
        merged = merged.merge(admet_subset, on="name", how="left")

    if dl_df is not None and "name" in dl_df.columns:
        dl_cols = ["name"] + [c for c in dl_df.columns
                               if c not in merged.columns and c != "name"]
        dl_subset = dl_df[dl_cols].drop_duplicates(subset=["name"])
        merged = merged.merge(dl_subset, on="name", how="left")

    # De-fragment DataFrame after merging multiple wide descriptor tables
    merged = merged.copy()

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

    # Compute composite scores
    if scoring_method == "weighted_sum":
        merged = _score_weighted_sum(merged, weights)
    elif scoring_method == "topsis":
        merged = _score_topsis(merged, weights)
    elif scoring_method == "pareto":
        merged = _score_pareto(merged)
    else:
        logger.warning(f"Unknown scoring method '{scoring_method}', using weighted_sum.")
        merged = _score_weighted_sum(merged, weights)

    # Sort by composite score (descending)
    merged = merged.sort_values("composite_score", ascending=False).reset_index(drop=True)

    # Diversity filter
    if diversity_filter and ligands_sdf:
        merged = _apply_diversity_filter(merged, ligands_sdf, similarity_cutoff, top_n)

    # Select top N
    top_leads = merged.head(top_n).copy()
    top_leads["rank"] = range(1, len(top_leads) + 1)

    # Write outputs
    write_csv(top_leads, top_hits_csv_path)
    write_csv(merged, ranking_csv_path)

    # Write lead SDF files if ligands SDF available
    lead_names = {str(name) for name in top_leads["name"].tolist()}
    if ligands_sdf and os.path.isfile(ligands_sdf):
        mols = read_sdf(ligands_sdf)
        lead_mols = []
        best_lead_mol = None
        for mol in mols:
            if mol is None:
                continue
            name = get_mol_name(mol)
            if name in lead_names:
                lead_mols.append(mol)
                if best_lead_mol is None:
                    best_lead_mol = mol
        if lead_mols:
            write_sdf(lead_mols, leads_sdf_path)
        if best_lead_mol:
            write_sdf([best_lead_mol], best_lead_sdf_path)

    # ── Figure Generation ──
    try:
        from reports.generator import generate_lead_selection_figures
        plot_df = merged.copy()
        # Rename for radar chart if available
        rename_map = {
            "affinity_score": "Docking_Norm",
            "druglikeness_score": "DrugLikeness_Norm",
            "admet_score": "ADMET_Norm",
            "risk_score_norm": "Risk_Score_Norm"
        }
        plot_df = plot_df.rename(columns=rename_map)
        
        # fallback for BBB_Norm
        if "BBB_status" in plot_df.columns:
            plot_df["BBB_Norm"] = plot_df["BBB_status"].apply(lambda x: 1.0 if x == "PASS" else (0.5 if x == "WARNING" else 0.0))

        generated_figs = generate_lead_selection_figures(plot_df, output_dir)
    except Exception as e:
        logger.error(f"Failed to generate lead selection figures: {e}")
        generated_figs = []

    duration = time.time() - start_time
    metrics = {
        "total_input": len(dock_df_unfiltered),
        "total_selected": len(top_leads),
        "scoring_method": scoring_method,
        "top_hits_csv": top_hits_csv_path,
        "ranking_csv": ranking_csv_path,
        "leads_sdf": leads_sdf_path,
        "best_lead_sdf": best_lead_sdf_path,
        "top_leads": top_leads.to_dict(orient="records"),
        "generated_figures": generated_figs,
        "duration": round(duration, 2),
        "status": "completed",
    }
    
    if fallback_used:
        metrics.update({
            "affinity_cutoff": old_cutoff,
            "best_affinity_observed": best_affinity_observed,
            "cutoff_pass_count": 0,
            "fallback_used": True,
            "fallback_mode": fallback_mode
        })
    elif cutoff_changed:
        metrics.update({
            "cutoff_changed": True,
            "old_cutoff": old_cutoff,
            "new_cutoff": affinity_cutoff
        })

    write_json(metrics, report_path)
    
    log_dict = {
        "selected": len(top_leads),
        "scoring_method": scoring_method,
        "duration": f"{duration:.1f}s",
    }
    
    if generated_figs:
        fig_msg = "\nGenerated Figures:\n"
        for f in generated_figs:
            fig_msg += f"  ✓ {os.path.basename(f)}\n"
        log_dict["figures"] = fig_msg

    log_step(logger, "Lead Selection", "completed", log_dict)
    
    if generated_figs:
        print(f"\nGenerated {len(generated_figs)} figures:")
        print(f"{os.path.join(output_dir, 'figures', 'lead_selection')}/")
        for f in generated_figs:
            print(f"  ✓ {os.path.basename(f)}")
        print()

    if len(top_leads) > 0:
        best_lead = top_leads.iloc[0]
        name = best_lead.get("name", "Unknown")
        affinity = best_lead.get("best_affinity", "N/A")
        
        # Check if the column actually exists in the df before checking its boolean value
        has_admet = "admet_passed" in best_lead.index
        admet_status = "PASS" if not has_admet or str(best_lead["admet_passed"]).lower() == "true" else "FAIL"
        
        has_dl = "druglikeness_passed" in best_lead.index
        dl_status = "PASS" if not has_dl or str(best_lead["druglikeness_passed"]).lower() == "true" else "FAIL"

        print("\n==================================")
        print("LEAD SELECTED")
        print("=============\n")
        print("Molecule:")
        print(f"{name}\n")
        print("Docking Score:")
        if affinity != "N/A":
            print(f"{affinity} kcal/mol\n")
        else:
            print("N/A\n")
        print("ADMET Status:")
        print(f"{admet_status}\n")
        print("Drug-Likeness:")
        print(f"{dl_status}\n")
        print("Rank:")
        print("1\n")
        print("Lead File:")
        print(f"{best_lead_sdf_path}\n")
        print("==================================\n")

    return metrics


# ──────────────────────────────────────────────────────────────
# Scoring Methods
# ──────────────────────────────────────────────────────────────

def _score_weighted_sum(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    Compute weighted sum composite score.

    Normalizes affinity (more negative = better), ADMET pass rate, and
    druglikeness properties to [0, 1] then combines with weights.
    """
    df = df.copy()

    # Affinity score: normalize to [0, 1] (lower affinity = higher score)
    if "best_affinity" in df.columns:
        aff = df["best_affinity"].fillna(0)
        aff_min, aff_max = aff.min(), aff.max()
        if aff_min != aff_max:
            df["affinity_score"] = (aff_max - aff) / (aff_max - aff_min)
        else:
            df["affinity_score"] = 1.0
    else:
        df["affinity_score"] = 0.0

    # ADMET score: use violation_count if available, else binary pass
    has_admet = False
    if "violation_count" in df.columns:
        vc = df["violation_count"].fillna(0)
        vc_max = max(vc.max(), 1)
        df["admet_score"] = 1.0 - (vc / vc_max)
        has_admet = True
    elif "admet_passed" in df.columns:
        df["admet_score"] = df["admet_passed"].apply(
            lambda x: 1.0 if str(x).lower() == "true" else 0.0
        )
        has_admet = True
    else:
        df["admet_score"] = 1.0  # Assume pass if no data

    # Druglikeness score: use LogP and MW proximity to ideal
    has_dl = False
    if "LogP" in df.columns and "MW" in df.columns:
        logp_score = 1.0 - (abs(df["LogP"].fillna(2.5) - 2.5) / 5.0).clip(0, 1)
        mw_score = 1.0 - (abs(df["MW"].fillna(350) - 350) / 350).clip(0, 1)
        df["dl_score"] = (logp_score + mw_score) / 2.0
        has_dl = True
    elif "druglikeness_passed" in df.columns:
        df["dl_score"] = df["druglikeness_passed"].apply(
            lambda x: 1.0 if str(x).lower() == "true" else 0.0
        )
        has_dl = True
    else:
        df["dl_score"] = 1.0

    # Weighted composite
    w_aff = weights.get("affinity", 0.5)
    w_admet = weights.get("admet", 0.3) if has_admet else 0.0
    w_dl = weights.get("druglikeness", 0.2) if has_dl else 0.0
    
    total = w_aff + w_admet + w_dl
    if total == 0:
        total = 1.0
        w_aff = 1.0

    df["composite_score"] = (
        (df["affinity_score"] * w_aff +
         df["admet_score"] * w_admet +
         df["dl_score"] * w_dl) / total
    ).round(4)

    return df


def _score_topsis(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    Compute TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution).
    """
    df = _score_weighted_sum(df, weights)

    criteria = ["affinity_score"]
    if "violation_count" in df.columns or "admet_passed" in df.columns:
        criteria.append("admet_score")
    if ("LogP" in df.columns and "MW" in df.columns) or "druglikeness_passed" in df.columns:
        criteria.append("dl_score")

    available = [c for c in criteria if c in df.columns]

    if not available:
        df["composite_score"] = 0.0
        return df

    matrix = df[available].fillna(0).values

    # Normalize
    norms = np.sqrt(np.sum(matrix ** 2, axis=0))
    norms[norms == 0] = 1
    normalized = matrix / norms

    # Weight
    w_dict = {"affinity_score": weights.get("affinity", 0.5),
              "admet_score": weights.get("admet", 0.3),
              "dl_score": weights.get("druglikeness", 0.2)}
    w_list = [w_dict[c] for c in available]
    
    total_w = sum(w_list)
    if total_w == 0:
        w_list = [1.0 / len(w_list) for _ in w_list]
    else:
        w_list = [w / total_w for w in w_list]

    weighted = normalized * np.array(w_list)

    # Ideal and anti-ideal
    ideal = np.max(weighted, axis=0)
    anti_ideal = np.min(weighted, axis=0)

    # Distances
    d_ideal = np.sqrt(np.sum((weighted - ideal) ** 2, axis=1))
    d_anti = np.sqrt(np.sum((weighted - anti_ideal) ** 2, axis=1))

    # Closeness coefficient
    denom = d_ideal + d_anti
    denom[denom == 0] = 1
    df["composite_score"] = np.round(d_anti / denom, 4)

    return df


def _score_pareto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign Pareto front rankings. Molecules on the first front get highest scores.
    """
    df = df.copy()

    criteria = []
    if "best_affinity" in df.columns:
        criteria.append(("best_affinity", -1))  # Lower is better
    if "violation_count" in df.columns:
        criteria.append(("violation_count", -1))  # Lower is better

    if not criteria:
        df["composite_score"] = 1.0
        return df

    # Simple iterative Pareto front extraction
    remaining = df.index.tolist()
    front_rank = pd.Series(0, index=df.index, dtype=float)
    current_front = 1

    while remaining:
        front = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                # Check if j dominates i
                j_better_in_all = True
                j_better_in_any = False
                for col, direction in criteria:
                    vi = df.loc[i, col] * direction
                    vj = df.loc[j, col] * direction
                    if vj < vi:
                        j_better_in_all = False
                    if vj > vi:
                        j_better_in_any = True
                if j_better_in_all and j_better_in_any:
                    dominated = True
                    break
            if not dominated:
                front.append(i)

        for idx in front:
            front_rank[idx] = current_front
            remaining.remove(idx)
        current_front += 1

    max_front = front_rank.max()
    if max_front > 0:
        df["composite_score"] = np.round(1.0 - (front_rank - 1) / max(max_front, 1), 4)
    else:
        df["composite_score"] = 1.0

    return df


# ──────────────────────────────────────────────────────────────
# Diversity Filter
# ──────────────────────────────────────────────────────────────

def _apply_diversity_filter(
    df: pd.DataFrame,
    ligands_sdf: str,
    similarity_cutoff: float,
    top_n: int,
) -> pd.DataFrame:
    """Remove structurally redundant leads based on Tanimoto similarity."""
    if not os.path.isfile(ligands_sdf):
        return df

    mols = read_sdf(ligands_sdf)
    name_to_mol = {}
    for mol in mols:
        if mol is None:
            continue
        name = get_mol_name(mol)
        name_to_mol[name] = mol

    selected_indices = []
    selected_mols = []

    for idx, row in df.iterrows():
        name = row.get("name", "")
        mol = name_to_mol.get(name)
        if mol is None:
            selected_indices.append(idx)
            continue

        # Check similarity against already selected
        too_similar = False
        for sel_mol in selected_mols:
            try:
                sim = compute_tanimoto_similarity(mol, sel_mol)
                if sim > similarity_cutoff:
                    too_similar = True
                    break
            except Exception:
                pass

        if not too_similar:
            selected_indices.append(idx)
            selected_mols.append(mol)

        if len(selected_indices) >= top_n * 2:
            break

    return df.loc[selected_indices].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _load_df(source, description: str) -> pd.DataFrame:
    """Load a DataFrame from path or return if already a DataFrame."""
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, str):
        validate_file_exists(source, description)
        return pd.read_csv(source)
    raise ValueError(f"Expected str path or DataFrame for {description}")
