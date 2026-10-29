"""
DrugLab ADMET Screening Module.

Predicts Absorption, Distribution, Metabolism, Excretion, and Toxicity (ADMET)
properties for a set of molecules.

Primary backend: ADMET-AI (comprehensive, ~104 endpoints).
Fallback backend: RDKit descriptor-based heuristics when ADMET-AI is unavailable.

Features:
    - Three threshold profiles: conservative, balanced, permissive
    - BBB handling with target-type awareness (general, CNS, peripheral)
    - ADMET risk score (0-100) with per-endpoint breakdown
    - Structured reporting with BBB status and risk metrics
"""

import os
import time
import logging
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from utils.logger import setup_logger, log_step, suppress_stdout_stderr
from utils.io import (
    ensure_dir,
    write_json,
    read_sdf,
    write_sdf,
    write_csv,
    validate_file_exists,
)
from utils.chemistry import get_mol_name, calculate_basic_properties

logger = setup_logger("admet", level="INFO")

# Try importing ADMET-AI
_HAS_ADMET_AI = False
try:
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"admet_ai.*")
    warnings.filterwarnings("ignore", message=r".*path is deprecated.*")
    warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated.*")
    from admet_ai import ADMETModel
    _HAS_ADMET_AI = True
    logger.debug("ADMET-AI backend available.")
except ImportError:
    logger.debug("ADMET-AI not available; will use RDKit descriptor fallback.")


# ──────────────────────────────────────────────────────────────
# Threshold Profiles
# ──────────────────────────────────────────────────────────────
# Calibrated against DrugBank-approved drug percentiles from ADMET-AI.
# Each profile balances sensitivity (catching problematic compounds)
# vs. hit retention (keeping viable drug candidates).

THRESHOLD_PROFILES = {
    # Conservative: Minimize false negatives. Strict cutoffs.
    # ~60-70% of approved drugs would pass.
    "conservative": {
        # Toxicity (lower cutoffs = stricter)
        "ClinTox":              {"max": 0.3},
        "hERG":                 {"max": 0.3},
        "AMES":                 {"max": 0.3},
        "DILI":                 {"max": 0.4},
        "Carcinogens_Lagunin":  {"max": 0.3},
        "Skin_Reaction":        {"max": 0.4},
        # Absorption (higher mins = stricter)
        "HIA_Hou":              {"min": 0.7},
        "Bioavailability_Ma":   {"min": 0.5},
        "Caco2_Wang":           {"min": -6.0},
        "PAMPA_NCATS":          {"min": 0.5},
        # Metabolism — all CYP isoforms checked
        "CYP1A2_Veith":         {"max": 0.5},
        "CYP2C9_Veith":         {"max": 0.5},
        "CYP2C19_Veith":        {"max": 0.5},
        "CYP2D6_Veith":         {"max": 0.5},
        "CYP3A4_Veith":         {"max": 0.5},
    },

    # Balanced: Default. Reasonable cutoffs for virtual screening.
    # ~80% of approved drugs would pass.
    "balanced": {
        # Toxicity
        "ClinTox":              {"max": 0.5},
        "hERG":                 {"max": 0.5},
        "AMES":                 {"max": 0.5},
        "DILI":                 {"max": 0.6},
        "Carcinogens_Lagunin":  {"max": 0.5},
        "Skin_Reaction":        {"max": 0.6},
        # Absorption
        "HIA_Hou":              {"min": 0.5},
        "Bioavailability_Ma":   {"min": 0.4},
        # Metabolism — major CYP only
        "CYP2D6_Veith":         {"max": 0.6},
        "CYP3A4_Veith":         {"max": 0.6},
    },

    # Permissive: Maximize hit retention. Only flag severe issues.
    # ~90%+ of approved drugs would pass.
    "permissive": {
        # Toxicity — only the most critical
        "ClinTox":              {"max": 0.7},
        "hERG":                 {"max": 0.7},
        "AMES":                 {"max": 0.7},
        "DILI":                 {"max": 0.8},
        # Absorption — minimal filtering
        "HIA_Hou":              {"min": 0.3},
        # Metabolism — informational only (no hard cutoffs)
    },
}

# BBB Thresholds
BBB_THRESHOLDS = {
    "pass": 0.7,      # BBB_Martins >= 0.7 → PASS
    "warning": 0.3,   # 0.3 <= BBB_Martins < 0.7 → WARNING
    # < 0.3 → FAIL
}

# CNS-related keywords for auto-detection
CNS_KEYWORDS = {
    "epilepsy", "brain", "cns", "alzheimer", "parkinson", "glioblastoma",
    "seizure", "neurological", "neuropathic", "schizophrenia", "depression",
    "anxiety", "migraine", "stroke", "dementia", "huntington", "multiple sclerosis",
    "gaba", "serotonin", "dopamine", "glutamate", "benzodiazepine",
}


# ──────────────────────────────────────────────────────────────
# Risk Score Weights
# ──────────────────────────────────────────────────────────────
# Each endpoint's maximum contribution to the 0-100 risk score.
# Weights are proportional to clinical importance.

RISK_WEIGHTS = {
    "hERG":                 20,    # Cardiotoxicity (most critical)
    "DILI":                 15,    # Hepatotoxicity
    "AMES":                 15,    # Mutagenicity
    "ClinTox":              10,    # Clinical toxicity
    "Carcinogens_Lagunin":  5,     # Carcinogenicity
    "HIA_Hou":              10,    # Absorption (inverted: low = bad)
    "BBB_Martins":          5,     # Context-dependent
    "CYP_inhibition":       10,    # Aggregate CYP inhibition
    "Clearance":            10,    # Metabolic stability
}


# Fallback RDKit-based thresholds (used when ADMET-AI is unavailable)
FALLBACK_THRESHOLDS = {
    "hia_proxy":       {"min": 0.3},   # TPSA < 140 → good HIA
    "bbb_proxy":       {"max": 0.7},   # TPSA < 90 → BBB permeable
    "logp_safe":       {"min": -2.0, "max": 5.0},
    "mw_safe":         {"max": 600.0},
    "tpsa_absorption":  {"max": 140.0},
}


# ──────────────────────────────────────────────────────────────
# BBB Handling
# ──────────────────────────────────────────────────────────────

def evaluate_bbb(bbb_score: float, target_type: str = "general") -> dict:
    """
    Evaluate BBB permeability based on BBB_Martins prediction.

    Args:
        bbb_score: BBB_Martins prediction (0-1, higher = more permeable).
        target_type: 'general', 'cns', or 'peripheral'.

    Returns:
        Dict with BBB_status, BBB_note, and BBB_impact.
    """
    if bbb_score >= BBB_THRESHOLDS["pass"]:
        bbb_status = "PASS"
    elif bbb_score >= BBB_THRESHOLDS["warning"]:
        bbb_status = "WARNING"
    else:
        bbb_status = "FAIL"

    bbb_note = ""
    bbb_impact = "neutral"

    if target_type == "cns":
        if bbb_status == "PASS":
            bbb_note = "Good BBB penetration — desirable for CNS target."
            bbb_impact = "positive"
        elif bbb_status == "FAIL":
            bbb_note = "Poor BBB penetration — problematic for CNS target."
            bbb_impact = "negative"
        else:
            bbb_note = "Uncertain BBB penetration — may limit CNS efficacy."
            bbb_impact = "cautionary"

    elif target_type == "peripheral":
        if bbb_status == "PASS":
            bbb_note = "High BBB penetration — may cause CNS side effects for peripheral target."
            bbb_impact = "cautionary"
        elif bbb_status == "FAIL":
            bbb_note = "Low BBB penetration — desirable for peripheral target."
            bbb_impact = "positive"
        else:
            bbb_note = "Moderate BBB penetration — acceptable for peripheral target."
            bbb_impact = "neutral"

    else:  # general
        bbb_note = f"BBB permeability: {bbb_status}. Informational only."
        bbb_impact = "neutral"

    return {
        "BBB_status": bbb_status,
        "BBB_score": round(bbb_score, 4),
        "BBB_note": bbb_note,
        "BBB_impact": bbb_impact,
    }


def detect_target_type(project_name: str | None = None, config: dict | None = None) -> str:
    """
    Auto-detect target type from project name or config keywords.

    Returns 'cns', 'peripheral', or 'general'.
    """
    if config and config.get("target_type", "auto") != "auto":
        return config["target_type"]

    search_text = ""
    if project_name:
        search_text = project_name.lower()
    if config:
        search_text += " " + str(config).lower()

    for keyword in CNS_KEYWORDS:
        if keyword in search_text:
            logger.info(f"Auto-detected CNS target type (keyword: '{keyword}')")
            return "cns"

    return "general"


# ──────────────────────────────────────────────────────────────
# ADMET Risk Score
# ──────────────────────────────────────────────────────────────

def calculate_admet_risk_score(predictions: dict, target_type: str = "general") -> dict:
    """
    Calculate an explainable ADMET risk score (0-100).

    Args:
        predictions: Dict of ADMET endpoint → predicted value.
        target_type: 'general', 'cns', or 'peripheral'.

    Returns:
        Dict with 'admet_risk_score' and 'admet_risk_breakdown'.
    """
    breakdown = {}
    total_score = 0.0

    # hERG — higher = more cardiotoxic risk
    herg = predictions.get("hERG", 0.0)
    herg_contribution = herg * RISK_WEIGHTS["hERG"]
    breakdown["hERG"] = round(herg_contribution, 1)
    total_score += herg_contribution

    # DILI — higher = more hepatotoxic risk
    dili = predictions.get("DILI", 0.0)
    dili_contribution = dili * RISK_WEIGHTS["DILI"]
    breakdown["DILI"] = round(dili_contribution, 1)
    total_score += dili_contribution

    # AMES — higher = more mutagenic risk
    ames = predictions.get("AMES", 0.0)
    ames_contribution = ames * RISK_WEIGHTS["AMES"]
    breakdown["AMES"] = round(ames_contribution, 1)
    total_score += ames_contribution

    # ClinTox
    clintox = predictions.get("ClinTox", 0.0)
    clintox_contribution = clintox * RISK_WEIGHTS["ClinTox"]
    breakdown["ClinTox"] = round(clintox_contribution, 1)
    total_score += clintox_contribution

    # Carcinogens
    carc = predictions.get("Carcinogens_Lagunin", 0.0)
    carc_contribution = carc * RISK_WEIGHTS["Carcinogens_Lagunin"]
    breakdown["Carcinogens_Lagunin"] = round(carc_contribution, 1)
    total_score += carc_contribution

    # HIA — inverted: low absorption = high risk
    hia = predictions.get("HIA_Hou", 1.0)
    hia_contribution = (1.0 - hia) * RISK_WEIGHTS["HIA_Hou"]
    breakdown["HIA_Hou"] = round(hia_contribution, 1)
    total_score += hia_contribution

    # BBB — context-dependent
    bbb = predictions.get("BBB_Martins", 0.5)
    bbb_weight = RISK_WEIGHTS["BBB_Martins"]
    if target_type == "cns":
        # For CNS: low BBB = high risk
        bbb_contribution = (1.0 - bbb) * bbb_weight
    elif target_type == "peripheral":
        # For peripheral: high BBB = risk (CNS side effects)
        bbb_contribution = bbb * bbb_weight
    else:
        # General: BBB is informational, minimal contribution
        bbb_contribution = abs(bbb - 0.5) * bbb_weight * 0.5
    breakdown["BBB_Martins"] = round(bbb_contribution, 1)
    total_score += bbb_contribution

    # CYP inhibition — aggregate across major isoforms
    cyp_endpoints = ["CYP1A2_Veith", "CYP2C9_Veith", "CYP2C19_Veith",
                     "CYP2D6_Veith", "CYP3A4_Veith"]
    cyp_values = [predictions.get(ep, 0.0) for ep in cyp_endpoints]
    cyp_avg = sum(cyp_values) / len(cyp_values) if cyp_values else 0.0
    cyp_contribution = cyp_avg * RISK_WEIGHTS["CYP_inhibition"]
    breakdown["CYP_inhibition"] = round(cyp_contribution, 1)
    total_score += cyp_contribution

    # Clearance — high clearance = fast metabolism = risk
    clearance_hep = predictions.get("Clearance_Hepatocyte_AZ", 50.0)
    clearance_mic = predictions.get("Clearance_Microsome_AZ", 50.0)
    # Normalize: > 100 µL/min/mg is high clearance
    clearance_normalized = min(max((clearance_hep + clearance_mic) / 2.0, 0.0), 200.0) / 200.0
    clearance_contribution = clearance_normalized * RISK_WEIGHTS["Clearance"]
    breakdown["Clearance"] = round(clearance_contribution, 1)
    total_score += clearance_contribution

    # Clamp to 0-100
    total_score = min(max(round(total_score, 1), 0.0), 100.0)

    return {
        "admet_risk_score": total_score,
        "admet_risk_breakdown": breakdown,
    }


# ──────────────────────────────────────────────────────────────
# Interactive Profile Selection
# ──────────────────────────────────────────────────────────────

def prompt_admet_profile_selection() -> str:
    """
    Prompt user to select an ADMET threshold profile.

    Returns:
        Profile name ('conservative', 'balanced', or 'permissive').
    """
    print()
    print("ADMET Screening Profile:")
    print()
    print("  [1] Conservative — strict cutoffs, minimize false negatives")
    print("  [2] Balanced     — reasonable defaults for virtual screening (recommended)")
    print("  [3] Permissive   — relaxed cutoffs, maximize hit retention")
    print()
    print("Choice [2]: ", end="", flush=True)

    try:
        raw = input().strip()
    except (EOFError, KeyboardInterrupt):
        print("\nUsing balanced profile.")
        return "balanced"

    profiles = {"1": "conservative", "2": "balanced", "3": "permissive"}
    choice = profiles.get(raw, "balanced")
    print(f"Using ADMET profile: {choice}")
    return choice


# ──────────────────────────────────────────────────────────────
# Main Screening Function
# ──────────────────────────────────────────────────────────────

def screen_admet(
    input_sdf: str,
    output_dir: str,
    profile: str = "balanced",
    target_type: str = "auto",
    filter_toxic: bool = True,
    filter_low_absorption: bool = True,
    thresholds: dict | None = None,
    backend: str = "auto",
    project_name: str | None = None,
) -> dict:
    """
    Screen molecules for ADMET properties.

    Args:
        input_sdf: Path to input SDF file.
        output_dir: Path to the output directory.
        profile: Threshold profile — 'conservative', 'balanced', or 'permissive'.
        target_type: 'auto', 'general', 'cns', or 'peripheral'.
        filter_toxic: If True, remove molecules flagged as toxic.
        filter_low_absorption: If True, remove molecules with poor absorption.
        thresholds: Dict of property → {min, max} threshold overrides.
        backend: "admet_ai", "rdkit", or "auto" (try ADMET-AI first).
        project_name: Project name for auto target-type detection.

    Returns:
        Dict containing screening results and output file paths.
    """
    start_time = time.time()
    log_step(logger, "ADMET Screening", "started", {
        "input": input_sdf, "output_dir": output_dir,
        "profile": profile, "target_type": target_type,
    })

    if not os.path.exists(input_sdf):
        raise FileNotFoundError(
            f"Input not found: {input_sdf}\n"
            f"  Expected a valid file or directory path."
        )

    # Resolve target type
    resolved_target_type = detect_target_type(
        project_name=project_name,
        config={"target_type": target_type},
    )
    logger.info(f"ADMET profile: {profile}, Target type: {resolved_target_type}")

    # Output paths
    admet_dir = os.path.join(output_dir, "admet")
    ensure_dir(admet_dir)

    passed_sdf_path = os.path.join(admet_dir, "passed_ligands.sdf")
    failed_sdf_path = os.path.join(admet_dir, "failed_ligands.sdf")
    csv_path = os.path.join(admet_dir, "admet_results.csv")
    report_path = os.path.join(admet_dir, "admet_report.json")

    # Load molecules — support both files and directories
    if os.path.isdir(input_sdf):
        from core.ligand_preparation import load_ligands
        logger.info(f"Loading ligands from directory: {input_sdf}")
        mols = load_ligands(input_sdf)
    else:
        mols = read_sdf(input_sdf)
    logger.info(f"Loaded {len(mols)} molecules for ADMET screening.")

    # Determine backend
    use_admet_ai = False
    if backend == "auto":
        use_admet_ai = _HAS_ADMET_AI
    elif backend == "admet_ai":
        if not _HAS_ADMET_AI:
            raise ImportError("ADMET-AI requested but not installed.")
        use_admet_ai = True

    # Select thresholds from profile
    if use_admet_ai:
        active_thresholds = dict(THRESHOLD_PROFILES.get(profile, THRESHOLD_PROFILES["balanced"]))
    else:
        active_thresholds = dict(FALLBACK_THRESHOLDS)

    # Apply user overrides
    if thresholds:
        active_thresholds.update(thresholds)

    # Run predictions
    if use_admet_ai:
        logger.info("Using ADMET-AI backend for predictions.")
        prediction_df = _predict_admet_ai(mols)
    else:
        logger.info("Using RDKit descriptor fallback for ADMET estimation.")
        prediction_df = _predict_rdkit_fallback(mols)

    # Apply filters
    passed_mols = []
    failed_mols = []
    passed_records = []
    failed_records = []

    for idx, mol in enumerate(mols):
        if mol is None:
            continue

        name = get_mol_name(mol, idx)
        row = prediction_df.iloc[idx] if idx < len(prediction_df) else pd.Series()

        violations = []

        # Check each threshold
        for prop, bounds in active_thresholds.items():
            if prop not in row.index:
                continue
            val = row[prop]
            if pd.isna(val):
                continue

            if "min" in bounds and val < bounds["min"]:
                violations.append(f"{prop} = {val:.3f} < {bounds['min']}")
            if "max" in bounds and val > bounds["max"]:
                violations.append(f"{prop} = {val:.3f} > {bounds['max']}")

        # Categorize violations into FAIL and WARNING
        fail_violations = []
        warning_violations = []

        for v in violations:
            prop_name = v.split(" = ")[0].strip()

            is_toxic_fail = prop_name in (
                "ClinTox", "hERG", "AMES", "toxicity_proxy",
            )
            is_absorption_fail = prop_name in (
                "HIA_Hou", "Bioavailability_Ma", "Caco2_Wang",
                "PAMPA_NCATS", "hia_proxy", "tpsa_absorption",
            )

            if filter_toxic and is_toxic_fail:
                fail_violations.append(v)
            elif filter_low_absorption and is_absorption_fail:
                fail_violations.append(v)
            else:
                warning_violations.append(v)

        should_fail = len(fail_violations) > 0

        if should_fail:
            status = "FAIL"
        elif len(warning_violations) > 0:
            status = "WARNING"
        else:
            status = "PASS"

        # BBB evaluation
        bbb_info = {"BBB_status": "N/A", "BBB_score": 0.0, "BBB_note": "", "BBB_impact": "neutral"}
        if use_admet_ai and "BBB_Martins" in row.index:
            bbb_score = row["BBB_Martins"]
            if not pd.isna(bbb_score):
                bbb_info = evaluate_bbb(bbb_score, resolved_target_type)

        # Risk score calculation
        risk_info = {"admet_risk_score": 0.0, "admet_risk_breakdown": {}}
        if use_admet_ai:
            pred_dict = {col: row[col] for col in row.index if not pd.isna(row[col])}
            risk_info = calculate_admet_risk_score(pred_dict, resolved_target_type)

        record = {
            "name": name,
            "admet_status": status,
            "admet_warnings": "; ".join(warning_violations) if warning_violations else "none",
            "admet_failures": "; ".join(fail_violations) if fail_violations else "none",
            "violations": violations,
            "violation_count": len(violations),
            # BBB
            "BBB_status": bbb_info["BBB_status"],
            "BBB_score": bbb_info["BBB_score"],
            "BBB_note": bbb_info["BBB_note"],
            "BBB_impact": bbb_info["BBB_impact"],
            # Risk score
            "admet_risk_score": risk_info["admet_risk_score"],
            "admet_risk_breakdown": risk_info["admet_risk_breakdown"],
        }

        # Add all prediction columns
        for col in row.index:
            val = row[col]
            if isinstance(val, float):
                record[col] = round(val, 4)
            else:
                record[col] = val

        # Tag mol properties
        mol.SetProp("admet_status", status)
        mol.SetProp("admet_passed", str(not should_fail))
        mol.SetProp("admet_warnings", record["admet_warnings"])
        mol.SetProp("admet_failures", record["admet_failures"])
        mol.SetProp("admet_violations", "; ".join(violations) if violations else "none")
        mol.SetProp("BBB_status", bbb_info["BBB_status"])
        mol.SetProp("BBB_score", str(bbb_info["BBB_score"]))
        mol.SetProp("admet_risk_score", str(risk_info["admet_risk_score"]))

        if should_fail:
            failed_mols.append(mol)
            failed_records.append(record)
            logger.info(f"  ✗ {name}: FAIL (risk={risk_info['admet_risk_score']:.0f}). "
                       f"Failures: {', '.join(fail_violations)}")
        else:
            passed_mols.append(mol)
            passed_records.append(record)
            if status == "WARNING":
                logger.info(f"  ⚠ {name}: WARNING (risk={risk_info['admet_risk_score']:.0f}, "
                           f"BBB={bbb_info['BBB_status']}). Warnings: {', '.join(warning_violations)}")
            else:
                logger.info(f"  ✓ {name}: PASS (risk={risk_info['admet_risk_score']:.0f}, "
                           f"BBB={bbb_info['BBB_status']})")

    # Write outputs
    write_sdf(passed_mols, passed_sdf_path)
    write_sdf(failed_mols, failed_sdf_path)

    all_records = passed_records + failed_records
    if all_records:
        df_out = pd.DataFrame(all_records)
        write_csv(df_out, csv_path)

    duration = time.time() - start_time
    metrics = {
        "profile": profile,
        "target_type": resolved_target_type,
        "total_input": len(mols),
        "total_passed": len(passed_mols),
        "total_failed": len(failed_mols),
        "backend": "admet_ai" if use_admet_ai else "rdkit_fallback",
        "passed_sdf": passed_sdf_path,
        "failed_sdf": failed_sdf_path,
        "admet_csv": csv_path,
        "passed": passed_records,
        "failed": failed_records,
        "duration": round(duration, 2),
        "status": "completed",
    }

    # ── Figure Generation ──
    try:
        from reports.generator import generate_admet_figures
        if all_records:
            plot_df = pd.DataFrame(all_records)
            # We must map admet_status to ADMET_Status for the plot
            plot_df["ADMET_Status"] = [rec.get("admet_status") for rec in all_records]
            plot_df["Risk_Score"] = [rec.get("admet_risk_score") for rec in all_records]
            plot_df["name"] = [rec.get("name") for rec in all_records]
            
            generated_figs = generate_admet_figures(plot_df, output_dir)
            metrics["generated_figures"] = generated_figs
        else:
            metrics["generated_figures"] = []
    except Exception as e:
        logger.error(f"Failed to generate ADMET figures: {e}")
        metrics["generated_figures"] = []

    write_json(metrics, report_path)
    
    log_dict = {
        "profile": profile,
        "target_type": resolved_target_type,
        "passed_count": len(passed_mols),
        "failed_count": len(failed_mols),
        "backend": metrics["backend"],
        "duration": f"{duration:.1f}s",
    }
    
    if metrics["generated_figures"]:
        fig_msg = "\nGenerated Figures:\n"
        for f in metrics["generated_figures"]:
            fig_msg += f"  ✓ {os.path.basename(f)}\n"
        log_dict["figures"] = fig_msg

    log_step(logger, "ADMET Screening", "completed", log_dict)
    
    if metrics["generated_figures"]:
        print(f"\nGenerated {len(metrics['generated_figures'])} figures:")
        print(f"{os.path.join(output_dir, 'figures', 'admet')}/")
        for f in metrics["generated_figures"]:
            print(f"  ✓ {os.path.basename(f)}")
        print()

    return metrics


# ──────────────────────────────────────────────────────────────
# ADMET-AI Backend
# ──────────────────────────────────────────────────────────────

def _predict_admet_ai(mols: list) -> pd.DataFrame:
    """Run ADMET-AI predictions on a list of RDKit Mol objects silently."""
    import warnings
    warnings.filterwarnings("ignore")
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    smiles_list = []
    for mol in mols:
        if mol is not None:
            smi = Chem.MolToSmiles(mol)
            smiles_list.append(smi)
        else:
            smiles_list.append("")

    with suppress_stdout_stderr():
        model = ADMETModel()
        predictions = model.predict(smiles=smiles_list)

    return predictions


# ──────────────────────────────────────────────────────────────
# RDKit Fallback Backend
# ──────────────────────────────────────────────────────────────

def _predict_rdkit_fallback(mols: list) -> pd.DataFrame:
    """
    Estimate ADMET-like properties from RDKit descriptors.

    This is a heuristic fallback when ADMET-AI is not installed.
    """
    records = []

    for mol in mols:
        if mol is None:
            records.append({})
            continue

        mol_noH = Chem.RemoveHs(mol)

        mw = Descriptors.MolWt(mol_noH)
        logp = Descriptors.MolLogP(mol_noH)
        tpsa = Descriptors.TPSA(mol_noH)
        hbd = rdMolDescriptors.CalcNumHBD(mol_noH)
        hba = rdMolDescriptors.CalcNumHBA(mol_noH)
        rotbonds = rdMolDescriptors.CalcNumRotatableBonds(mol_noH)
        rings = rdMolDescriptors.CalcNumRings(mol_noH)
        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol_noH)

        # Heuristic ADMET proxies
        hia_proxy = 1.0 if (tpsa < 140 and logp > -2.0) else 0.0
        bbb_proxy = 1.0 if (tpsa < 90 and mw < 400 and 0 < logp < 3) else 0.0
        pgp_proxy = 1.0 if (mw > 400 or hbd > 3) else 0.0
        cyp_risk = 1.0 if (aromatic_rings >= 3 and logp > 3.0) else 0.0

        has_basic_n = mol_noH.HasSubstructMatch(Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")) if mol_noH else False
        herg_proxy = 1.0 if (logp > 3.5 and has_basic_n) else 0.0

        ames_alerts = ["[N+](=O)[O-]", "[NX2]=O", "N=N"]
        ames_proxy = 0.0
        for smarts in ames_alerts:
            pat = Chem.MolFromSmarts(smarts)
            if pat and mol_noH.HasSubstructMatch(pat):
                ames_proxy = 1.0
                break

        toxicity_proxy = max(herg_proxy, ames_proxy)

        records.append({
            "MW": round(mw, 2),
            "LogP": round(logp, 2),
            "TPSA": round(tpsa, 2),
            "HBD": hbd,
            "HBA": hba,
            "RotBonds": rotbonds,
            "hia_proxy": hia_proxy,
            "bbb_proxy": bbb_proxy,
            "BBB_Martins": bbb_proxy,  # Map to standard name for BBB evaluation
            "pgp_proxy": pgp_proxy,
            "cyp_risk": cyp_risk,
            "hERG": herg_proxy,
            "AMES": ames_proxy,
            "tpsa_absorption": tpsa,
            "logp_safe": logp,
            "mw_safe": mw,
            "toxicity_proxy": toxicity_proxy,
        })

    return pd.DataFrame(records)
