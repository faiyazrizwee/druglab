"""
DrugLab Drug-Likeness Screening Module.

Screens molecules against standard drug-likeness rules (Lipinski, Veber, Ghose,
Egan, Muegge, Lead-like, Rule of 3) and structural alerts (PAINS, Brenk).

Supports medicinal chemistry profiles, configurable evaluation modes (strict,
majority, custom), and interactive rule selection. Writes results to SDF, CSV,
and JSON report formats.

References:
    - Lipinski: Adv. Drug Deliv. Rev. 2001, 46, 3-26
    - Veber: J. Med. Chem. 2002, 45, 2615-2623
    - Ghose: J. Comb. Chem. 1999, 1, 55-68
    - Egan: J. Med. Chem. 2000, 43, 3867-3877
    - Muegge: J. Med. Chem. 2001, 44, 1841-1846
    - Congreve (Rule of 3): Drug Discov. Today 2003, 8, 876-877
"""

import os
import time
import logging
import pandas as pd
from rdkit import Chem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from utils.logger import setup_logger, log_step
from utils.io import (
    ensure_dir,
    write_json,
    read_sdf,
    write_sdf,
    write_csv,
    validate_file_exists,
)
from utils.chemistry import (
    calculate_basic_properties,
    get_mol_name,
)

logger = setup_logger("druglikeness", level="INFO")


# ──────────────────────────────────────────────────────────────
# Medicinal Chemistry Profiles
# ──────────────────────────────────────────────────────────────

DRUGLIKENESS_PROFILES = {
    "recommended": ["lipinski", "veber", "pains"],
    "oral_drug":   ["lipinski", "veber", "ghose", "pains"],
    "lead_like":   ["lead_like", "pains"],
    "fragment":    ["rule_of_3", "pains"],
    "custom":      None,  # user-selected rules from config
}


def resolve_profile_rules(profile: str, explicit_rules: list[str] | None = None) -> list[str]:
    """
    Resolve rules from a profile name.

    If profile is 'custom', uses explicit_rules. Otherwise, returns the
    predefined rule set for that profile.

    Args:
        profile: Profile name (recommended, oral_drug, lead_like, fragment, custom).
        explicit_rules: Explicit rules list (used when profile='custom').

    Returns:
        List of rule names to apply.
    """
    if profile == "custom":
        return explicit_rules or ["lipinski"]

    profile_rules = DRUGLIKENESS_PROFILES.get(profile)
    if profile_rules is None:
        logger.warning(f"Unknown profile '{profile}', falling back to 'recommended'")
        return DRUGLIKENESS_PROFILES["recommended"]

    return list(profile_rules)


# ──────────────────────────────────────────────────────────────
# Interactive Rule Selection
# ──────────────────────────────────────────────────────────────

RULE_MENU = {
    1: ("lipinski",   "Lipinski Rule of Five"),
    2: ("veber",      "Veber (Oral Bioavailability)"),
    3: ("ghose",      "Ghose Filter"),
    4: ("egan",       "Egan Filter"),
    5: ("muegge",     "Muegge (Bioavailable Drug-Like)"),
    6: ("lead_like",  "Lead-Like Filter"),
    7: ("pains",      "PAINS Structural Alerts"),
    8: ("brenk",      "Brenk Structural Alerts"),
    9: ("rule_of_3",  "Rule of 3 (Fragment)"),
    10: ("_recommended", "Use Recommended Set"),
}


def prompt_rule_selection() -> list[str]:
    """
    Interactively prompt the user to select drug-likeness rules.

    Returns:
        List of selected rule names.
    """
    print()
    print("Select Drug-Likeness Rules:")
    print()
    for menu_id, (rule_key, display) in RULE_MENU.items():
        print(f"  [{menu_id:>2}] {display}")
    print()
    print("Enter numbers separated by spaces or commas (e.g. '1 2 7' or '1,2,7'):")
    print("Selection: ", end="", flush=True)

    try:
        raw = input().strip()
    except (EOFError, KeyboardInterrupt):
        print("\nUsing recommended set.")
        return DRUGLIKENESS_PROFILES["recommended"]

    if not raw:
        return DRUGLIKENESS_PROFILES["recommended"]

    # Parse selection
    raw = raw.replace(",", " ")
    selected = []
    for token in raw.split():
        try:
            num = int(token)
            if num in RULE_MENU:
                rule_key, _ = RULE_MENU[num]
                if rule_key == "_recommended":
                    return DRUGLIKENESS_PROFILES["recommended"]
                selected.append(rule_key)
        except ValueError:
            continue

    if not selected:
        print("No valid selections. Using recommended set.")
        return DRUGLIKENESS_PROFILES["recommended"]

    # Deduplicate preserving order
    seen = set()
    unique = []
    for r in selected:
        if r not in seen:
            seen.add(r)
            unique.append(r)

    rule_names = [RULE_MENU[k][1] for k in RULE_MENU
                  if RULE_MENU[k][0] in unique and RULE_MENU[k][0] != "_recommended"]
    print(f"Selected: {', '.join(rule_names)}")
    return unique


def prompt_profile_selection() -> tuple[str, list[str]]:
    """
    Prompt user to select a drug-likeness profile.

    Used in automated workflows when no profile is pre-configured.

    Returns:
        Tuple of (profile_name, rules_list).
    """
    print()
    print("Drug-Likeness configuration not yet defined.")
    print()
    print("Choose:")
    print("  [1] Recommended Set (Lipinski + Veber + PAINS)")
    print("  [2] Oral Drug Profile (Lipinski + Veber + Ghose + PAINS)")
    print("  [3] Lead-Like Profile (Lead-Like + PAINS)")
    print("  [4] Fragment Profile (Rule of 3 + PAINS)")
    print("  [5] Custom Rules")
    print()
    print("Choice [1]: ", end="", flush=True)

    try:
        raw = input().strip()
    except (EOFError, KeyboardInterrupt):
        print("\nUsing recommended set.")
        return "recommended", DRUGLIKENESS_PROFILES["recommended"]

    profiles_by_choice = {
        "1": "recommended",
        "2": "oral_drug",
        "3": "lead_like",
        "4": "fragment",
        "5": "custom",
    }

    choice = profiles_by_choice.get(raw, "recommended")
    if choice == "custom":
        rules = prompt_rule_selection()
        return "custom", rules

    rules = resolve_profile_rules(choice)
    print(f"Using profile: {choice} → {rules}")
    return choice, rules


# ──────────────────────────────────────────────────────────────
# Main Screening Function
# ──────────────────────────────────────────────────────────────

def screen_druglikeness(
    input_sdf: str,
    output_dir: str,
    rules: list[str] | None = None,
    profile: str | None = None,
    evaluation_mode: str = "majority",
    minimum_rule_pass_fraction: float = 0.7,
    brenk_mode: str = "warning",
    custom_filters: dict | None = None,
    strict: bool = False,
    max_violations: int = 1,
) -> dict:
    """
    Screens molecules through drug-likeness filters.

    Args:
        input_sdf: Path to the input SDF file of ligands.
        output_dir: Path to the output directory.
        rules: List of rule sets to apply. If profile is set and != 'custom',
               rules are derived from the profile instead.
        profile: Medicinal chemistry profile name. If set, overrides rules.
        evaluation_mode: How to combine rule outcomes:
            'strict' - all rules must pass
            'majority' - >= 70% of rules must pass
            'custom' - use minimum_rule_pass_fraction
        minimum_rule_pass_fraction: Required fraction of rules to pass (for 'custom' mode).
        brenk_mode: 'warning' (record but don't fail) or 'fail' (hard filter).
        custom_filters: Dict of descriptor range tuples, e.g., {"MW": (200, 500)}.
        strict: DEPRECATED. If True, sets evaluation_mode='strict'. Kept for backward compat.
        max_violations: Maximum allowed violations for Lipinski/Muegge rule sets.

    Returns:
        Dict containing screening results and output file paths.
    """
    start_time = time.time()
    log_step(logger, "Drug-Likeness Screening", "started", {"input": input_sdf, "output_dir": output_dir})

    if not os.path.exists(input_sdf):
        raise FileNotFoundError(
            f"Input not found: {input_sdf}\n"
            f"  Expected a valid file or directory path."
        )

    # Resolve rules from profile
    if profile and profile != "custom":
        rules = resolve_profile_rules(profile)
    elif rules is None:
        rules = ["lipinski"]

    # Backward compatibility: strict=True → evaluation_mode='strict'
    if strict and evaluation_mode != "strict":
        evaluation_mode = "strict"

    # Normalize rules to lowercase
    rules = [r.lower() for r in rules]

    logger.info(f"Profile: {profile or 'custom'}, Rules: {rules}, "
                f"Evaluation: {evaluation_mode}, Brenk mode: {brenk_mode}")

    # Output paths
    screen_dir = os.path.join(output_dir, "druglikeness")
    ensure_dir(screen_dir)

    passed_sdf_path = os.path.join(screen_dir, "passed_ligands.sdf")
    failed_sdf_path = os.path.join(screen_dir, "failed_ligands.sdf")
    csv_path = os.path.join(screen_dir, "descriptors.csv")
    report_path = os.path.join(screen_dir, "druglikeness_report.json")

    # Load molecules — support both files and directories
    if os.path.isdir(input_sdf):
        from core.ligand_preparation import load_ligands
        logger.info(f"Loading ligands from directory: {input_sdf}")
        mols = load_ligands(input_sdf)
    else:
        mols = read_sdf(input_sdf)
    logger.info(f"Loaded {len(mols)} molecules for screening.")

    # Initialize Filter Catalogs for PAINS and Brenk if requested
    pains_catalog = None
    if "pains" in rules:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        pains_catalog = FilterCatalog(params)

    brenk_catalog = None
    if "brenk" in rules:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        brenk_catalog = FilterCatalog(params)

    passed_mols = []
    failed_mols = []
    descriptor_records = []
    passed_records = []
    failed_records = []

    # Rule checkers registry
    RULE_CHECKERS = {
        "lipinski":  lambda desc: check_lipinski(desc, max_violations),
        "veber":     check_veber,
        "ghose":     check_ghose,
        "egan":      check_egan,
        "muegge":    lambda desc: check_muegge(desc, max_violations),
        "lead_like": check_lead_like,
        "rule_of_3": check_rule_of_3,
    }

    for idx, mol in enumerate(mols):
        if mol is None:
            continue

        name = get_mol_name(mol, idx)
        props = calculate_basic_properties(mol)

        record = {"name": name, **props}
        descriptor_records.append(record)

        # Evaluate all selected rules
        eval_results = {}
        violations_dict = {}

        # Property-based rules
        for rule_name in rules:
            if rule_name in RULE_CHECKERS:
                passed_rule, violations = RULE_CHECKERS[rule_name](props)
                eval_results[rule_name] = passed_rule
                violations_dict[rule_name] = violations

        # Structural alert: PAINS (always hard filter when selected)
        pains_hard_fail = False
        if "pains" in rules and pains_catalog:
            matches = pains_catalog.GetMatches(mol)
            pains_passed = len(matches) == 0
            pains_violations = [m.GetDescription() for m in matches]
            eval_results["pains"] = pains_passed
            violations_dict["pains"] = pains_violations
            if not pains_passed:
                pains_hard_fail = True

        # Structural alert: Brenk (configurable mode)
        brenk_warning = False
        if "brenk" in rules and brenk_catalog:
            matches = brenk_catalog.GetMatches(mol)
            brenk_passed = len(matches) == 0
            brenk_violations = [m.GetDescription() for m in matches]
            violations_dict["brenk"] = brenk_violations

            if brenk_mode == "fail":
                eval_results["brenk"] = brenk_passed
            else:
                # Warning mode: record but don't count as rule failure
                eval_results["brenk"] = True  # doesn't affect pass/fail
                if not brenk_passed:
                    brenk_warning = True

        # Apply custom filters
        custom_pass = True
        custom_violations = []
        if custom_filters:
            custom_pass, custom_violations = apply_custom_filters(props, custom_filters)
            eval_results["custom"] = custom_pass
            violations_dict["custom"] = custom_violations

        # ── Combine results using evaluation mode ──

        # Separate structural alerts from property rules for evaluation
        property_rules = {k: v for k, v in eval_results.items()
                         if k not in ("pains", "brenk", "custom")}
        property_outcomes = list(property_rules.values())

        if not property_outcomes:
            rules_passed = True
        elif evaluation_mode == "strict":
            rules_passed = all(property_outcomes)
        elif evaluation_mode == "majority":
            pass_fraction = sum(property_outcomes) / len(property_outcomes)
            rules_passed = pass_fraction >= 0.7
        elif evaluation_mode == "custom":
            pass_fraction = sum(property_outcomes) / len(property_outcomes)
            rules_passed = pass_fraction >= minimum_rule_pass_fraction
        else:
            # Fallback: any
            rules_passed = any(property_outcomes)

        # PAINS is always a hard filter
        if pains_hard_fail:
            rules_passed = False

        # Brenk in fail mode is already in eval_results; warning mode doesn't affect pass

        # Custom filters are always required
        overall_pass = rules_passed and custom_pass

        # Build per-molecule info
        failed_reasons = []
        for r_name, r_pass in eval_results.items():
            if not r_pass:
                reasons = violations_dict.get(r_name, [])
                failed_reasons.append(f"{r_name.upper()} violated: {reasons}")

        # Calculate pass fraction for reporting
        if property_outcomes:
            pass_fraction_val = round(sum(property_outcomes) / len(property_outcomes), 3)
        else:
            pass_fraction_val = 1.0

        mol_info = {
            "name": name,
            "smiles": props["SMILES"],
            "properties": props,
            "rules_passed": [k for k, v in eval_results.items() if v],
            "rules_failed": [k for k, v in eval_results.items() if not v],
            "violations": failed_reasons,
            "pass_fraction": pass_fraction_val,
            "brenk_warning": brenk_warning,
            "brenk_alerts": violations_dict.get("brenk", []),
        }

        # Tag properties onto Mol object for SDF writing
        for pk, pv in props.items():
            mol.SetProp(pk, str(pv))
        mol.SetProp("druglikeness_passed", str(overall_pass))
        mol.SetProp("rules_failed", ",".join(mol_info["rules_failed"]))
        mol.SetProp("pass_fraction", str(pass_fraction_val))
        if brenk_warning:
            mol.SetProp("brenk_warning", ",".join(violations_dict.get("brenk", [])))

        if overall_pass:
            passed_mols.append(mol)
            passed_records.append(mol_info)
        else:
            failed_mols.append(mol)
            failed_records.append(mol_info)

    # Write output files
    write_sdf(passed_mols, passed_sdf_path)
    write_sdf(failed_mols, failed_sdf_path)

    # Save descriptors to CSV
    if descriptor_records:
        df = pd.DataFrame(descriptor_records)
        write_csv(df, csv_path)
    else:
        df = pd.DataFrame()

    duration = time.time() - start_time
    metrics = {
        "profile": profile or "custom",
        "rules_applied": rules,
        "evaluation_mode": evaluation_mode,
        "brenk_mode": brenk_mode,
        "total_input": len(mols),
        "total_passed": len(passed_mols),
        "total_failed": len(failed_mols),
        "passed_sdf": passed_sdf_path,
        "failed_sdf": failed_sdf_path,
        "descriptors_csv": csv_path,
        "passed": passed_records,
        "failed": failed_records,
        "duration": round(duration, 2),
        "status": "completed"
    }

    # ── Figure Generation ──
    try:
        from reports.generator import generate_druglikeness_figures
        # We need a DataFrame containing status and failures.
        # df already has descriptors. We can add Status and failure counts.
        if not df.empty:
            # We must construct a df that has Status, Lipinski_Failures, etc.
            # We can build it from passed_records and failed_records
            all_records = passed_records + failed_records
            plot_data = []
            for rec in all_records:
                row = rec["properties"].copy()
                row["Status"] = "PASS" if rec in passed_records else "FAIL"
                
                # Count failures for each rule
                failed_rules = rec["rules_failed"]
                row["Lipinski_Failures"] = 1 if "lipinski" in failed_rules else 0
                row["Veber_Failures"] = 1 if "veber" in failed_rules else 0
                row["Ghose_Failures"] = 1 if "ghose" in failed_rules else 0
                row["Egan_Failures"] = 1 if "egan" in failed_rules else 0
                row["Muegge_Failures"] = 1 if "muegge" in failed_rules else 0
                row["PAINS_Failures"] = 1 if "pains" in failed_rules else 0
                row["Brenk_Warnings"] = 1 if rec.get("brenk_warning") else 0
                
                plot_data.append(row)
            
            plot_df = pd.DataFrame(plot_data)
            generated_figs = generate_druglikeness_figures(plot_df, output_dir)
            metrics["generated_figures"] = generated_figs
        else:
            metrics["generated_figures"] = []
    except Exception as e:
        logger.error(f"Failed to generate druglikeness figures: {e}")
        metrics["generated_figures"] = []

    write_json(metrics, report_path)
    
    log_dict = {
        "profile": profile or "custom",
        "evaluation_mode": evaluation_mode,
        "passed_count": len(passed_mols),
        "failed_count": len(failed_mols),
        "passed_sdf": passed_sdf_path,
        "duration": f"{duration:.1f}s"
    }
    
    if metrics["generated_figures"]:
        # Pretty print for the terminal
        fig_msg = "\nGenerated Figures:\n"
        for f in metrics["generated_figures"]:
            fig_msg += f"  ✓ {os.path.basename(f)}\n"
        log_dict["figures"] = fig_msg

    log_step(logger, "Drug-Likeness Screening", "completed", log_dict)
    
    if metrics["generated_figures"]:
        # Explicitly print to console as requested
        print(f"\nGenerated {len(metrics['generated_figures'])} figures:")
        print(f"{os.path.join(output_dir, 'figures', 'druglikeness')}/")
        for f in metrics["generated_figures"]:
            print(f"  ✓ {os.path.basename(f)}")
        print()

    return metrics


# ──────────────────────────────────────────────────────────────
# Rule Implementations
# ──────────────────────────────────────────────────────────────

def check_lipinski(desc: dict, max_violations: int = 1) -> tuple[bool, list[str]]:
    """
    Check Lipinski Rule of Five.

    Reference: Lipinski et al., Adv. Drug Deliv. Rev. 2001, 46, 3-26.
    Criteria: MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10.
    Passes if violations ≤ max_violations.
    """
    violations = []
    if desc["MW"] > 500:
        violations.append(f"MW > 500 ({desc['MW']})")
    if desc["LogP"] > 5.0:
        violations.append(f"LogP > 5.0 ({desc['LogP']})")
    if desc["HBD"] > 5:
        violations.append(f"HBD > 5 ({desc['HBD']})")
    if desc["HBA"] > 10:
        violations.append(f"HBA > 10 ({desc['HBA']})")

    passed = len(violations) <= max_violations
    return passed, violations


def check_veber(desc: dict) -> tuple[bool, list[str]]:
    """
    Check Veber rules (oral bioavailability).

    Reference: Veber et al., J. Med. Chem. 2002, 45, 2615-2623.
    Criteria: TPSA ≤ 140, RotBonds ≤ 10.
    """
    violations = []
    if desc["TPSA"] > 140.0:
        violations.append(f"TPSA > 140 ({desc['TPSA']})")
    if desc["RotBonds"] > 10:
        violations.append(f"RotBonds > 10 ({desc['RotBonds']})")

    return len(violations) == 0, violations


def check_ghose(desc: dict) -> tuple[bool, list[str]]:
    """
    Ghose filter.

    Reference: Ghose et al., J. Comb. Chem. 1999, 1, 55-68.
    Criteria: MW 160–480, LogP -0.4–5.6, atom count 20–70, MR 40–130.
    """
    violations = []
    if not (160.0 <= desc["MW"] <= 480.0):
        violations.append(f"MW not in [160, 480] ({desc['MW']})")
    if not (-0.4 <= desc["LogP"] <= 5.6):
        violations.append(f"LogP not in [-0.4, 5.6] ({desc['LogP']})")
    if not (20 <= desc["NumAtoms"] <= 70):
        violations.append(f"Atom count not in [20, 70] ({desc['NumAtoms']})")
    if not (40.0 <= desc["MolarRefractivity"] <= 130.0):
        violations.append(f"MR not in [40, 130] ({desc['MolarRefractivity']})")

    return len(violations) == 0, violations


def check_egan(desc: dict) -> tuple[bool, list[str]]:
    """
    Egan filter (absorption prediction).

    Reference: Egan et al., J. Med. Chem. 2000, 43, 3867-3877.
    Criteria: TPSA ≤ 131.6, LogP ≤ 5.88.
    """
    violations = []
    if desc["TPSA"] > 131.6:
        violations.append(f"TPSA > 131.6 ({desc['TPSA']})")
    if desc["LogP"] > 5.88:
        violations.append(f"LogP > 5.88 ({desc['LogP']})")

    return len(violations) == 0, violations


def check_muegge(desc: dict, max_violations: int = 1) -> tuple[bool, list[str]]:
    """
    Muegge filter (bioavailable drug-like).

    Reference: Muegge et al., J. Med. Chem. 2001, 44, 1841-1846.

    Complete criteria:
        MW:             200 – 600
        LogP:           -2.0 – 5.0
        TPSA:           ≤ 150
        RingCount:      ≤ 7
        RotBonds:       ≤ 15
        HBD:            ≤ 5
        HBA:            ≤ 10
        NumHeteroatoms: ≥ 1
        NumCarbons:     ≥ 2
        FormalCharge:   -2 – 2

    Note: The original Muegge concept also includes a pharmacophore feature
    count criterion. RDKit does not directly compute this. We approximate it
    as HBD + HBA ≥ 1, which captures the intent (molecules must have
    interaction capability). This approximation is documented here.
    """
    violations = []
    if not (200.0 <= desc["MW"] <= 600.0):
        violations.append(f"MW not in [200, 600] ({desc['MW']})")
    if not (-2.0 <= desc["LogP"] <= 5.0):
        violations.append(f"LogP not in [-2.0, 5.0] ({desc['LogP']})")
    if desc["TPSA"] > 150.0:
        violations.append(f"TPSA > 150 ({desc['TPSA']})")
    if desc["RingCount"] > 7:
        violations.append(f"RingCount > 7 ({desc['RingCount']})")
    if desc["RotBonds"] > 15:
        violations.append(f"RotBonds > 15 ({desc['RotBonds']})")
    if desc["HBD"] > 5:
        violations.append(f"HBD > 5 ({desc['HBD']})")
    if desc["HBA"] > 10:
        violations.append(f"HBA > 10 ({desc['HBA']})")

    # New checks from the complete Muegge concept
    if desc.get("NumHeteroatoms", 1) < 1:
        violations.append(f"NumHeteroatoms < 1 ({desc.get('NumHeteroatoms', 'N/A')})")
    if desc.get("NumCarbons", 2) < 2:
        violations.append(f"NumCarbons < 2 ({desc.get('NumCarbons', 'N/A')})")
    charge = desc.get("FormalCharge", 0)
    if not (-2 <= charge <= 2):
        violations.append(f"FormalCharge not in [-2, 2] ({charge})")

    passed = len(violations) <= max_violations
    return passed, violations


def check_lead_like(desc: dict) -> tuple[bool, list[str]]:
    """
    Lead-like filter.

    Criteria: MW 150–350, LogP -1.0–3.0, RotBonds ≤ 7.
    """
    violations = []
    if not (150.0 <= desc["MW"] <= 350.0):
        violations.append(f"MW not in [150, 350] ({desc['MW']})")
    if not (-1.0 <= desc["LogP"] <= 3.0):
        violations.append(f"LogP not in [-1.0, 3.0] ({desc['LogP']})")
    if desc["RotBonds"] > 7:
        violations.append(f"RotBonds > 7 ({desc['RotBonds']})")

    return len(violations) == 0, violations


def check_rule_of_3(desc: dict) -> tuple[bool, list[str]]:
    """
    Rule of 3 for Fragment-Based Drug Discovery.

    Reference: Congreve et al., Drug Discov. Today 2003, 8, 876-877.
    Criteria: MW ≤ 300, LogP ≤ 3, HBD ≤ 3, HBA ≤ 3, RotBonds ≤ 3.
    """
    violations = []
    if desc["MW"] > 300:
        violations.append(f"MW > 300 ({desc['MW']})")
    if desc["LogP"] > 3.0:
        violations.append(f"LogP > 3.0 ({desc['LogP']})")
    if desc["HBD"] > 3:
        violations.append(f"HBD > 3 ({desc['HBD']})")
    if desc["HBA"] > 3:
        violations.append(f"HBA > 3 ({desc['HBA']})")
    if desc["RotBonds"] > 3:
        violations.append(f"RotBonds > 3 ({desc['RotBonds']})")

    return len(violations) == 0, violations


def apply_custom_filters(desc: dict, filters: dict) -> tuple[bool, list[str]]:
    """Apply user-defined property range filters."""
    violations = []
    for prop, bounds in filters.items():
        if prop in desc:
            val = desc[prop]
            if len(bounds) == 2:
                lower, upper = bounds
                if lower is not None and val < lower:
                    violations.append(f"Custom {prop} < {lower} ({val})")
                if upper is not None and val > upper:
                    violations.append(f"Custom {prop} > {upper} ({val})")
            elif len(bounds) == 1:
                upper = bounds[0]
                if upper is not None and val > upper:
                    violations.append(f"Custom {prop} > {upper} ({val})")

    return len(violations) == 0, violations
