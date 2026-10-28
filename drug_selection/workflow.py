"""
DrugLab Drug Selection Workflow Orchestrator.

Runs the full drug-selection pipeline:
    Protein Prep → Ligand Prep → Pocket Detection → Drug-Likeness →
    ADMET → Docking → Lead Selection

Supports stop-and-resume at any stage.
"""

import os
import time
import logging

from utils.logger import setup_logger, log_step, log_summary
from utils.config import load_config, save_config_snapshot, DrugLabConfig
from utils.io import ensure_dir

from core.protein_preparation import prepare_protein
from core.ligand_preparation import prepare_ligands
from core.pocket_detection import detect_pockets
from core.druglikeness import screen_druglikeness
from core.admet import screen_admet
from core.docking import run_docking
from drug_selection.lead_selection import select_leads

logger = setup_logger("drug_selection_workflow", level="INFO")

# Ordered list of pipeline stages
STAGES = [
    "protein_preparation",
    "ligand_preparation",
    "pocket_detection",
    "druglikeness",
    "admet",
    "docking",
    "lead_selection",
]


def run_drug_selection(
    protein_pdb: str,
    ligand_library: str,
    output_dir: str = "results",
    config: DrugLabConfig | None = None,
    stop_after: str | None = None,
    resume_from: str | None = None,
) -> dict:
    """
    Run the drug selection pipeline.

    Args:
        protein_pdb: Path to the input protein PDB file.
        ligand_library: Path to the ligand library (SDF file or directory).
        output_dir: Path to the output directory.
        config: DrugLabConfig (loaded from YAML or defaults).
        stop_after: Stop after this stage name.
        resume_from: Resume from this stage name (skip prior stages).

    Returns:
        Dict of stage → result metrics.
    """
    start_time = time.time()
    log_step(logger, "Drug Selection Workflow", "started", {
        "protein": protein_pdb,
        "library": ligand_library,
        "output_dir": output_dir,
    })

    if config is None:
        config = load_config()

    ensure_dir(output_dir)
    save_config_snapshot(config, output_dir)

    # Determine active stages
    active_stages = _resolve_stages(STAGES, stop_after, resume_from)
    logger.info(f"Active stages: {active_stages}")

    results = {}

    # Track outputs passed between stages
    ctx = {
        "protein_pdb": protein_pdb,
        "ligand_library": ligand_library,
        "output_dir": output_dir,
        "clean_pdb": None,
        "protein_pdbqt": None,
        "prepared_ligands_sdf": None,
        "docking_box": None,
        "passed_dl_sdf": None,
        "passed_admet_sdf": None,
        "admet_csv": None,
        "dl_csv": None,
        "docking_csv": None,
    }

    # Execute each active stage
    for stage in active_stages:
        try:
            if stage == "protein_preparation":
                result = prepare_protein(
                    input_pdb=protein_pdb,
                    output_dir=output_dir,
                    chain_id=config.protein.chain_id,
                    remove_water=config.protein.remove_water,
                    remove_heterogens=config.protein.remove_heterogens,
                    add_hydrogens=config.protein.add_hydrogens,
                    fix_missing_residues=config.protein.fix_missing_residues,
                    fix_missing_atoms=config.protein.fix_missing_atoms,
                    ph=config.protein.ph,
                    keep_ligands=config.protein.keep_ligands,
                )
                ctx["clean_pdb"] = result.get("clean_pdb")
                ctx["protein_pdbqt"] = result.get("pdbqt")

            elif stage == "ligand_preparation":
                result = prepare_ligands(
                    input_path=ligand_library,
                    output_dir=output_dir,
                    add_hydrogens=config.ligand.add_hydrogens,
                    generate_3d=config.ligand.generate_3d,
                    num_conformers=config.ligand.num_conformers,
                    energy_minimize=config.ligand.energy_minimize,
                    ph=config.ligand.ph,
                    max_tautomers=config.ligand.max_tautomers,
                    convert_to_pdbqt=config.ligand.convert_to_pdbqt,
                )
                ctx["prepared_ligands_sdf"] = result.get("combined_sdf")

            elif stage == "pocket_detection":
                pdb = ctx["clean_pdb"]
                if not pdb:
                    pdb = os.path.join(output_dir, "protein_preparation", "protein_clean.pdb")
                    if not os.path.exists(pdb):
                        pdb = protein_pdb
                pocket_kwargs = {
                    "protein_pdb": pdb,
                    "output_dir": output_dir,
                    "method": config.pocket.method,
                    "selection_mode": config.pocket.selection_mode,
                    "min_pocket_size": config.pocket.min_pocket_size,
                    "max_pockets": config.pocket.max_pockets,
                    "padding": config.pocket.padding,
                    "max_box_dimension": config.pocket.max_box_dimension,
                    "warn_large_boxes": config.pocket.warn_large_boxes,
                }
                if config.pocket.reference_ligand:
                    pocket_kwargs["reference_ligand"] = config.pocket.reference_ligand
                if config.pocket.manual_center:
                    pocket_kwargs["manual_center"] = tuple(config.pocket.manual_center)
                if config.pocket.manual_size:
                    pocket_kwargs["manual_size"] = tuple(config.pocket.manual_size)

                result = detect_pockets(**pocket_kwargs)
                ctx["docking_box"] = result.get("docking_box")

            elif stage == "druglikeness":
                input_sdf = ctx["prepared_ligands_sdf"]
                if not input_sdf:
                    input_sdf = os.path.join(
                        output_dir, "ligand_preparation", "prepared_ligands.sdf"
                    )
                if not os.path.exists(input_sdf):
                    input_sdf = _get_latest_ligands_sdf(ctx, output_dir)
                result = screen_druglikeness(
                    input_sdf=input_sdf,
                    output_dir=output_dir,
                    profile=config.druglikeness.profile,
                    rules=config.druglikeness.rules,
                    evaluation_mode=config.druglikeness.evaluation_mode,
                    minimum_rule_pass_fraction=config.druglikeness.minimum_rule_pass_fraction,
                    brenk_mode=config.druglikeness.brenk_mode,
                    custom_filters=config.druglikeness.custom_filters,
                    max_violations=config.druglikeness.max_violations,
                )
                ctx["passed_dl_sdf"] = result.get("passed_sdf")
                ctx["dl_csv"] = result.get("descriptors_csv")

            elif stage == "admet":
                input_sdf = ctx["passed_dl_sdf"]
                if not input_sdf:
                    input_sdf = os.path.join(
                        output_dir, "druglikeness", "passed_ligands.sdf"
                    )
                if not os.path.exists(input_sdf):
                    input_sdf = _get_latest_ligands_sdf(ctx, output_dir)
                result = screen_admet(
                    input_sdf=input_sdf,
                    output_dir=output_dir,
                    profile=config.admet.profile,
                    target_type=config.admet.target_type,
                    filter_toxic=config.admet.filter_toxic,
                    filter_low_absorption=config.admet.filter_low_absorption,
                    thresholds=config.admet.thresholds,
                )
                ctx["passed_admet_sdf"] = result.get("passed_sdf")
                ctx["admet_csv"] = result.get("admet_csv")

            elif stage == "docking":
                protein_pdbqt = ctx["protein_pdbqt"]
                if not protein_pdbqt:
                    protein_pdbqt = os.path.join(
                        output_dir, "protein_preparation", "protein.pdbqt"
                    )
                ligands_sdf = ctx["passed_admet_sdf"]
                if not ligands_sdf:
                    ligands_sdf = os.path.join(
                        output_dir, "admet", "passed_ligands.sdf"
                    )
                if not os.path.exists(ligands_sdf):
                    ligands_sdf = _get_latest_ligands_sdf(ctx, output_dir)
                    
                docking_box = ctx.get("docking_box")
                if not docking_box:
                    docking_box = os.path.join(
                        output_dir, "pocket_detection", "docking_box.json"
                    )
                if isinstance(docking_box, str) and not os.path.exists(docking_box):
                    print("\n[!] Pocket detection was skipped or docking_box.json is missing.")
                    from core.workflow_engine import WorkflowEngine
                    engine = WorkflowEngine()
                    engine._collect_docking_pocket(ctx)
                    docking_box = ctx.get("docking_box")

                result = run_docking(
                    protein_pdbqt=protein_pdbqt,
                    ligands_path=ligands_sdf,
                    docking_box=docking_box,
                    output_dir=output_dir,
                    exhaustiveness=config.docking.exhaustiveness,
                    n_poses=config.docking.n_poses,
                    energy_range=config.docking.energy_range,
                    seed=config.docking.seed,
                    n_cpus=config.docking.n_cpus,
                    score_only=config.docking.score_only,
                )
                ctx["docking_csv"] = result.get("results_csv")

            elif stage == "lead_selection":
                docking_csv = ctx["docking_csv"]
                if not docking_csv:
                    docking_csv = os.path.join(
                        output_dir, "docking", "docking_results.csv"
                    )
                admet_csv = ctx.get("admet_csv")
                if not admet_csv:
                    admet_csv = os.path.join(
                        output_dir, "admet", "admet_results.csv"
                    )
                    if not os.path.exists(admet_csv):
                        admet_csv = None
                dl_csv = ctx.get("dl_csv")
                if not dl_csv:
                    dl_csv = os.path.join(
                        output_dir, "druglikeness", "descriptors.csv"
                    )
                    if not os.path.exists(dl_csv):
                        dl_csv = None
                passed_admet_sdf = ctx.get("passed_admet_sdf")
                if not passed_admet_sdf:
                    passed_admet_sdf = os.path.join(
                        output_dir, "admet", "passed_ligands.sdf"
                    )
                    if not os.path.exists(passed_admet_sdf):
                        passed_admet_sdf = _get_latest_ligands_sdf(ctx, output_dir)

                result = select_leads(
                    docking_results=docking_csv,
                    output_dir=output_dir,
                    admet_results=admet_csv,
                    druglikeness_results=dl_csv,
                    ligands_sdf=passed_admet_sdf,
                    top_n=config.lead_selection.top_n,
                    scoring_method=config.lead_selection.scoring_method,
                    weights=config.lead_selection.weights,
                    affinity_cutoff=config.lead_selection.affinity_cutoff,
                    on_no_hits=getattr(config.lead_selection, "on_no_hits", "fallback"),
                    diversity_filter=config.lead_selection.diversity_filter,
                    similarity_cutoff=config.lead_selection.similarity_cutoff,
                )

            results[stage] = result

        except Exception as e:
            logger.error(f"Stage '{stage}' failed: {e}")
            results[stage] = {"status": "failed", "error": str(e)}
            raise

    duration = time.time() - start_time
    log_step(logger, "Drug Selection Workflow", "completed", {
        "stages": len(active_stages),
        "duration": f"{duration:.1f}s",
    })

    # Print summary table
    summary = {}
    for stage, result in results.items():
        summary[stage] = {
            "status": result.get("status", "unknown"),
            "duration": result.get("duration", "—"),
        }
    log_summary(logger, summary)

    # Write scientific validation report
    validation_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_protein": protein_pdb,
        "input_library": ligand_library,
        "duration_seconds": round(duration, 2),
        "configuration": {
            "protein": {
                "remove_water": config.protein.remove_water,
                "ph": config.protein.ph,
            },
            "ligand": {
                "add_hydrogens": config.ligand.add_hydrogens,
                "generate_3d": config.ligand.generate_3d,
            },
            "pocket": {
                "method": config.pocket.method,
                "selection_mode": config.pocket.selection_mode,
                "padding": config.pocket.padding,
                "max_box_dimension": config.pocket.max_box_dimension,
            },
            "druglikeness": {
                "profile": config.druglikeness.profile,
                "evaluation_mode": config.druglikeness.evaluation_mode,
                "brenk_mode": config.druglikeness.brenk_mode,
                "rules": config.druglikeness.rules,
            },
            "admet": {
                "profile": config.admet.profile,
                "target_type": config.admet.target_type,
            },
            "docking": {
                "exhaustiveness": config.docking.exhaustiveness,
                "seed": config.docking.seed,
            }
        },
        "results": {
            stage: {
                "status": result.get("status", "unknown"),
                "total_input": result.get("total_input") if isinstance(result, dict) else None,
                "total_passed": result.get("total_passed") if isinstance(result, dict) else None,
            } for stage, result in results.items()
        }
    }
    from utils.io import write_json
    reports_dir = os.path.join(output_dir, "reports")
    ensure_dir(reports_dir)
    write_json(validation_report, os.path.join(reports_dir, "scientific_validation.json"))

    # Generate standalone interactive HTML report
    try:
        from reports.generator import generate_report
        proj_name = getattr(config.project, "name", "Run") if hasattr(config, "project") else "Run"
        generate_report(output_dir=output_dir, report_title=f"DrugLab Screening Report — {proj_name}")
    except Exception as report_err:
        logger.warning(f"HTML report generation failed: {report_err}")

    return results


def _resolve_stages(
    all_stages: list[str],
    stop_after: str | None,
    resume_from: str | None,
) -> list[str]:
    """Filter stages based on stop_after and resume_from."""
    start_idx = 0
    end_idx = len(all_stages)

    if resume_from:
        for i, s in enumerate(all_stages):
            if s == resume_from:
                start_idx = i
                break
        else:
            logger.warning(f"Unknown resume stage: {resume_from}")

    if stop_after:
        for i, s in enumerate(all_stages):
            if s == stop_after:
                end_idx = i + 1
                break
        else:
            logger.warning(f"Unknown stop stage: {stop_after}")

    return all_stages[start_idx:end_idx]


def _get_latest_ligands_sdf(ctx: dict, output_dir: str) -> str | None:
    paths = [
        ctx.get("passed_admet_sdf"),
        os.path.join(output_dir, "admet", "passed_ligands.sdf"),
        ctx.get("passed_dl_sdf"),
        os.path.join(output_dir, "druglikeness", "passed_ligands.sdf"),
        ctx.get("prepared_ligands_sdf"),
        os.path.join(output_dir, "ligand_preparation", "prepared_ligands.sdf"),
    ]
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None
