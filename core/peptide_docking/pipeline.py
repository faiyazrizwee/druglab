"""
End-to-End Orchestrator for DrugLab 2.0 Peptide Docking Pipeline.
"""

from __future__ import annotations

import os
import sys
import json
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Any
import yaml

from core.peptide_docking.config import PeptideDockingConfig, load_peptide_config
from core.peptide_docking.input_validation import (
    PeptideCandidate,
    validate_protein_pdb,
    validate_peptide_sequence,
    parse_fasta_file,
    parse_fasta_content,
    validate_peptide_structure,
    compute_file_checksum,
)
from core.peptide_docking.protein_preparation import (
    inspect_protein_structure,
    prepare_receptor_for_lightdock,
)
from core.peptide_docking.peptide_filtering import filter_peptides
from core.peptide_docking.peptide_admet import predict_peptide_admet, display_peptide_admet_summary
from core.peptide_docking.peptide_preparation import prepare_peptide_candidate
from core.peptide_docking.binding_site import setup_binding_site_and_restraints
from core.peptide_docking.pocket_detection import interactive_peptide_pocket_detection
from core.peptide_docking.lightdock_runner import run_lightdock_docking, get_lightdock_version
from core.peptide_docking.pose_analysis import analyze_docked_pose
from core.peptide_docking.ranking import rank_peptide_candidates
from core.peptide_docking.lead_selection import rank_and_select_peptide_leads
from core.peptide_docking.reporting import generate_peptide_html_report
from utils.io import ensure_dir
from utils.logger import setup_logger

logger = setup_logger("peptide_pipeline", level="INFO")


def run_peptide_docking_pipeline(
    protein_input: str | None = None,
    peptide_fasta: str | None = None,
    peptide_sequence: str | None = None,
    peptide_structure: str | None = None,
    output_dir: str = "results",
    config: PeptideDockingConfig | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
    """
    Execute the full DrugLab 2.0 Protein–Peptide Docking Workflow.
    """
    start_time = time.time()
    start_iso = datetime.now().isoformat()

    # 1. Load and merge config
    overrides: dict[str, Any] = {}
    if protein_input:
        overrides.setdefault("protein", {})["input"] = protein_input
    if peptide_fasta:
        overrides.setdefault("peptide", {})["fasta"] = peptide_fasta
    if peptide_sequence:
        overrides.setdefault("peptide", {})["sequence"] = peptide_sequence
    if peptide_structure:
        overrides.setdefault("peptide", {})["structure"] = peptide_structure
    if output_dir:
        overrides["output"] = {"directory": output_dir}

    if config is None:
        cfg = load_peptide_config(config_path=config_path, overrides=overrides)
    else:
        cfg = config
        if protein_input:
            cfg.protein.input = protein_input
        if peptide_fasta:
            cfg.peptide.fasta = peptide_fasta
        if peptide_sequence:
            cfg.peptide.sequence = peptide_sequence
        if peptide_structure:
            cfg.peptide.structure = peptide_structure
        if output_dir:
            cfg.output_dir = output_dir

    # 2. Setup run directory
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"peptide_run_{timestamp_str}"
    run_dir = Path(ensure_dir(Path(cfg.output_dir) / run_id))

    logger.info(f"Starting DrugLab 2.0 Peptide Docking Pipeline [Run ID: {run_id}]")
    logger.info(f"Results will be stored in: {run_dir.resolve()}")

    # Save config snapshot
    config_snapshot_path = run_dir / "config_snapshot.yaml"
    with open(config_snapshot_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, default_flow_style=False)

    # 3. Protein target resolution
    prot_target = cfg.protein.input
    if not prot_target:
        raise ValueError("No protein receptor target specified in arguments or configuration.")

    # Check if target is a 4-letter PDB ID
    if not os.path.isfile(prot_target) and len(prot_target.strip()) == 4 and prot_target.strip().isalnum():
        logger.info(f"Target '{prot_target}' appears to be a PDB ID. Fetching from RCSB PDB...")
        from utils.fetch import fetch_pdb
        prot_target = fetch_pdb(prot_target.strip().upper(), str(run_dir / f"{prot_target.strip().upper()}.pdb"))

    prot_path = Path(prot_target).resolve()
    if not prot_path.is_file():
        raise FileNotFoundError(f"Receptor PDB structure not found: {prot_target}")

    # 4. Protein Preparation
    prep_prot_dir = run_dir / "prepared_proteins"
    inspect_res = inspect_protein_structure(prot_path)
    logger.info(
        f"Receptor inspection: chains={inspect_res.get('chains')}, residues={inspect_res.get('num_residues')}, "
        f"waters={inspect_res.get('num_waters')}, metals={inspect_res.get('metals')}, cofactors={inspect_res.get('cofactors')}"
    )

    prep_prot_result = prepare_receptor_for_lightdock(
        input_pdb=prot_path,
        output_dir=prep_prot_dir,
        chains=cfg.protein.chains,
        water_policy=getattr(cfg.protein, "water_policy", "remove_bulk"),
        keep_metals=getattr(cfg.protein, "keep_metals", True),
        keep_cofactors=getattr(cfg.protein, "keep_cofactors", True),
        keep_heteroatoms=getattr(cfg.protein, "keep_heteroatoms", True),
        remove_hydrogens=cfg.protein.remove_hydrogens,
        biological_assembly=getattr(cfg.protein, "biological_assembly", None),
    )
    prepared_receptor_pdb = prep_prot_result["prepared_path"]

    # 5. Peptide Candidate Ingestion (Single or Multi-Peptide Library)
    candidates: list[PeptideCandidate] = []
    if cfg.peptide.structure and os.path.isfile(cfg.peptide.structure):
        cand = validate_peptide_structure(cfg.peptide.structure)
        candidates.append(cand)
    elif cfg.peptide.fasta and (os.path.isfile(cfg.peptide.fasta) or os.path.isdir(cfg.peptide.fasta)):
        candidates = parse_fasta_file(
            cfg.peptide.fasta,
            allow_nonstandard=cfg.peptide.allow_nonstandard_residues,
        )
    elif cfg.peptide.sequence:
        raw_seq = cfg.peptide.sequence.strip()
        if raw_seq.startswith(">"):
            candidates = parse_fasta_content(
                raw_seq,
                allow_nonstandard=cfg.peptide.allow_nonstandard_residues,
                source_name="raw_fasta",
            )
        elif "," in raw_seq or ";" in raw_seq or "\n" in raw_seq:
            tokens = [t.strip() for t in re.split(r"[,;\n\r]+", raw_seq) if t.strip()]
            for idx, tok in enumerate(tokens, 1):
                try:
                    v_seq = validate_peptide_sequence(tok, allow_nonstandard=cfg.peptide.allow_nonstandard_residues)
                    cid = f"peptide_{idx}"
                    candidates.append(PeptideCandidate(id=cid, sequence=v_seq, source="sequence"))
                except Exception as e:
                    logger.warning(f"Skipping invalid sequence item '{tok}': {e}")
        else:
            seq = validate_peptide_sequence(
                raw_seq,
                allow_nonstandard=cfg.peptide.allow_nonstandard_residues,
            )
            cand = PeptideCandidate(
                id="peptide_1",
                sequence=seq,
                source="sequence",
            )
            candidates.append(cand)
    else:
        raise ValueError("No peptide input provided (must supply fasta, sequence, or structure).")

    if not candidates:
        raise ValueError("No valid peptide candidates found to process.")

    logger.info(f"Loaded {len(candidates)} peptide candidate(s) for screening.")
    try:
        from rich.console import Console
        Console().print(f"[bold green]✓ Ingested {len(candidates)} peptide candidate(s) into screening pipeline.[/bold green]")
    except Exception:
        pass

    # 6. Peptide Filtering & Diagnostics
    passed_cands, filtered_cands, filtering_df = filter_peptides(
        candidates=candidates,
        config=cfg.filtering,
        output_dir=run_dir,
    )
    for cand in passed_cands:
        warn = cand.properties.get("length_warning")
        if warn:
            try:
                from rich.console import Console
                Console().print(
                    f"[bold yellow]⚠ Warning:[/bold yellow] [yellow]Peptide '{cand.id}' ({cand.sequence}): "
                    f"{warn}. Length filter is bypassed — candidate proceeds to docking.[/yellow]"
                )
            except Exception:
                logger.warning(f"Peptide '{cand.id}' ({cand.sequence}): {warn}. Length filter bypassed.")

    # 7. Peptide ADMET Checking
    logger.info(f"Running peptide ADMET assessment using provider: {cfg.admet.provider.upper()}...")
    admet_df = predict_peptide_admet(
        candidates=candidates,
        config=cfg.admet,
        output_dir=run_dir,
    )
    if not admet_df.empty:
        try:
            display_peptide_admet_summary(admet_df, provider_name=cfg.admet.provider)
        except Exception as e:
            logger.warning(f"Failed to display ADMET summary table: {e}")

    # 8. Peptide Structural Preparation
    prep_pep_dir = run_dir / "prepared_peptides"
    prepared_candidates: list[PeptideCandidate] = []
    for cand in passed_cands:
        try:
            cand = prepare_peptide_candidate(
                candidate=cand,
                output_dir=prep_pep_dir,
                chain_id="B",
                n_term_cap=getattr(cfg.peptide, "n_term_cap", "zwitterion"),
                c_term_cap=getattr(cfg.peptide, "c_term_cap", "zwitterion"),
                is_cyclic=getattr(cfg.peptide, "is_cyclic", False),
                disulfide_pairs=getattr(cfg.peptide, "disulfide_pairs", None),
                num_conformers=cfg.peptide.conformers,
                minimize=cfg.peptide.minimize,
                forcefield=cfg.peptide.forcefield,
                seed=cfg.peptide.seed,
            )
            prepared_candidates.append(cand)
        except Exception as e:
            logger.error(f"Failed to prepare peptide {cand.id}: {e}")
            cand.status = "failed"
            cand.failure_reason = f"Preparation error: {e}"

    if not prepared_candidates:
        logger.warning("No peptide candidates successfully passed filtering and preparation.")

    # 9. Binding Site & Restraints Setup (per run or primary receptor)
    ref_pep_pdb = prepared_candidates[0].prepared_structure_path if prepared_candidates else None
    bs_result = None
    existing_restraints = run_dir / "restraints.list"
    if existing_restraints.is_file():
        logger.info(f"Using pre-existing LightDock restraints file: {existing_restraints}")
        from core.peptide_docking.binding_site import BindingSiteResult
        with open(existing_restraints, "r", encoding="utf-8") as rf:
            restr_lines = [l.strip() for l in rf if l.strip()]
        bs_result = BindingSiteResult(
            mode="custom_restraints",
            restraints_path=str(existing_restraints),
            receptor_restraints=restr_lines,
        )
    elif cfg.binding_site.mode in ("pocket_gui", "gui"):
        logger.info("Launching 3D Graphical Box Tuner GUI for binding pocket...")
        pocket_res = interactive_peptide_pocket_detection(
            receptor_pdb=prepared_receptor_pdb,
            output_dir=run_dir,
            launch_gui=True,
            auto_confirm=False,
        )
        if pocket_res.get("restraints_path"):
            from core.peptide_docking.binding_site import BindingSiteResult
            restr_list = pocket_res.get("restraints") or pocket_res.get("restraint_lines") or []
            bs_result = BindingSiteResult(
                mode="pocket_gui",
                restraints_path=pocket_res["restraints_path"],
                receptor_restraints=restr_list,
            )
    elif ref_pep_pdb:
        bs_result = setup_binding_site_and_restraints(
            receptor_pdb=prepared_receptor_pdb,
            peptide_pdb=ref_pep_pdb,
            config=cfg.binding_site,
            output_dir=run_dir,
        )

    # 10. Docking Execution
    docking_output_dir = run_dir / "docking"
    completed_count = 0
    lightdock_ver = get_lightdock_version()

    for idx, cand in enumerate(prepared_candidates, 1):
        logger.info(f"[{idx}/{len(prepared_candidates)}] Docking peptide candidate '{cand.id}' ({cand.sequence})...")
        try:
            from rich.console import Console
            Console().print(f"[bold cyan]▶ [{idx}/{len(prepared_candidates)}] Docking peptide '{cand.id}' ({cand.length} AA)...[/bold cyan]")
        except Exception:
            pass
        cand.status = "docking"

        rst_file = bs_result.restraints_path if bs_result else None
        dock_res = run_lightdock_docking(
            candidate_id=cand.id,
            receptor_pdb=prepared_receptor_pdb,
            peptide_pdb=cand.prepared_structure_path,
            output_dir=docking_output_dir,
            config=cfg.lightdock,
            restraints_path=rst_file,
            num_poses_to_generate=5,
            cpu_cores=cfg.resources.max_cpu_processes,
        )

        cand.status = dock_res.status
        if dock_res.status == "completed":
            completed_count += 1
            # Analyze each generated pose
            analyzed_poses = []
            for p_info in dock_res.poses:
                pose_pdb = p_info.get("pdb_path")
                if pose_pdb and os.path.isfile(pose_pdb):
                    try:
                        metrics = analyze_docked_pose(
                            pose_pdb_path=pose_pdb,
                            receptor_chain="A",
                            peptide_chain="B",
                            restraints=bs_result.receptor_restraints if bs_result else None,
                        )
                        p_info.update(metrics)
                    except Exception as e:
                        logger.warning(f"Pose analysis failed on {pose_pdb}: {e}")
                analyzed_poses.append(p_info)
            cand.docking_results = analyzed_poses
        else:
            cand.failure_reason = dock_res.error_message
            logger.error(f"Docking failed for candidate {cand.id}: {dock_res.error_message}")

    # 11. Results Ranking & Scoring
    cand_df, poses_df = rank_peptide_candidates(
        candidates=candidates,
        output_dir=run_dir,
    )

    # 12. Multi-Objective Lead Selection
    leads_df = rank_and_select_peptide_leads(
        candidates=candidates,
        candidates_df=cand_df,
        poses_df=poses_df,
        admet_df=admet_df,
        output_dir=run_dir,
    )

    elapsed_total = round(time.time() - start_time, 2)
    end_iso = datetime.now().isoformat()

    # 13. Build Run Manifest
    manifest = {
        "run_id": run_id,
        "workflow_mode": "peptide",
        "title": cfg.report.title,
        "start_time": start_iso,
        "end_time": end_iso,
        "runtime_sec": elapsed_total,
        "original_receptor_path": str(prot_path),
        "prepared_receptor_path": str(prepared_receptor_pdb),
        "receptor_checksum": prep_prot_result.get("prepared_checksum"),
        "receptor_chains": prep_prot_result.get("chains"),
        "terminal_chemistry": {
            "n_term_cap": getattr(cfg.peptide, "n_term_cap", "zwitterion"),
            "c_term_cap": getattr(cfg.peptide, "c_term_cap", "zwitterion"),
            "is_cyclic": getattr(cfg.peptide, "is_cyclic", False),
        },
        "receptor_inspection": {
            "chains": inspect_res.get("chains", []),
            "missing_residues": len(inspect_res.get("missing_residues", [])),
            "alternate_conformations": len(inspect_res.get("alternate_conformations", [])),
            "metals": inspect_res.get("metals", []),
            "cofactors": inspect_res.get("cofactors", []),
            "water_policy": getattr(cfg.protein, "water_policy", "remove_bulk"),
        },
        "total_peptides": len(candidates),
        "passed_filtering": len(passed_cands),
        "filtered_peptides": len(filtered_cands),
        "length_warnings_count": sum(1 for c in passed_cands if "length_warning" in c.properties),
        "admet_provider": cfg.admet.provider,
        "admet_enabled": cfg.admet.enabled,
        "completed_docking": completed_count,
        "leads_selected_count": len(leads_df) if leads_df is not None else 0,
        "lightdock_version": lightdock_ver,
        "scoring_function": cfg.lightdock.scoring_function,
        "preset": cfg.lightdock.preset,
        "seed": cfg.lightdock.seed,
        "binding_site_mode": bs_result.mode if bs_result else cfg.binding_site.mode,
        "restraints_count": len(bs_result.receptor_restraints) if bs_result else 0,
        "output_directory": str(run_dir.resolve()),
        "status": "success" if completed_count > 0 or not candidates else "completed_with_warnings",
    }

    manifest_path = run_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 14. Interactive HTML Report
    report_html_path = None
    if cfg.report.html:
        report_html_path = generate_peptide_html_report(
            output_dir=run_dir,
            manifest=manifest,
            candidates_df=cand_df,
            poses_df=poses_df,
            filtering_df=filtering_df,
            admet_df=admet_df,
            leads_df=leads_df,
        )

    logger.info(f"Peptide Docking Pipeline finished in {elapsed_total:.2f}s.")
    if report_html_path:
        logger.info(f"Interactive Report: file://{report_html_path}")

    return {
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "manifest": manifest,
        "report_html": report_html_path,
        "candidates_df": cand_df,
        "poses_df": poses_df,
        "leads_df": leads_df,
    }
