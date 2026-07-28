"""
DrugLab Lead Optimization Workflow Orchestrator.

Runs the full lead-optimization pipeline:
    Analogue Generation → Drug-Likeness Screening → ADMET Screening →
    Docking → Candidate Ranking
"""

import os
import time
import logging

from utils.logger import setup_logger, log_step, log_summary
from utils.config import load_config, save_config_snapshot, DrugLabConfig
from utils.io import ensure_dir, validate_file_exists

from lead_optimization.analogue_generation import generate_analogues
from lead_optimization.screening import screen_analogues
from core.docking import run_docking
from lead_optimization.ranking import rank_candidates

logger = setup_logger("lead_optimization_workflow", level="INFO")


def run_lead_optimization(
    lead_file: str,
    protein_file: str,
    docking_box: dict | str = None,
    output_dir: str = "results",
    config: DrugLabConfig | None = None,
) -> dict:
    """
    Run the lead optimization pipeline.

    Args:
        lead_file: Path to the lead molecule file (SDF, MOL, PDB, etc).
        protein_file: Path to the protein file (PDB or PDBQT).
        docking_box: Docking box dict or path to docking_box.json. Optional if PDB is provided.
        output_dir: Path to the output directory.
        config: DrugLabConfig instance.

    Returns:
        Dict of stage → result metrics.
    """
    start_time = time.time()
    log_step(logger, "Lead Optimization Workflow", "started", {
        "lead": lead_file,
        "protein": protein_file,
        "output_dir": output_dir,
    })

    validate_file_exists(lead_file, "Lead molecule file")
    validate_file_exists(protein_file, "Protein file")

    if config is None:
        config = load_config()

    ensure_dir(output_dir)
    save_config_snapshot(config, output_dir)

    results = {}

    # Stage 0: Standalone Preparation
    try:
        lead_sdf = lead_file
        if not lead_file.lower().endswith(".sdf"):
            from core.ligand_preparation import prepare_ligands
            logger.info("Preparing lead molecule...")
            prep_res = prepare_ligands(
                input_path=lead_file,
                output_dir=os.path.join(output_dir, "lead_preparation"),
            )
            lead_sdf = prep_res["combined_sdf"]

        protein_pdbqt = protein_file
        clean_pdb = protein_file
        if protein_file.lower().endswith(".pdb"):
            from core.protein_preparation import prepare_protein
            logger.info("Preparing protein...")
            prep_res = prepare_protein(
                input_pdb=protein_file,
                output_dir=os.path.join(output_dir, "protein_preparation"),
            )
            protein_pdbqt = prep_res["pdbqt"]
            clean_pdb = prep_res["clean_pdb"]

        if not docking_box:
            from core.pocket_detection import detect_pockets, interactive_pocket_selection
            import gemmi
            logger.info("Detecting pockets...")
            pocket_res = detect_pockets(
                protein_pdb=clean_pdb,
                output_dir=os.path.join(output_dir, "pocket_detection"),
            )
            
            pockets = pocket_res.get("pockets", [])
            doc = gemmi.read_structure(clean_pdb)
            max_dim = config.pocket.max_box_dimension if config else 30.0
            
            selected, method = interactive_pocket_selection(pockets, max_dim, doc)
            docking_box = {
                "center": selected["center"],
                "size": selected["size"]
            }
    except Exception as e:
        logger.error(f"Standalone preparation failed: {e}")
        raise

    # Stage 1: Analogue Generation
    try:
        logger.info("Stage 1/5: Analogue Generation")
        gen_result = generate_analogues(
            lead_sdf=lead_sdf,
            output_dir=output_dir,
            strategies=config.optimization.strategies,
            max_analogues=config.optimization.max_analogues,
            similarity_range=tuple(config.optimization.similarity_range),
            mw_range=tuple(config.optimization.mw_range),
            custom_fragments=config.optimization.custom_fragments,
        )
        results["analogue_generation"] = gen_result
        analogues_sdf = gen_result.get("analogues_sdf")
    except Exception as e:
        logger.error(f"Analogue generation failed: {e}")
        results["analogue_generation"] = {"status": "failed", "error": str(e)}
        raise

    # Stage 2: Combined Screening (Drug-Likeness + ADMET)
    try:
        logger.info("Stage 2/5: Drug-Likeness + ADMET Screening")
        screen_result = screen_analogues(
            input_sdf=analogues_sdf,
            output_dir=output_dir,
            profile=config.druglikeness.profile,
            evaluation_mode=config.druglikeness.evaluation_mode,
            minimum_rule_pass_fraction=config.druglikeness.minimum_rule_pass_fraction,
            brenk_mode=config.druglikeness.brenk_mode,
            druglikeness_rules=config.druglikeness.rules,
            max_violations=config.druglikeness.max_violations,
            strict=config.druglikeness.strict,
            admet_profile=config.admet.profile,
            admet_target_type=config.admet.target_type,
            filter_toxic=config.admet.filter_toxic,
            filter_low_absorption=config.admet.filter_low_absorption,
            admet_thresholds=config.admet.thresholds,
        )
        results["screening"] = screen_result
        passed_sdf = screen_result.get("final_passed_sdf")
    except Exception as e:
        logger.error(f"Screening failed: {e}")
        results["screening"] = {"status": "failed", "error": str(e)}
        raise

    # Stage 3: Docking
    try:
        logger.info("Stage 3/5: Docking Analogues")
        if passed_sdf and os.path.isfile(passed_sdf):
            print("\n" + "="*60)
            print("Analogue Docking Configuration")
            print("="*30 + "\n")
            print(f"Current exhaustiveness: {config.docking.exhaustiveness}")
            print("Enter new exhaustiveness (or press Enter to keep current): ", end="", flush=True)
            try:
                ans = input().strip()
                if ans:
                    config.docking.exhaustiveness = int(ans)
                    print(f"Exhaustiveness set to {config.docking.exhaustiveness}\n")
            except (EOFError, KeyboardInterrupt):
                print()
            except ValueError:
                print("Invalid input. Keeping current exhaustiveness.\n")

            dock_result = run_docking(
                protein_pdbqt=protein_pdbqt,
                ligands_path=passed_sdf,
                docking_box=docking_box,
                output_dir=os.path.join(output_dir, "analogue_docking"),
                exhaustiveness=config.docking.exhaustiveness,
                n_poses=config.docking.n_poses,
                energy_range=config.docking.energy_range,
                seed=config.docking.seed,
                n_cpus=config.docking.n_cpus,
            )
            results["docking"] = dock_result
        else:
            logger.warning("No molecules passed screening; skipping docking.")
            results["docking"] = {"status": "skipped", "reason": "no passed molecules"}
    except Exception as e:
        logger.error(f"Docking failed: {e}")
        results["docking"] = {"status": "failed", "error": str(e)}
        raise

    # Stage 4: Ranking
    try:
        logger.info("Stage 4/5: Ranking Candidates")
        dock_csv = results.get("docking", {}).get("results_csv")
        admet_csv = screen_result.get("admet", {}).get("admet_csv") if isinstance(screen_result, dict) else None
        dl_csv = screen_result.get("druglikeness", {}).get("descriptors_csv") if isinstance(screen_result, dict) else None

        if dock_csv and os.path.isfile(dock_csv):
            rank_result = rank_candidates(
                docking_results=dock_csv,
                output_dir=output_dir,
                lead_sdf=lead_sdf,
                admet_results=admet_csv,
                druglikeness_results=dl_csv,
                analogues_sdf=passed_sdf,
                top_n=config.optimization.top_n,
            )
            results["ranking"] = rank_result
        else:
            logger.warning("No docking results available; skipping ranking.")
            results["ranking"] = {"status": "skipped", "reason": "no docking results"}
    except Exception as e:
        logger.error(f"Ranking failed: {e}")
        results["ranking"] = {"status": "failed", "error": str(e)}
        raise

    duration = time.time() - start_time
    log_step(logger, "Lead Optimization Workflow", "completed", {
        "stages": len(results),
        "duration": f"{duration:.1f}s",
    })

    # Print summary
    try:
        from utils.chemistry import get_mol_name
        from utils.io import read_sdf
        
        # Original Lead Name
        orig_lead_name = "Unknown"
        if lead_sdf and os.path.exists(lead_sdf):
            mols = read_sdf(lead_sdf)
            if mols and mols[0]:
                orig_lead_name = get_mol_name(mols[0])

        gen_stats = results.get("analogue_generation", {})
        analogues_gen = gen_stats.get("total_generated", 0)
        
        screen_stats = results.get("screening", {})
        dl_stats = screen_stats.get("druglikeness", {})
        dl_passed = dl_stats.get("passed", 0)
        
        admet_stats = screen_stats.get("admet", {})
        admet_passed = admet_stats.get("passed", 0)
        
        dock_stats = results.get("docking", {})
        docked = dock_stats.get("completed", 0)
        
        rank_stats = results.get("ranking", {})
        top_candidates = rank_stats.get("top_candidates", [])
        
        best_name = "None"
        best_docking = "N/A"
        improvement = "N/A"
        
        if top_candidates:
            best_lead = top_candidates[0]
            best_name = best_lead.get("name", "Unknown")
            best_docking = best_lead.get("best_affinity", "N/A")
            if best_docking != "N/A":
                best_docking = f"{best_docking} kcal/mol"
                
            # Compute improvement
            if orig_lead_name != "Unknown" and best_docking != "N/A":
                lead_affinity = None
                dock_csv = os.path.join(output_dir, "docking", "docking_results.csv")
                if os.path.exists(dock_csv):
                    import pandas as pd
                    try:
                        dock_df = pd.read_csv(dock_csv)
                        match = dock_df[dock_df['name'] == orig_lead_name]
                        if not match.empty:
                            lead_affinity = match.iloc[0]['best_affinity']
                    except Exception:
                        pass
                
                if lead_affinity is not None:
                    # e.g., -5.6 -> -8.4 is an improvement of 2.8
                    try:
                        val = float(best_lead.get("best_affinity", 0))
                        imp = round(lead_affinity - val, 2)
                        sign = "+" if imp > 0 else ""
                        improvement = f"{sign}{imp} kcal/mol"
                    except Exception:
                        pass
            
        print("\n" + "="*60)
        print("Lead Optimization Complete")
        print("==========================")
        print(f"\nOriginal Lead:\n{orig_lead_name}")
        print(f"\nAnalogues Generated:\n{analogues_gen}")
        print(f"\nDrug-Like:\n{dl_passed}")
        print(f"\nADMET Passed:\n{admet_passed}")
        print(f"\nDocked:\n{docked}")
        
        if top_candidates:
            print(f"\nBest Optimized Lead:\n{best_name}")
            print(f"\nDocking Score:\n{best_docking}")
            
            if improvement != "N/A":
                print(f"\nImprovement:\n{improvement}")
            
            best_sdf = rank_stats.get("best_optimized_lead_sdf", "")
            if best_sdf:
                print(f"\nOutput:\n{best_sdf}")
        print("\n" + "="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Failed to print summary: {e}")

    return results
