#!/usr/bin/env python3
"""
DrugLab — Computational Drug Discovery Platform.

Entry point supporting three execution modes:

    Mode 1 — Interactive Workflow (no args):
        $ druglab
        $ python druglab.py

    Mode 2 — CLI Module Commands:
        $ druglab protein prepare --input protein.pdb
        $ druglab ligand prepare --input ligands/
        $ druglab pocket detect --protein protein_clean.pdb
        $ druglab screen druglikeness --input prepared_ligands.sdf
        $ druglab screen admet --input passed_ligands.sdf
        $ druglab dock --protein protein.pdbqt --ligands ligands.sdf --box box.json
        $ druglab select --docking-results docking_results.csv

    Mode 3 — Full Automation:
        $ druglab run --protein protein.pdb --library ligands/
        $ druglab optimize --lead lead.sdf --protein protein.pdbqt --box box.json
"""

import sys
import os
import argparse

# Ensure the project root is on sys.path so relative imports work
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main():
    """Main entry point for DrugLab."""
    parser = argparse.ArgumentParser(
        prog="druglab",
        description="DrugLab — Computational Drug Discovery Platform",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug streaming logs on console")
    subparsers = parser.add_subparsers(dest="command")

    # ── Mode 3: Full Automation ──
    _add_run_subcommand(subparsers)
    _add_optimize_subcommand(subparsers)

    # ── Mode 2: CLI Module Commands ──
    _add_protein_subcommand(subparsers)
    _add_ligand_subcommand(subparsers)
    _add_pocket_subcommand(subparsers)
    _add_screen_subcommand(subparsers)
    _add_dock_subcommand(subparsers)
    _add_select_subcommand(subparsers)
    _add_report_subcommand(subparsers)
    _add_fetch_subcommand(subparsers)

    args = parser.parse_args()

    if getattr(args, "verbose", False):
        from utils.logger import set_verbose
        set_verbose(True)

    # Mode 1: No command → interactive menu
    if args.command is None:
        _run_interactive()
        return

    # Dispatch to handlers
    handler = getattr(args, "handler", None)
    if handler:
        handler(args)
    else:
        parser.print_help()


# ══════════════════════════════════════════════════════════════
# MODE 1: Interactive Workflow
# ══════════════════════════════════════════════════════════════

def _run_interactive():
    """Launch the interactive workflow builder."""
    from utils.config import load_config
    from core.workflow_engine import create_default_engine

    config = load_config()
    engine = create_default_engine()
    engine.run_interactive(config=config)


# ══════════════════════════════════════════════════════════════
# MODE 3: Full Automation — `druglab run`
# ══════════════════════════════════════════════════════════════

def _add_run_subcommand(subparsers):
    run_parser = subparsers.add_parser(
        "run",
        help="Run full drug selection pipeline",
    )
    run_parser.add_argument("--protein", default=None, help="Input protein PDB file or 4-letter PDB ID (e.g. 7DTD)")
    run_parser.add_argument("--library", default=None, help="Ligand library (SDF, directory, or compound name)")
    run_parser.add_argument("--output-dir", default="results", help="Output directory")
    run_parser.add_argument("--config", default=None, help="Path to YAML config file")
    run_parser.add_argument("--stop-after", default=None,
                            help="Stop after this stage (e.g., ligand_preparation)")
    run_parser.add_argument("--resume", default=None,
                            help="Resume from this stage")
    run_parser.set_defaults(handler=_handle_run)


def _handle_run(args):
    from utils.logger import print_banner
    from utils.config import load_config
    from drug_selection.workflow import run_drug_selection

    print_banner()

    protein = args.protein
    library = args.library

    # 1. Interactive prompt if protein is not provided
    if not protein:
        print("\nProvide protein target:")
        print("  [1] Enter local PDB file path")
        print("  [2] Download from RCSB PDB by ID (e.g. 7DTD, 4MZI, 6LU7)")
        try:
            choice = input("Choice [1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return

        if choice == "2":
            try:
                pdb_id = input("Enter 4-character RCSB PDB ID: ").strip().upper()
                from utils.fetch import fetch_pdb
                protein = fetch_pdb(pdb_id, os.path.join("protein", f"{pdb_id}.pdb"))
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return
        else:
            default_hint = " [default: protein/7DTD.pdb]" if os.path.isfile("protein/7DTD.pdb") else ""
            try:
                p = input(f"Protein PDB file path{default_hint}: ").strip()
                protein = p if p else "protein/7DTD.pdb"
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return

    # Auto-fetch if 4-character PDB ID passed directly
    if protein and not os.path.isfile(protein) and len(protein.strip()) == 4 and protein.strip().isalnum():
        print(f"Target '{protein}' appears to be a PDB ID. Fetching from RCSB PDB...")
        from utils.fetch import fetch_pdb
        try:
            protein = fetch_pdb(protein.strip().upper(), os.path.join("protein", f"{protein.strip().upper()}.pdb"))
        except Exception as e:
            print(f"Failed to fetch PDB {protein}: {e}")

    # 2. Interactive prompt if library is not provided
    if not library:
        print("\nProvide ligand library:")
        print("  [1] Enter local SDF file or directory")
        print("  [2] Download compound from PubChem by name (e.g. apigenin, quercetin, aspirin)")
        print("  [3] Download compound from PubChem by CID")
        try:
            choice = input("Choice [1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return

        if choice == "2":
            try:
                name = input("Enter compound name: ").strip()
                from utils.fetch import fetch_ligand_by_name
                library = fetch_ligand_by_name(name, os.path.join("ligands", f"{name.lower().replace(' ', '_')}.sdf"))
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return
        elif choice == "3":
            try:
                cid = input("Enter PubChem CID: ").strip()
                from utils.fetch import fetch_ligand_by_cid
                library = fetch_ligand_by_cid(cid, os.path.join("ligands", f"CID_{cid}.sdf"))
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return
        else:
            default_hint = " [default: ligands/]" if os.path.isdir("ligands") else ""
            try:
                lib = input(f"Ligand file or directory path{default_hint}: ").strip()
                library = lib if lib else "ligands"
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return

    # Auto-fetch if compound name passed directly and doesn't exist locally
    if library and not os.path.exists(library):
        print(f"Library '{library}' not found locally. Attempting to fetch from PubChem...")
        from utils.fetch import fetch_ligand_by_name
        try:
            library = fetch_ligand_by_name(library, os.path.join("ligands", f"{library.lower().replace(' ', '_')}.sdf"))
        except Exception as e:
            print(f"Failed to fetch from PubChem: {e}")

    config = load_config(config_path=args.config)
    run_drug_selection(
        protein_pdb=protein,
        ligand_library=library,
        output_dir=args.output_dir,
        config=config,
        stop_after=args.stop_after,
        resume_from=args.resume,
    )


# ══════════════════════════════════════════════════════════════
# MODE 3: Full Automation — `druglab optimize`
# ══════════════════════════════════════════════════════════════

def _add_optimize_subcommand(subparsers):
    opt_parser = subparsers.add_parser(
        "optimize",
        aliases=["optimize-run"],
        help="Run full lead optimization pipeline",
    )
    opt_parser.add_argument("--lead", required=True, help="Lead molecule SDF file")
    opt_parser.add_argument("--protein", required=True, help="Prepared protein PDBQT file")
    opt_parser.add_argument("--box", required=True, help="Docking box JSON file")
    opt_parser.add_argument("--output-dir", default="results", help="Output directory")
    opt_parser.add_argument("--config", default=None, help="Path to YAML config file")
    opt_parser.set_defaults(handler=_handle_optimize)


def _handle_optimize(args):
    from utils.logger import print_banner
    from utils.config import load_config
    from lead_optimization.workflow import run_lead_optimization

    print_banner()

    config = load_config(config_path=args.config)
    run_lead_optimization(
        lead_sdf=args.lead,
        protein_pdbqt=args.protein,
        docking_box=args.box,
        output_dir=args.output_dir,
        config=config,
    )


# ══════════════════════════════════════════════════════════════
# MODE 2: CLI Module Commands
# ══════════════════════════════════════════════════════════════

# ── protein prepare ──

def _add_protein_subcommand(subparsers):
    protein_parser = subparsers.add_parser("protein", help="Protein operations")
    protein_sub = protein_parser.add_subparsers(dest="protein_cmd")

    prep = protein_sub.add_parser("prepare", help="Prepare a protein PDB file")
    prep.add_argument("--input", required=True, help="Input PDB file")
    prep.add_argument("--output-dir", default="results", help="Output directory")
    prep.add_argument("--chain", default=None, help="Chain ID to select")
    prep.add_argument("--ph", type=float, default=7.4, help="pH for protonation")
    prep.add_argument("--keep-water", action="store_true", help="Keep water molecules")
    prep.add_argument("--config", default=None, help="YAML config file")
    prep.set_defaults(handler=_handle_protein_prepare)


def _handle_protein_prepare(args):
    from utils.config import load_config
    from core.protein_preparation import prepare_protein

    config = load_config(config_path=args.config)
    prepare_protein(
        input_pdb=args.input,
        output_dir=args.output_dir,
        chain_id=args.chain,
        remove_water=not args.keep_water,
        ph=args.ph,
    )


# ── ligand prepare ──

def _add_ligand_subcommand(subparsers):
    ligand_parser = subparsers.add_parser("ligand", help="Ligand operations")
    ligand_sub = ligand_parser.add_subparsers(dest="ligand_cmd")

    prep = ligand_sub.add_parser("prepare", help="Prepare ligands")
    prep.add_argument("--input", required=True, help="Input SDF file or directory")
    prep.add_argument("--output-dir", default="results", help="Output directory")
    prep.add_argument("--no-3d", action="store_true", help="Skip 3D generation")
    prep.add_argument("--conformers", type=int, default=1, help="Number of conformers")
    prep.add_argument("--config", default=None, help="YAML config file")
    prep.set_defaults(handler=_handle_ligand_prepare)


def _handle_ligand_prepare(args):
    from utils.config import load_config
    from core.ligand_preparation import prepare_ligands

    config = load_config(config_path=args.config)
    prepare_ligands(
        input_path=args.input,
        output_dir=args.output_dir,
        generate_3d=not args.no_3d,
        num_conformers=args.conformers,
    )


# ── pocket detect ──

def _add_pocket_subcommand(subparsers):
    pocket_parser = subparsers.add_parser("pocket", help="Pocket operations")
    pocket_sub = pocket_parser.add_subparsers(dest="pocket_cmd")

    detect = pocket_sub.add_parser("detect", help="Detect binding pockets")
    detect.add_argument("--protein", required=True, help="Protein PDB file")
    detect.add_argument("--output-dir", default="results", help="Output directory")
    detect.add_argument("--method", default="geometric",
                        choices=["geometric", "ligand_based", "manual"],
                        help="Detection method")
    detect.add_argument("--reference-ligand", default=None,
                        help="Reference ligand for ligand_based mode")
    detect.add_argument("--center", nargs=3, type=float, default=None,
                        help="Manual box center (x y z)")
    detect.add_argument("--size", nargs=3, type=float, default=None,
                        help="Manual box size (sx sy sz)")
    detect.add_argument("--config", default=None, help="YAML config file")
    detect.set_defaults(handler=_handle_pocket_detect)


def _handle_pocket_detect(args):
    from utils.config import load_config
    from core.pocket_detection import detect_pockets

    config = load_config(config_path=args.config)
    kwargs = {
        "protein_pdb": args.protein,
        "output_dir": args.output_dir,
        "method": args.method,
    }
    if args.reference_ligand:
        kwargs["reference_ligand"] = args.reference_ligand
    if args.center:
        kwargs["manual_center"] = tuple(args.center)
    if args.size:
        kwargs["manual_size"] = tuple(args.size)

    detect_pockets(**kwargs)


# ── screen druglikeness / admet ──

def _add_screen_subcommand(subparsers):
    screen_parser = subparsers.add_parser("screen", help="Screening operations")
    screen_sub = screen_parser.add_subparsers(dest="screen_cmd")

    # druglikeness
    dl = screen_sub.add_parser("druglikeness", help="Drug-likeness screening")
    dl.add_argument("--input", required=True, help="Input SDF file")
    dl.add_argument("--output-dir", default="results", help="Output directory")
    dl.add_argument("--rules", nargs="+", default=["lipinski"],
                    help="Rule sets (lipinski veber ghose egan muegge pains brenk)")
    dl.add_argument("--strict", action="store_true", help="All rules must pass")
    dl.add_argument("--max-violations", type=int, default=1,
                    help="Max violations for Lipinski/Muegge")
    dl.add_argument("--config", default=None, help="YAML config file")
    dl.set_defaults(handler=_handle_screen_druglikeness)

    # admet
    admet = screen_sub.add_parser("admet", help="ADMET screening")
    admet.add_argument("--input", required=True, help="Input SDF file")
    admet.add_argument("--output-dir", default="results", help="Output directory")
    admet.add_argument("--no-filter-toxic", action="store_true",
                       help="Don't filter toxic molecules")
    admet.add_argument("--config", default=None, help="YAML config file")
    admet.set_defaults(handler=_handle_screen_admet)


def _handle_screen_druglikeness(args):
    from utils.config import load_config
    from core.druglikeness import screen_druglikeness

    config = load_config(config_path=args.config)
    screen_druglikeness(
        input_sdf=args.input,
        output_dir=args.output_dir,
        rules=args.rules,
        strict=args.strict,
        max_violations=args.max_violations,
    )


def _handle_screen_admet(args):
    from utils.config import load_config
    from core.admet import screen_admet

    config = load_config(config_path=args.config)
    screen_admet(
        input_sdf=args.input,
        output_dir=args.output_dir,
        filter_toxic=not args.no_filter_toxic,
    )


# ── dock ──

def _add_dock_subcommand(subparsers):
    dock_parser = subparsers.add_parser("dock", help="Molecular docking")
    dock_parser.add_argument("--protein", required=True, help="Protein PDBQT file")
    dock_parser.add_argument("--ligands", required=True,
                             help="Ligands SDF file or PDBQT directory")
    dock_parser.add_argument("--box", required=True, help="Docking box JSON file")
    dock_parser.add_argument("--output-dir", default="results", help="Output directory")
    dock_parser.add_argument("--exhaustiveness", type=int, default=32,
                             help="Vina exhaustiveness")
    dock_parser.add_argument("--n-poses", type=int, default=10,
                             help="Number of poses per ligand")
    dock_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    dock_parser.add_argument("--config", default=None, help="YAML config file")
    dock_parser.set_defaults(handler=_handle_dock)


def _handle_dock(args):
    from utils.config import load_config
    from core.docking import run_docking

    config = load_config(config_path=args.config)
    run_docking(
        protein_pdbqt=args.protein,
        ligands_path=args.ligands,
        docking_box=args.box,
        output_dir=args.output_dir,
        exhaustiveness=args.exhaustiveness,
        n_poses=args.n_poses,
        seed=args.seed,
    )


# ── select ──

def _add_select_subcommand(subparsers):
    sel_parser = subparsers.add_parser("select", help="Lead selection")
    sel_parser.add_argument("--docking-results", required=True,
                            help="Docking results CSV")
    sel_parser.add_argument("--admet-results", default=None,
                            help="ADMET results CSV (optional)")
    sel_parser.add_argument("--druglikeness-results", default=None,
                            help="Druglikeness descriptors CSV (optional)")
    sel_parser.add_argument("--ligands-sdf", default=None,
                            help="Ligands SDF for diversity filter (optional)")
    sel_parser.add_argument("--output-dir", default="results", help="Output directory")
    sel_parser.add_argument("--top-n", type=int, default=5, help="Number of top leads")
    sel_parser.add_argument("--scoring", default="weighted_sum",
                            choices=["weighted_sum", "topsis", "pareto"],
                            help="Scoring method")
    sel_parser.add_argument("--config", default=None, help="YAML config file")
    sel_parser.set_defaults(handler=_handle_select)


def _handle_select(args):
    from utils.config import load_config
    from drug_selection.lead_selection import select_leads

    config = load_config(config_path=args.config)
    select_leads(
        docking_results=args.docking_results,
        output_dir=args.output_dir,
        admet_results=args.admet_results,
        druglikeness_results=args.druglikeness_results,
        ligands_sdf=args.ligands_sdf,
        top_n=args.top_n,
        scoring_method=args.scoring,
    )


# ── report ──

def _add_report_subcommand(subparsers):
    rep_parser = subparsers.add_parser("report", help="Generate standalone interactive HTML report")
    rep_parser.add_argument("--input", required=True, help="Path to DrugLab output directory")
    rep_parser.add_argument("--output", default=None, help="Custom path for HTML report")
    rep_parser.add_argument("--title", default="DrugLab Virtual Screening & Optimization Report", help="Report title")
    rep_parser.set_defaults(handler=_handle_report)


def _handle_report(args):
    from reports.generator import generate_report
    path = generate_report(output_dir=args.input, output_file=args.output, report_title=args.title)
    print(f"Report generated successfully: {path}")


# ── fetch ──

def _add_fetch_subcommand(subparsers):
    fetch_parser = subparsers.add_parser("fetch", help="Download PDB structures or PubChem ligands")
    fetch_sub = fetch_parser.add_subparsers(dest="fetch_type")

    # fetch pdb
    pdb_parser = fetch_sub.add_parser("pdb", help="Fetch PDB from RCSB")
    pdb_parser.add_argument("id", help="4-character PDB ID (e.g. 7DTD, 4MZI)")
    pdb_parser.add_argument("--output", default=None, help="Destination file path")
    pdb_parser.set_defaults(handler=_handle_fetch_pdb)

    # fetch ligand
    lig_parser = fetch_sub.add_parser("ligand", help="Fetch ligand SDF by name from PubChem")
    lig_parser.add_argument("name", help="Compound name (e.g. apigenin, quercetin)")
    lig_parser.add_argument("--output", default=None, help="Destination SDF file path")
    lig_parser.set_defaults(handler=_handle_fetch_ligand)

    # fetch cid
    cid_parser = fetch_sub.add_parser("cid", help="Fetch ligand SDF by PubChem CID")
    cid_parser.add_argument("cid", help="PubChem CID number")
    cid_parser.add_argument("--output", default=None, help="Destination SDF file path")
    cid_parser.set_defaults(handler=_handle_fetch_cid)


def _handle_fetch_pdb(args):
    from utils.fetch import fetch_pdb
    out = fetch_pdb(pdb_id=args.id, output_path=args.output)
    print(f"Downloaded PDB structure to: {out}")


def _handle_fetch_ligand(args):
    from utils.fetch import fetch_ligand_by_name
    out = fetch_ligand_by_name(name=args.name, output_path=args.output)
    print(f"Downloaded ligand SDF to: {out}")


def _handle_fetch_cid(args):
    from utils.fetch import fetch_ligand_by_cid
    out = fetch_ligand_by_cid(cid=args.cid, output_path=args.output)
    print(f"Downloaded ligand SDF to: {out}")


# ══════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
