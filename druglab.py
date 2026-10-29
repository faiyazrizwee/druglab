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
import re
import glob
import argparse
import warnings

# Suppress known third-party deprecation warnings (e.g. admet_ai resources.path)
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"admet_ai.*")
warnings.filterwarnings("ignore", message=r".*path is deprecated.*")
warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated.*")

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
    _add_peptide_subcommand(subparsers)

    args = parser.parse_args()

    if getattr(args, "verbose", False):
        from utils.logger import set_verbose
        set_verbose(True)

    # Mode 1: No command → interactive menu
    if args.command is None:
        _run_interactive()
        return

    # Direct peptide command with no subcommand → interactive peptide submenu
    if args.command == "peptide" and getattr(args, "peptide_cmd", None) is None:
        _run_peptide_interactive()
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
    """Launch the top-level DrugLab 2.0 docking workflow selector."""
    if not sys.stdin.isatty():
        print("DrugLab 2.0 — Non-interactive invocation without subcommand. Run 'druglab --help' for usage.")
        return

    from rich.console import Console
    from rich.panel import Panel
    console = Console()

    banner_text = (
        "[bold cyan]DrugLab 2.0[/bold cyan]\n"
        "[dim]Computational Drug Discovery Platform[/dim]\n\n"
        "Select docking workflow:\n\n"
        "  [bold yellow]1.[/bold yellow] Small-molecule–protein docking (Virtual screening & Lead optimization)\n"
        "  [bold yellow]2.[/bold yellow] Protein–peptide docking (LightDock engine)\n"
        "  [bold yellow]3.[/bold yellow] Exit"
    )
    console.print()
    console.print(Panel(banner_text, border_style="cyan", padding=(0, 2)))

    try:
        choice = input("\nEnter choice [1-3] (default: 1): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return

    if choice in ("3", "exit", "quit", "q", "Q"):
        print("Exiting DrugLab. Goodbye!")
        return
    elif choice in ("2", "peptide"):
        _run_peptide_interactive()
    else:
        # Option 1: existing small-molecule interactive workflow
        from utils.config import load_config
        from core.workflow_engine import create_default_engine

        config = load_config()
        engine = create_default_engine()
        engine.run_interactive(config=config)


def _prompt_terminal_chemistry():
    """Prompt user for peptide terminal chemistry selection, defaulting to physiological zwitterion."""
    print("\nSelect peptide terminal chemistry:")
    print("  [1] Physiological zwitterionic termini (-NH3+, -COO-) [default]")
    print("  [2] Capped termini (N-terminal ACE, C-terminal NHE / amide)")
    print("  [3] Cyclic peptide (head-to-tail backbone cyclization, no free termini)")
    print("  [4] Neutral free termini (-NH2, -COOH)")
    try:
        tchoice = input("Choice [1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return "zwitterion", "zwitterion", False
    if tchoice == "2":
        return "ace", "nhe", False
    elif tchoice == "3":
        return "cyclic", "cyclic", True
    elif tchoice == "4":
        return "free", "free", False
    return "zwitterion", "zwitterion", False


def _run_peptide_interactive():
    """Launch the dedicated Protein–Peptide Docking interactive submenu mirroring Small-Molecule workflow."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    console = Console()

    while True:
        banner = (
            "[bold magenta]🧬 DrugLab 2.0 — Protein–Peptide Docking Platform[/bold magenta]\n"
            "[dim]LightDock GSO Engine, ADMET Screening, 3D Pocket Tuner & Multi-Objective Lead Selection[/dim]"
        )
        console.print()
        console.print(Panel(banner, border_style="magenta", padding=(0, 2)))

        table = Table(
            title="Select Workflow or Computational Modules",
            border_style="dim",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Option", style="bold yellow", width=10, justify="center")
        table.add_column("Category", style="bold cyan", width=12)
        table.add_column("Workflow / Module Name", style="bold white", width=28)
        table.add_column("Description", style="dim", width=46)

        # Presets
        table.add_row("[A]", "Pipeline", "Complete Peptide Pipeline", "Steps 1–6: Protein Prep → Peptide Prep → Pocket → ADMET → Docking → Leads")
        table.add_row("[C]", "Custom", "Custom Module Range", "Run selected step range (e.g. 1-3, or 1,2,5)")
        table.add_section()

        # Modules
        table.add_row("[1]", "Module", "Protein Preparation", "Chains, assembly, altlocs, selective water/metal/ligand retention")
        table.add_row("[2]", "Module", "Peptide (Ligand) Prep", "Sequence/PDB, zwitterion/caps, cyclization, S-S bonds, 3D conformers")
        table.add_row("[3]", "Module", "Pocket Detection", "Cavity detection & interactive 3D box tuner GUI → LightDock restraints")
        table.add_row("[4]", "Module", "ADMET & Safety Profile", "Physicochemical descriptors, Caco-2/PAMPA/RRCK/BBB permeability, hemolytic risk")
        table.add_row("[5]", "Module", "LightDock Docking", "Glowworm Swarm Optimization with DFIRE potentials and restraints")
        table.add_row("[6]", "Module", "Lead Selection & Triage", "Multi-objective ranking (Score, contacts, H-bonds, clashes, toxicity)")
        table.add_section()

        # Utilities
        table.add_row("[R]", "Utility", "Generate HTML Report", "Standalone interactive 3Dmol report with Focus Peptide & style controls")
        table.add_row("[Q]", "System", "Return to Main Menu", "Return to top-level DrugLab menu")

        console.print(table)
        console.print()

        try:
            choice_raw = input("Selection [default: A (Complete Pipeline)]: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Returning to main menu.[/dim]")
            return

        choice = choice_raw.upper() if choice_raw else "A"

        if choice in ("Q", "RETURN", "BACK", "EXIT"):
            return
        elif choice == "A":
            class Args:
                mode = "peptide"
                protein = None
                peptide_fasta = None
                peptide_seq = None
                peptide_structure = None
                output_dir = "results"
                config = None
                preset = "fast"
                scoring = "fastdfire"
                binding_mode = None
                restraints = None
                admet_provider = None
            _handle_run_peptide(Args())
            return
        elif choice == "C":
            _interactive_peptide_custom_range()
            return
        elif choice == "1":
            _interactive_peptide_protein_prep()
            return
        elif choice == "2":
            _interactive_peptide_ligand_prep()
            return
        elif choice == "3":
            _interactive_peptide_pocket_detection()
            return
        elif choice == "4":
            _interactive_peptide_admet()
            return
        elif choice == "5":
            _interactive_peptide_docking()
            return
        elif choice == "6":
            _interactive_peptide_lead_selection()
            return
        elif choice == "R":
            _interactive_peptide_report()
            return
        else:
            console.print(f"[yellow]Unrecognized option: '{choice_raw}'. Please choose from the menu.[/yellow]")


def _interactive_peptide_protein_prep():
    """Interactive handler for Module 1: Protein Preparation."""
    from core.peptide_docking.protein_preparation import inspect_protein_structure, prepare_receptor_for_lightdock
    from utils.fetch import fetch_pdb

    print("\n--- Module 1: Receptor Protein Preparation ---")
    print("Provide receptor protein target:")
    print("  [1] Enter local PDB file path")
    print("  [2] Download from RCSB PDB by ID (e.g. 7DTD, 1SSH, 6LU7)")
    try:
        choice = input("Choice [1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    if choice == "2":
        try:
            pdb_id = input("Enter 4-character RCSB PDB ID: ").strip().upper()
            protein = fetch_pdb(pdb_id, os.path.join("protein", f"{pdb_id}.pdb"))
        except (EOFError, KeyboardInterrupt):
            return
    else:
        default_hint = " [default: protein/7DTD.pdb]" if os.path.isfile("protein/7DTD.pdb") else ""
        try:
            p = input(f"Protein PDB file path{default_hint}: ").strip()
            protein = p if p else "protein/7DTD.pdb"
        except (EOFError, KeyboardInterrupt):
            return

    if not os.path.isfile(protein):
        print(f"Error: Protein file not found: {protein}")
        return

    print(f"\n[1/3] Inspecting receptor structure: {protein}...")
    insp = inspect_protein_structure(protein)
    print(f"  • Chains detected: {', '.join(insp.get('chains', []))}")
    print(f"  • Total residues: {insp.get('num_residues', 0)}")
    print(f"  • Crystallographic waters: {insp.get('num_waters', 0)}")
    print(f"  • Metals: {', '.join(insp.get('metals', [])) if insp.get('metals') else 'None'}")
    print(f"  • Cofactors: {', '.join(insp.get('cofactors', [])) if insp.get('cofactors') else 'None'}")
    if insp.get("missing_residues"):
        print(f"  • Missing residues: {len(insp['missing_residues'])} (REMARK 465)")
    if insp.get("alternate_conformations"):
        print(f"  • Alternate conformations: {len(insp['alternate_conformations'])} residues (resolved to highest occupancy)")

    # Water policy selection
    print("\n[2/3] Select water policy:")
    print("  [1] Remove bulk waters, retain interfacial waters within 4.5Å of binding surface [default]")
    print("  [2] Retain all crystallographic waters")
    print("  [3] Remove all waters")
    try:
        w_choice = input("Choice [1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    w_map = {"1": "remove_bulk", "": "remove_bulk", "2": "keep_all", "3": "remove_all"}
    water_policy = w_map.get(w_choice, "remove_bulk")

    # Chain selection
    try:
        c_input = input(f"\nEnter chains to keep (comma-separated, or press Enter for all [{','.join(insp.get('chains', []))}])): ").strip()
        chains = [c.strip() for c in c_input.split(",") if c.strip()] if c_input else None
    except (EOFError, KeyboardInterrupt):
        return

    out_dir = "results/prepared_proteins"
    print(f"\n[3/3] Preparing LightDock-compatible receptor in {out_dir}...")
    res = prepare_receptor_for_lightdock(
        input_pdb=protein,
        output_dir=out_dir,
        chains=chains,
        water_policy=water_policy,
        keep_metals=True,
        keep_cofactors=True,
    )
    print(f"✓ Receptor preparation complete!")
    print(f"  Prepared PDB: {res['prepared_path']}")
    print(f"  Chains retained: {res['chains']}")


def _interactive_peptide_ligand_prep():
    """Interactive handler for Module 2: Peptide Ligand Preparation."""
    from core.peptide_docking.input_validation import PeptideCandidate, validate_peptide_sequence, parse_fasta_file
    from core.peptide_docking.peptide_preparation import prepare_peptide_candidate

    print("\n--- Module 2: Peptide (Ligand) Preparation ---")
    print("Provide peptide input:")
    print("  [1] Enter amino-acid sequence (e.g. ACDEF)")
    print("  [2] Enter local FASTA file path (single or multi-peptide library)")
    print("  [3] Enter local peptide PDB structure path")
    try:
        choice = input("Choice [1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    candidates = []
    if choice == "2":
        try:
            f_path = input("Enter FASTA file path: ").strip()
            if os.path.isfile(f_path):
                candidates = parse_fasta_file(f_path)
            else:
                print(f"Error: FASTA file not found: {f_path}")
                return
        except (EOFError, KeyboardInterrupt):
            return
    elif choice == "3":
        try:
            s_path = input("Enter peptide PDB file path: ").strip()
            if os.path.isfile(s_path):
                candidates = [PeptideCandidate(id="peptide_pdb", sequence="", source="structure", structure_path=s_path)]
            else:
                print(f"Error: Peptide PDB not found: {s_path}")
                return
        except (EOFError, KeyboardInterrupt):
            return
    else:
        try:
            seq = input("Enter amino-acid sequence: ").strip()
            if not seq:
                print("Error: No sequence provided.")
                return
            v_seq = validate_peptide_sequence(seq)
            candidates = [PeptideCandidate(id="peptide_1", sequence=v_seq, source="sequence")]
        except (EOFError, KeyboardInterrupt):
            return

    n_cap, c_cap, is_cyclic = _prompt_terminal_chemistry()

    out_dir = "results/prepared_peptides"
    print(f"\nPreparing {len(candidates)} peptide candidate(s) (termini: N={n_cap}, C={c_cap}, cyclic={is_cyclic})...")
    for cand in candidates:
        try:
            res = prepare_peptide_candidate(
                candidate=cand,
                output_dir=out_dir,
                n_term_cap=n_cap,
                c_term_cap=c_cap,
                is_cyclic=is_cyclic,
            )
            print(f"✓ Candidate '{cand.id}' prepared: {res.prepared_structure_path}")
        except Exception as e:
            print(f"✗ Failed preparing candidate '{cand.id}': {e}")


def _interactive_peptide_pocket_detection():
    """Interactive handler for Module 3: Pocket Detection & Restraints."""
    from core.peptide_docking.pocket_detection import interactive_peptide_pocket_detection

    print("\n--- Module 3: Pocket Detection & LightDock Restraints ---")
    default_rec = "results/prepared_proteins/receptor.pdb"
    if not os.path.isfile(default_rec):
        default_rec = "protein/7DTD.pdb"
    default_hint = f" [default: {default_rec}]" if os.path.isfile(default_rec) else ""
    try:
        rec_path = input(f"Enter receptor PDB file path{default_hint}: ").strip()
        receptor = rec_path if rec_path else default_rec
    except (EOFError, KeyboardInterrupt):
        return

    if not os.path.isfile(receptor):
        print(f"Error: Receptor file not found: {receptor}")
        return

    res = interactive_peptide_pocket_detection(
        receptor_pdb=receptor,
        output_dir="results/pocket",
        auto_confirm=False,
    )
    print(f"\n✓ Pocket detection complete!")
    if res.get("restraints_path"):
        print(f"  Restraints file: {res['restraints_path']}")
        print(f"  Total restraints generated: {res.get('num_restraints', 0)}")


def _interactive_peptide_admet():
    """Interactive handler for Module 4: Peptide ADMET Screening."""
    from core.peptide_docking.input_validation import PeptideCandidate, parse_fasta_file
    from core.peptide_docking.peptide_admet import predict_peptide_admet, display_peptide_admet_summary
    from core.peptide_docking.config import AdmetConfig

    print("\n--- Module 4: Peptide ADMET, Permeability & Safety Profile ---")
    print("Provide peptide input:")
    print("  [1] Enter sequence (single or comma-separated)")
    print("  [2] Enter FASTA file path")
    try:
        choice = input("Choice [1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    candidates = []
    if choice == "2":
        try:
            f_path = input("Enter FASTA file path: ").strip()
            if os.path.isfile(f_path):
                candidates = parse_fasta_file(f_path)
            else:
                print(f"Error: FASTA not found: {f_path}")
                return
        except (EOFError, KeyboardInterrupt):
            return
    else:
        try:
            seq = input("Enter peptide sequence (e.g. ACDEF,WYVIK): ").strip()
            if not seq:
                print("Error: No sequence provided.")
                return
            tokens = [s.strip() for s in re.split(r"[,;\n\r]+", seq) if s.strip()]
            for i, t in enumerate(tokens, 1):
                candidates.append(PeptideCandidate(id=f"peptide_{i}", sequence=t, source="sequence"))
        except (EOFError, KeyboardInterrupt):
            return

    print("\nSelect ADMET provider:")
    print("  [1] Local Peptide ADMET Engine (Descriptors, Caco-2, BBB, toxicity & half-life) [default]")
    print("  [2] PEPLife (Half-life & stability)")
    try:
        pchoice = input("Choice [1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    prov = "peplife" if pchoice == "2" else "local"
    cfg = AdmetConfig(provider=prov, enabled=True)
    out_dir = "results/admet"
    df = predict_peptide_admet(candidates=candidates, config=cfg, output_dir=out_dir)
    display_peptide_admet_summary(df, provider_name=prov)
    print(f"ADMET results saved to: {os.path.join(out_dir, 'admet_predictions.tsv')}")


def _interactive_peptide_docking():
    """Interactive handler for Module 5: LightDock Docking."""
    from core.peptide_docking.lightdock_runner import run_lightdock_docking
    from core.peptide_docking.config import LightdockConfig
    from core.peptide_docking.pose_analysis import analyze_docked_pose

    print("\n--- Module 5: LightDock Docking Engine ---")
    try:
        rec = input("Enter prepared receptor PDB: ").strip()
        pep = input("Enter prepared peptide PDB: ").strip()
        rst = input("Enter LightDock restraints file path (optional, press Enter for blind): ").strip() or None
    except (EOFError, KeyboardInterrupt):
        return

    if not os.path.isfile(rec) or not os.path.isfile(pep):
        print("Error: Receptor or peptide PDB file not found.")
        return

    print("\nSelect LightDock sampling preset:")
    print("  [1] Fast (10 swarms, 10 steps) [default]")
    print("  [2] Balanced (30 swarms, 25 steps)")
    print("  [3] Thorough (100 swarms, 50 steps)")
    try:
        pchoice = input("Choice [1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    preset_map = {"1": "fast", "": "fast", "2": "balanced", "3": "thorough"}
    preset = preset_map.get(pchoice, "fast")

    cfg = LightdockConfig(preset=preset)
    out_dir = "results/docking"
    res = run_lightdock_docking(
        candidate_id="peptide_cand",
        receptor_pdb=rec,
        peptide_pdb=pep,
        output_dir=out_dir,
        config=cfg,
        restraints_path=rst,
    )
    if res.status == "completed":
        print(f"\n✓ Docking completed successfully! Generated {len(res.poses)} poses.")
        for p in res.poses:
            p_path = p.get("pdb_path")
            if p_path and os.path.isfile(p_path):
                met = analyze_docked_pose(p_path)
                p.update(met)
            print(f"  Pose {p.get('rank')}: Score={p.get('lightdock_score')} | PDB: {p.get('pdb_path')}")
    else:
        print(f"Docking failed: {res.error_message}")


def _interactive_peptide_lead_selection():
    """Interactive handler for Module 6: Lead Selection & Triage."""
    from core.peptide_docking.lead_selection import rank_and_select_peptide_leads

    print("\n--- Module 6: Multi-Objective Lead Selection ---")
    runs = sorted(glob.glob("results/peptide_run_*"))
    default_run = runs[-1] if runs else "results"
    try:
        r_path = input(f"Enter peptide run directory [default: {default_run}]: ").strip()
        run_dir = r_path if r_path else default_run
    except (EOFError, KeyboardInterrupt):
        return

    leads_df = rank_and_select_peptide_leads(
        candidates=[],
        output_dir=run_dir,
        top_n=5,
    )
    if not leads_df.empty:
        print(f"\n✓ Top {len(leads_df)} Leads Selected:")
        for _, r in leads_df.iterrows():
            print(f"  #{r.get('rank')}: {r.get('candidate_id')} (Composite: {r.get('composite_score'):.2f}, LightDock: {r.get('lightdock_score'):.2f}, Tox: {r.get('hemolytic_risk')})")
        print(f"Lead summary: {os.path.join(run_dir, 'lead_peptides.csv')}")
        print(f"Lead complexes: {os.path.join(run_dir, 'leads')}")
    else:
        print(f"No leads selected. Check that docking_results.tsv exists in {run_dir}.")


def _interactive_peptide_report():
    """Interactive handler for Utility [R]: HTML Report."""
    from core.peptide_docking.reporting import generate_peptide_html_report
    import json

    print("\n--- Utility: Generate / View Peptide HTML Report ---")
    runs = sorted(glob.glob("results/peptide_run_*"))
    default_run = runs[-1] if runs else "results"
    try:
        r_path = input(f"Enter peptide run directory [default: {default_run}]: ").strip()
        run_dir = r_path if r_path else default_run
    except (EOFError, KeyboardInterrupt):
        return

    manifest_path = os.path.join(run_dir, "run_manifest.json")
    manifest = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as mf:
            manifest = json.load(mf)

    rep = generate_peptide_html_report(
        output_dir=run_dir,
        manifest=manifest,
    )
    print(f"\n✓ Report generated successfully!")
    print(f"  Interactive HTML Report: file://{rep}")


def _interactive_peptide_custom_range():
    """Interactive handler for [C] Custom Module Range."""
    print("\nPeptide Pipeline Steps:")
    print("  1. Protein Preparation")
    print("  2. Peptide (Ligand) Preparation")
    print("  3. Pocket Detection (3D Box GUI Tuner)")
    print("  4. ADMET & Safety Screening")
    print("  5. LightDock Docking")
    print("  6. Lead Selection & Triage")
    print("  R. HTML Report")
    try:
        rng = input("\nEnter module sequence (e.g. 1-3, 1,2,5, or 1-6): ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    steps = []
    for part in re.split(r"[, ]+", rng):
        if "-" in part:
            sub = part.split("-")
            if len(sub) == 2 and sub[0].isdigit() and sub[1].isdigit():
                steps.extend(range(int(sub[0]), int(sub[1]) + 1))
        elif part.isdigit():
            steps.append(int(part))
        elif part.upper() == "R":
            steps.append("R")

    for step in steps:
        if step == 1:
            _interactive_peptide_protein_prep()
        elif step == 2:
            _interactive_peptide_ligand_prep()
        elif step == 3:
            _interactive_peptide_pocket_detection()
        elif step == 4:
            _interactive_peptide_admet()
        elif step == 5:
            _interactive_peptide_docking()
        elif step == 6:
            _interactive_peptide_lead_selection()
        elif step == "R":
            _interactive_peptide_report()


# ══════════════════════════════════════════════════════════════
# MODE 3: Full Automation — `druglab run`
# ══════════════════════════════════════════════════════════════

def _add_run_subcommand(subparsers):
    run_parser = subparsers.add_parser(
        "run",
        help="Run full drug selection or peptide docking pipeline",
    )
    run_parser.add_argument(
        "--mode",
        default="small-molecule",
        choices=["small-molecule", "peptide"],
        help="Docking pipeline mode: small-molecule (default) or peptide",
    )
    run_parser.add_argument("--protein", default=None, help="Input protein PDB file or 4-letter PDB ID (e.g. 7DTD)")
    run_parser.add_argument("--library", default=None, help="Ligand library (SDF, directory, or compound name)")
    run_parser.add_argument("--output-dir", default="results", help="Output directory")
    run_parser.add_argument("--config", default=None, help="Path to YAML config file")
    run_parser.add_argument("--stop-after", default=None,
                            help="Stop after this stage (small-molecule mode)")
    run_parser.add_argument("--resume", default=None,
                            help="Resume from this stage (small-molecule mode)")

    # Peptide-specific options for --mode peptide
    run_parser.add_argument("--peptide-fasta", default=None, help="Peptide FASTA input (peptide mode)")
    run_parser.add_argument("--peptide-seq", default=None, help="Peptide amino-acid sequence (peptide mode)")
    run_parser.add_argument("--peptide-structure", default=None, help="Peptide PDB structure (peptide mode)")
    run_parser.add_argument("--preset", default="fast", choices=["fast", "balanced", "thorough"], help="Sampling preset (peptide mode)")
    run_parser.add_argument("--scoring", default="fastdfire", help="Scoring function (peptide mode)")
    run_parser.add_argument("--binding-mode", default="auto", choices=["auto", "explicit", "pocket_guided", "global"], help="Binding-site mode (peptide mode)")
    run_parser.add_argument("--restraints", nargs="+", default=None, help="Receptor residue restraints (e.g. A.TYR.45)")
    run_parser.add_argument(
        "--admet-provider",
        default=None,
        choices=["local", "peplife", "none", "pepadmet"],
        help="Peptide ADMET checking provider (peptide mode): local (default), peplife, or none",
    )

    run_parser.set_defaults(handler=_handle_run)


def _handle_run(args):
    # Dispatch based on mode
    if getattr(args, "mode", "small-molecule") == "peptide":
        _handle_run_peptide(args)
        return

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


def _handle_run_peptide(args):
    """Handler for protein–peptide docking runs."""
    from core.peptide_docking.pipeline import run_peptide_docking_pipeline
    from core.peptide_docking.config import load_peptide_config

    overrides = {}
    if getattr(args, "preset", None):
        overrides.setdefault("lightdock", {})["preset"] = args.preset
    if getattr(args, "scoring", None):
        overrides.setdefault("lightdock", {})["scoring_function"] = args.scoring
    if getattr(args, "binding_mode", None):
        overrides.setdefault("binding_site", {})["mode"] = args.binding_mode
    if getattr(args, "restraints", None):
        overrides.setdefault("binding_site", {})["receptor_residues"] = args.restraints
        overrides.setdefault("binding_site", {})["mode"] = "explicit"
    if getattr(args, "admet_provider", None):
        overrides.setdefault("admet", {})["provider"] = args.admet_provider
        overrides.setdefault("admet", {})["enabled"] = (args.admet_provider != "none")

    cfg = load_peptide_config(config_path=getattr(args, "config", None), overrides=overrides)

    protein = getattr(args, "protein", None) or cfg.protein.input
    if not protein:
        print("\nProvide receptor protein target:")
        print("  [1] Enter local PDB file path")
        print("  [2] Download from RCSB PDB by ID (e.g. 7DTD, 1SSH, 6LU7)")
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

    fasta = getattr(args, "peptide_fasta", None) or cfg.peptide.fasta
    seq = getattr(args, "peptide_seq", None) or cfg.peptide.sequence
    struct = getattr(args, "peptide_structure", None) or cfg.peptide.structure

    if not fasta and not seq and not struct:
        print("\nProvide peptide candidate:")
        print("  [1] Enter amino-acid sequence (single or comma-separated, e.g. ACDEF,WYVIK)")
        print("  [2] Enter local FASTA file path (single or multi-peptide library)")
        print("  [3] Enter local peptide PDB structure path")
        try:
            pchoice = input("Choice [1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return

        if pchoice == "2":
            try:
                fasta = input("Enter FASTA file path (or directory): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return
            if fasta and os.path.exists(fasta):
                from core.peptide_docking.input_validation import parse_fasta_file
                try:
                    loaded = parse_fasta_file(fasta)
                    print(f"Detected {len(loaded)} peptide candidate(s) in {fasta}")
                except Exception as e:
                    print(f"Notice reading FASTA: {e}")
        elif pchoice == "3":
            try:
                struct = input("Enter peptide PDB file path: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return
        else:
            try:
                seq = input("Enter amino-acid sequence: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return

            if seq and (len(seq) < cfg.filtering.min_length_aa or len(seq) > cfg.filtering.max_length_aa):
                try:
                    from rich.console import Console
                    Console().print(
                        f"[bold yellow]⚠ Warning:[/bold yellow] [yellow]Peptide sequence length ({len(seq)} AA) is outside recommended range "
                        f"({cfg.filtering.min_length_aa}–{cfg.filtering.max_length_aa} AA). "
                        f"Length filter is removed; continuing docking workflow.[/yellow]"
                    )
                except Exception:
                    print(f"Warning: Peptide length ({len(seq)} AA) outside recommended range. Length filter bypassed.")

    if getattr(args, "admet_provider", None) is None and sys.stdin.isatty():
        print("\nSelect ADMET checking provider:")
        print("  [1] Local Peptide ADMET Engine (Calculated descriptors, Caco-2, BBB, toxicity & half-life) [default]")
        print("  [2] PEPLife (Peptide half-life, metabolic stability & proteolytic susceptibility)")
        print("  [3] Skip ADMET checking")
        try:
            achoice = input("Choice [1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return

        provider_map = {
            "1": "local",
            "local": "local",
            "": "local",
            "2": "peplife",
            "peplife": "peplife",
            "3": "none",
            "none": "none",
            "4": "none",
            "pepadmet": "local",
        }
        chosen_provider = provider_map.get(achoice.lower(), "local")
        cfg.admet.provider = chosen_provider
        cfg.admet.enabled = (chosen_provider != "none")

    # Terminal capping prompt (defaults to physiological zwitterion)
    if sys.stdin.isatty() and not getattr(args, "n_term_cap", None):
        n_cap, c_cap, is_cyclic = _prompt_terminal_chemistry()
        cfg.peptide.n_term_cap = n_cap
        cfg.peptide.c_term_cap = c_cap
        cfg.peptide.is_cyclic = is_cyclic

    # Binding site / pocket mode prompt (defaults to 3D Box Tuner GUI)
    if sys.stdin.isatty() and not getattr(args, "binding_mode", None) and not getattr(args, "restraints", None):
        print("\nSelect binding site / pocket mode:")
        print("  [1] Cavity detection & 3D GUI Box Tuner (convert box residues to LightDock restraints) [default]")
        print("  [2] Global blind docking (full receptor surface exploration)")
        print("  [3] Enter custom residue restraints (e.g. A.TYR.45, A.ARG.78)")
        try:
            bchoice = input("Choice [1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return

        if bchoice == "2":
            cfg.binding_site.mode = "global"
        elif bchoice == "3":
            try:
                r_raw = input("Enter receptor restraints separated by space/comma: ").strip()
                if r_raw:
                    cfg.binding_site.receptor_residues = [x.strip() for x in re.split(r"[, ]+", r_raw) if x.strip()]
                    cfg.binding_site.mode = "explicit"
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return
        else:
            cfg.binding_site.mode = "pocket_gui"

    res = run_peptide_docking_pipeline(
        protein_input=protein,
        peptide_fasta=fasta,
        peptide_sequence=seq,
        peptide_structure=struct,
        output_dir=getattr(args, "output_dir", "results"),
        config=cfg,
    )
    completed = res.get("manifest", {}).get("completed_docking", 0)
    if completed > 0:
        print(f"\nPeptide docking completed successfully! ({completed} candidate(s) docked)")
    else:
        print(f"\nPeptide docking workflow completed with warnings (no poses successfully docked).")
    print(f"Results stored in: {res['run_dir']}")
    if res.get("report_html"):
        print(f"Interactive HTML Report: file://{res['report_html']}")


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


# ── peptide ──

def _add_peptide_subcommand(subparsers):
    peptide_parser = subparsers.add_parser("peptide", help="Protein–peptide docking operations (LightDock)")
    pep_sub = peptide_parser.add_subparsers(dest="peptide_cmd")

    # peptide run
    pep_run = pep_sub.add_parser("run", help="Run full protein–peptide docking pipeline")
    pep_run.add_argument("--protein", default=None, help="Input protein PDB file or 4-letter PDB ID")
    pep_run.add_argument("--fasta", default=None, help="Peptide FASTA file")
    pep_run.add_argument("--sequence", default=None, help="Peptide sequence (e.g. ACDEF)")
    pep_run.add_argument("--structure", default=None, help="Pre-existing peptide PDB structure")
    pep_run.add_argument("--output-dir", default="results", help="Output directory")
    pep_run.add_argument("--config", default=None, help="Path to peptide YAML config")
    pep_run.add_argument("--preset", default="fast", choices=["fast", "balanced", "thorough"], help="Sampling preset")
    pep_run.add_argument("--scoring", default="fastdfire", help="Scoring function")
    pep_run.add_argument("--binding-mode", default="auto", choices=["auto", "explicit", "pocket_guided", "pocket_gui", "global"])
    pep_run.add_argument("--restraints", nargs="+", default=None, help="Receptor residue restraints")
    pep_run.add_argument(
        "--admet-provider",
        default=None,
        choices=["local", "peplife", "none", "pepadmet"],
        help="Peptide ADMET checking provider: local (default), peplife, or none",
    )
    pep_run.add_argument("--n-term", default="zwitterion", choices=["zwitterion", "ace", "free"], help="N-terminal capping")
    pep_run.add_argument("--c-term", default="zwitterion", choices=["zwitterion", "nhe", "free"], help="C-terminal capping")
    pep_run.add_argument("--cyclic", action="store_true", default=False, help="Head-to-tail backbone cyclization")
    pep_run.set_defaults(handler=_handle_peptide_run)

    # peptide protein-prepare
    pep_pp = pep_sub.add_parser("protein-prepare", help="Inspect and prepare receptor protein for LightDock")
    pep_pp.add_argument("--protein", required=True, help="Input protein PDB file or 4-letter RCSB PDB ID")
    pep_pp.add_argument("--chains", nargs="+", default=None, help="Target chains to keep (e.g. A B)")
    pep_pp.add_argument("--water-policy", default="remove_bulk", choices=["remove_bulk", "keep_all", "remove_all"], help="Water policy")
    pep_pp.add_argument("--keep-metals", action="store_true", default=True, help="Retain coordinated metal ions")
    pep_pp.add_argument("--keep-cofactors", action="store_true", default=True, help="Retain bound organic cofactors")
    pep_pp.add_argument("--assembly", type=int, default=None, help="Biological assembly number (REMARK 350)")
    pep_pp.add_argument("--output-dir", default="results/prepared_proteins", help="Output directory")
    pep_pp.set_defaults(handler=_handle_peptide_protein_prepare)

    # peptide prepare
    pep_prep = pep_sub.add_parser("prepare", help="Prepare a peptide sequence, FASTA, or structure")
    pep_prep.add_argument("--sequence", default=None, help="Peptide amino-acid sequence")
    pep_prep.add_argument("--fasta", default=None, help="Peptide FASTA file path")
    pep_prep.add_argument("--structure", default=None, help="Input peptide PDB file")
    pep_prep.add_argument("--n-term", default="zwitterion", choices=["zwitterion", "ace", "free"], help="N-terminal capping")
    pep_prep.add_argument("--c-term", default="zwitterion", choices=["zwitterion", "nhe", "free"], help="C-terminal capping")
    pep_prep.add_argument("--cyclic", action="store_true", default=False, help="Head-to-tail backbone cyclization")
    pep_prep.add_argument("--conformers", type=int, default=1, help="Conformers count")
    pep_prep.add_argument("--output-dir", default="results/prepared_peptides", help="Output directory")
    pep_prep.set_defaults(handler=_handle_peptide_prepare)

    # peptide pocket
    pep_pkt = pep_sub.add_parser("pocket", help="Detect binding pockets and generate LightDock restraints")
    pep_pkt.add_argument("--protein", required=True, help="Prepared receptor PDB file")
    pep_pkt.add_argument("--gui", action="store_true", default=False, help="Launch interactive 3D Box Tuner GUI in browser")
    pep_pkt.add_argument("--auto-confirm", action="store_true", default=False, help="Auto-select top cavity without prompt")
    pep_pkt.add_argument("--output-dir", default="results/pocket", help="Output directory")
    pep_pkt.set_defaults(handler=_handle_peptide_pocket)

    # peptide admet
    pep_adm = pep_sub.add_parser("admet", help="Evaluate peptide ADMET, permeability & toxicity profile")
    pep_adm.add_argument("--sequence", default=None, help="Peptide sequence (single or comma-separated)")
    pep_adm.add_argument("--fasta", default=None, help="Peptide FASTA file")
    pep_adm.add_argument("--provider", default="local", choices=["local", "peplife", "none"], help="ADMET provider")
    pep_adm.add_argument("--output-dir", default="results/admet", help="Output directory")
    pep_adm.set_defaults(handler=_handle_peptide_admet)

    # peptide dock
    pep_dock = pep_sub.add_parser("dock", help="Run LightDock docking for prepared protein and peptide")
    pep_dock.add_argument("--protein", required=True, help="Prepared receptor PDB")
    pep_dock.add_argument("--peptide", required=True, help="Prepared peptide PDB")
    pep_dock.add_argument("--restraints", default=None, help="Path to restraints.list file")
    pep_dock.add_argument("--preset", default="fast", choices=["fast", "balanced", "thorough"], help="Sampling preset")
    pep_dock.add_argument("--scoring", default="fastdfire", help="Scoring function")
    pep_dock.add_argument("--output-dir", default="results/docking", help="Output directory")
    pep_dock.set_defaults(handler=_handle_peptide_dock)

    # peptide select-leads
    pep_leads = pep_sub.add_parser("select-leads", help="Triage and rank top lead candidates from docking run")
    pep_leads.add_argument("--run-dir", default=None, help="Path to peptide run directory")
    pep_leads.add_argument("--top-n", type=int, default=5, help="Number of leads to select")
    pep_leads.add_argument("--output-dir", default=None, help="Output directory for leads (default: <run-dir>/leads)")
    pep_leads.set_defaults(handler=_handle_peptide_select_leads)

    # peptide report
    pep_rep = pep_sub.add_parser("report", help="Generate or regenerate interactive HTML report for peptide run")
    pep_rep.add_argument("--input", required=True, help="Path to peptide run directory containing run_manifest.json")
    pep_rep.add_argument("--output", default="report.html", help="HTML report output file name")
    pep_rep.set_defaults(handler=_handle_peptide_report)


def _handle_peptide_run(args):
    # Map peptide run arguments into _handle_run_peptide
    class PeptideRunArgs:
        mode = "peptide"
        protein = args.protein
        peptide_fasta = args.fasta
        peptide_seq = args.sequence
        peptide_structure = args.structure
        output_dir = args.output_dir
        config = args.config
        preset = args.preset
        scoring = args.scoring
        binding_mode = args.binding_mode
        restraints = args.restraints
        admet_provider = getattr(args, "admet_provider", None)
        n_term_cap = getattr(args, "n_term", None)
        c_term_cap = getattr(args, "c_term", None)
        is_cyclic = getattr(args, "cyclic", False)

    _handle_run_peptide(PeptideRunArgs())


def _handle_peptide_protein_prepare(args):
    from core.peptide_docking.protein_preparation import inspect_protein_structure, prepare_receptor_for_lightdock
    from utils.fetch import fetch_pdb

    prot = args.protein
    if not os.path.isfile(prot) and len(prot.strip()) == 4 and prot.strip().isalnum():
        print(f"Target '{prot}' appears to be a PDB ID. Fetching from RCSB PDB...")
        prot = fetch_pdb(prot.strip().upper(), os.path.join(args.output_dir, f"{prot.strip().upper()}.pdb"))

    if not os.path.isfile(prot):
        print(f"Error: Protein file not found: {prot}")
        return

    print(f"\n--- Inspecting Receptor Structure: {prot} ---")
    insp = inspect_protein_structure(prot)
    chains_list = insp.get('available_chain_ids', [c['chain_id'] for c in insp.get('chains', []) if isinstance(c, dict)])
    print(f"  • Chains: {', '.join(chains_list) if chains_list else 'None'}")
    print(f"  • Total Residues: {insp.get('total_residues', insp.get('num_residues', 0))}")
    print(f"  • Waters: {insp.get('waters_count', 0)}")
    print(f"  • Metals: {', '.join(insp.get('metals_detected', [])) if insp.get('metals_detected') else 'None'}")
    print(f"  • Cofactors: {', '.join(insp.get('cofactors_detected', [])) if insp.get('cofactors_detected') else 'None'}")
    if insp.get("missing_residues_count", 0) > 0:
        print(f"  • Missing Residues: {insp.get('missing_residues_count')} (REMARK 465)")
    if insp.get("alternate_conformations_count", 0) > 0:
        print(f"  • Alternate Conformations: {insp.get('alternate_conformations_count')} residues with altlocs (resolved to highest occupancy)")

    res = prepare_receptor_for_lightdock(
        input_pdb=prot,
        output_dir=args.output_dir,
        chains=args.chains,
        water_policy=args.water_policy,
        keep_metals=args.keep_metals,
        keep_cofactors=args.keep_cofactors,
        biological_assembly=args.assembly,
    )
    print(f"\nReceptor prepared successfully for LightDock!")
    print(f"Saved to: {res['prepared_path']}")


def _handle_peptide_prepare(args):
    from core.peptide_docking.input_validation import PeptideCandidate, parse_fasta_file, validate_peptide_sequence
    from core.peptide_docking.peptide_preparation import prepare_peptide_candidate

    candidates = []
    if args.fasta and os.path.isfile(args.fasta):
        candidates = parse_fasta_file(args.fasta)
    elif args.structure and os.path.isfile(args.structure):
        candidates = [PeptideCandidate(id="peptide_pdb", sequence="", source="structure", structure_path=args.structure)]
    elif args.sequence:
        seqs = [s.strip() for s in re.split(r"[,;\n\r]+", args.sequence) if s.strip()]
        for i, s in enumerate(seqs, 1):
            candidates.append(PeptideCandidate(id=f"peptide_{i}", sequence=validate_peptide_sequence(s), source="sequence"))
    else:
        print("Error: Must provide either --sequence, --fasta, or --structure to prepare.")
        return

    print(f"\nPreparing {len(candidates)} peptide candidate(s)...")
    for cand in candidates:
        res = prepare_peptide_candidate(
            candidate=cand,
            output_dir=args.output_dir,
            n_term_cap=args.n_term,
            c_term_cap=args.c_term,
            is_cyclic=args.cyclic,
            num_conformers=args.conformers,
        )
        print(f"✓ Candidate '{cand.id}' prepared: {res.prepared_structure_path}")


def _handle_peptide_pocket(args):
    from core.peptide_docking.pocket_detection import interactive_peptide_pocket_detection
    if not os.path.isfile(args.protein):
        print(f"Error: Protein file not found: {args.protein}")
        return
    res = interactive_peptide_pocket_detection(
        receptor_pdb=args.protein,
        output_dir=args.output_dir,
        auto_confirm=args.auto_confirm,
    )
    print(f"\nPocket detection completed.")
    print(f"Restraints file: {res.get('restraints_path')}")
    print(f"Total restraints generated: {res.get('num_restraints', 0)}")


def _handle_peptide_admet(args):
    from core.peptide_docking.input_validation import PeptideCandidate, parse_fasta_file, validate_peptide_sequence
    from core.peptide_docking.peptide_admet import predict_peptide_admet, display_peptide_admet_summary
    from core.peptide_docking.config import AdmetConfig

    candidates = []
    if args.fasta and os.path.isfile(args.fasta):
        candidates = parse_fasta_file(args.fasta)
    elif args.sequence:
        seqs = [s.strip() for s in re.split(r"[,;\n\r]+", args.sequence) if s.strip()]
        for i, s in enumerate(seqs, 1):
            candidates.append(PeptideCandidate(id=f"peptide_{i}", sequence=validate_peptide_sequence(s), source="sequence"))
    else:
        print("Error: Must provide either --sequence or --fasta.")
        return

    cfg = AdmetConfig(provider=args.provider, enabled=(args.provider != "none"))
    df = predict_peptide_admet(candidates, config=cfg, output_dir=args.output_dir)
    display_peptide_admet_summary(df, provider_name=args.provider)
    out_tsv = os.path.join(args.output_dir, "admet_predictions.tsv")
    print(f"ADMET results saved to: {out_tsv}")


def _handle_peptide_dock(args):
    from core.peptide_docking.lightdock_runner import run_lightdock_docking
    from core.peptide_docking.config import LightdockConfig
    from core.peptide_docking.pose_analysis import analyze_docked_pose

    if not os.path.isfile(args.protein) or not os.path.isfile(args.peptide):
        print("Error: Receptor or peptide PDB file not found.")
        return

    cfg = LightdockConfig(preset=args.preset, scoring_function=args.scoring)
    res = run_lightdock_docking(
        candidate_id="peptide_cand",
        receptor_pdb=args.protein,
        peptide_pdb=args.peptide,
        output_dir=args.output_dir,
        config=cfg,
        restraints_path=args.restraints,
    )
    if res.status == "completed":
        print(f"\nDocking completed successfully! Generated {len(res.poses)} poses.")
        for p in res.poses:
            p_path = p.get("pdb_path")
            if p_path and os.path.isfile(p_path):
                met = analyze_docked_pose(p_path)
                p.update(met)
            print(f"  Pose {p.get('rank')}: Score={p.get('lightdock_score')} | PDB: {p.get('pdb_path')}")
    else:
        print(f"Docking failed: {res.error_message}")


def _handle_peptide_select_leads(args):
    from core.peptide_docking.lead_selection import rank_and_select_peptide_leads

    run_dir = args.run_dir
    if not run_dir:
        runs = sorted(glob.glob("results/peptide_run_*"))
        if runs:
            run_dir = runs[-1]
            print(f"Auto-selected latest run: {run_dir}")
        else:
            print("Error: No peptide run directory found or specified.")
            return

    leads_df = rank_and_select_peptide_leads(
        candidates=[],
        output_dir=run_dir,
        top_n=args.top_n,
    )
    if not leads_df.empty:
        print(f"\nTop {len(leads_df)} Leads Selected:")
        for _, r in leads_df.iterrows():
            print(f"  #{r.get('rank')}: {r.get('candidate_id')} (Composite: {r.get('composite_score'):.2f}, LightDock: {r.get('lightdock_score'):.2f}, Tox: {r.get('hemolytic_risk')})")
        print(f"Lead summary saved to: {os.path.join(run_dir, 'lead_peptides.csv')}")
        print(f"Lead complexes saved to: {os.path.join(run_dir, 'leads')}")
    else:
        print("No leads could be selected (check if docking_results.tsv exists in run directory).")


def _handle_peptide_report(args):
    from core.peptide_docking.reporting import generate_peptide_html_report
    import json

    manifest_path = os.path.join(args.input, "run_manifest.json")
    if not os.path.isfile(manifest_path):
        print(f"Error: run_manifest.json not found in {args.input}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    rep = generate_peptide_html_report(
        output_dir=args.input,
        manifest=manifest,
        output_filename=args.output,
    )
    print(f"Report generated successfully: {rep}")


# ══════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
