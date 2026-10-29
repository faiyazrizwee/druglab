"""
Interactive pocket detection, 3D graphical box tuner GUI, and LightDock restraint conversion.

Provides:
1. Automated geometric cavity detection on prepared receptor PDB structures.
2. Interactive selection between top scoring pockets, co-crystallized reference ligands,
   interactive 3D graphical box tuner GUI (core.pocket_gui), and manual definition.
3. Conversion of selected 3D bounding boxes into validated LightDock surface receptor restraints.
4. Export of LightDock restraints.list.
"""

from __future__ import annotations

import os
import sys
import numpy as np
from pathlib import Path
from typing import Any

import gemmi
from Bio.PDB import PDBParser

from core.pocket_detection import detect_pockets
from core.peptide_docking.binding_site import validate_residue_identifier
from utils.io import ensure_dir
from utils.logger import setup_logger

logger = setup_logger("peptide_pocket", level="INFO")


def extract_residues_in_bounding_box(
    receptor_pdb: str | Path,
    center: list[float] | tuple[float, float, float],
    size: list[float] | tuple[float, float, float],
) -> list[str]:
    """
    Find all receptor amino acid residues whose atoms fall within a 3D bounding box.

    Returns:
        List of canonical residue identifiers (e.g. ['A.TYR.45', 'A.GLU.50'])
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("receptor", str(receptor_pdb))

    cx, cy, cz = center
    sx, sy, sz = size
    half_x, half_y, half_z = sx / 2.0, sy / 2.0, sz / 2.0

    min_x, max_x = cx - half_x, cx + half_x
    min_y, max_y = cy - half_y, cy + half_y
    min_z, max_z = cz - half_z, cz + half_z

    contact_residues = set()

    for model in struct:
        for chain in model:
            for res in chain:
                # Accept standard amino acid residues
                if res.id[0] != " ":
                    continue
                resname = res.get_resname().strip().upper()
                resnum = res.id[1]

                for atom in res:
                    x, y, z = atom.get_coord()
                    if (min_x <= x <= max_x) and (min_y <= y <= max_y) and (min_z <= z <= max_z):
                        contact_residues.add((chain.id, resname, resnum))
                        break
        break  # Use first model

    # Sort residues by chain and residue number
    sorted_res = sorted(list(contact_residues), key=lambda x: (x[0], x[2]))
    return [f"{c}.{r}.{n}" for c, r, n in sorted_res]


def convert_pocket_to_lightdock_restraints(
    receptor_pdb: str | Path,
    selected_pocket: dict[str, Any],
    output_dir: str | Path,
) -> tuple[Path, list[str]]:
    """
    Convert a selected pocket (or GUI 3D box) into validated LightDock receptor restraints.

    Returns:
        (restraints_file_path, list_of_restraints)
    """
    out_dir = Path(ensure_dir(output_dir))
    restraints_path = out_dir / "restraints.list"

    center = selected_pocket.get("center", [0.0, 0.0, 0.0])
    size = selected_pocket.get("size", [20.0, 20.0, 20.0])

    # 1. First check if pocket has residues listed
    raw_residues = selected_pocket.get("residues", [])
    valid_restraints = []

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("receptor", str(receptor_pdb))

    if raw_residues:
        for r_str in raw_residues:
            # Handle formats 'A:ALA:1429', 'A.ALA.1429', or 'A.1429'
            clean = r_str.replace(":", ".")
            try:
                _, _, _, canon = validate_residue_identifier(clean, structure)
                valid_restraints.append(canon)
            except Exception:
                pass

    # 2. If no valid restraints from pre-extracted list, scan bounding box directly
    if not valid_restraints and center and size:
        box_residues = extract_residues_in_bounding_box(receptor_pdb, center, size)
        for b_res in box_residues:
            try:
                _, _, _, canon = validate_residue_identifier(b_res, structure)
                valid_restraints.append(canon)
            except Exception:
                pass

    # Dedup and sort
    unique_restraints = sorted(list(set(valid_restraints)))

    # Write restraints.list
    with open(restraints_path, "w", encoding="utf-8") as f:
        f.write("# LightDock receptor surface restraints generated from pocket detection / 3D GUI\n")
        for r in unique_restraints:
            f.write(f"R {r}\n")

    logger.info(f"Generated {len(unique_restraints)} LightDock receptor restraint(s) -> {restraints_path}")
    return restraints_path, unique_restraints


def interactive_peptide_pocket_detection(
    receptor_pdb: str | Path,
    output_dir: str | Path,
    max_pockets: int = 3,
    max_box_dimension: float = 30.0,
    auto_confirm: bool = False,
    launch_gui: bool = False,
) -> dict[str, Any]:
    """
    Run interactive pocket detection with 3D GUI box tuner support for peptide docking.

    Displays detected pockets, allows launching the 3D graphical GUI tuner,
    and converts the chosen region into LightDock receptor restraints.
    """
    out_dir = Path(ensure_dir(output_dir))
    rec_path = Path(receptor_pdb).resolve()

    # Detect geometric cavities
    pocket_res = detect_pockets(
        protein_pdb=str(rec_path),
        output_dir=str(out_dir),
        method="geometric",
        selection_mode="best_score",
        max_pockets=max_pockets,
        max_box_dimension=max_box_dimension,
    )

    pockets = pocket_res.get("pockets", [])
    if not pockets:
        logger.warning("No geometric pockets detected; falling back to global unrestrained docking.")
        restraints_file = out_dir / "restraints.list"
        with open(restraints_file, "w", encoding="utf-8") as f:
            f.write("# Global blind docking (no surface restraints)\n")
        return {
            "mode": "global",
            "restraints_path": str(restraints_file),
            "restraints_count": 0,
            "num_restraints": 0,
            "restraints": [],
            "selected_pocket": None,
        }

    selected_pocket = pockets[0]
    selection_method = "best_score"

    if launch_gui:
        try:
            from core.pocket_gui import launch_pocket_gui
            doc = gemmi.read_structure(str(rec_path))
            gui_pocket, gui_method = launch_pocket_gui(
                pockets=pockets,
                doc=doc,
                protein_pdb=str(rec_path),
                max_box_dimension=max_box_dimension,
            )
            selected_pocket = gui_pocket
            selection_method = "graphical_gui"
        except Exception as e:
            logger.warning(f"Could not launch graphical GUI: {e}. Falling back to Pocket 1.")
            selected_pocket = pockets[0]
            selection_method = "best_score"
    elif not auto_confirm and sys.stdin.isatty():
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(
            title="Detected Receptor Binding Pockets",
            border_style="cyan",
            header_style="bold cyan",
            show_header=True,
        )
        table.add_column("Option", style="bold yellow", width=8, justify="center")
        table.add_column("Pocket ID", style="bold white", width=12)
        table.add_column("Score", justify="right", width=10)
        table.add_column("Volume (Å³)", justify="right", width=14)
        table.add_column("Center (X, Y, Z)", justify="center", width=24)
        table.add_column("Residues", justify="center", width=12)

        for i, p in enumerate(pockets[:max_pockets]):
            pid = p.get("pocket_id", i + 1)
            score = f"{p.get('score', 0.0):.2f}"
            vol = f"{p.get('volume', 0.0):.1f}"
            c = p.get("center", [0, 0, 0])
            c_str = f"({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})"
            n_res = str(len(p.get("residues", [])))
            table.add_row(f"[{i + 1}]", f"Pocket {pid}", score, vol, c_str, n_res)

        console.print()
        console.print(table)
        console.print()

        gui_opt = len(pockets[:max_pockets]) + 1
        global_opt = len(pockets[:max_pockets]) + 2

        print("Binding Site Selection Options:")
        for i, p in enumerate(pockets[:max_pockets]):
            print(f"  [{i + 1}] Select Pocket {p.get('pocket_id', i + 1)}")
        print(f"  [{gui_opt}] Graphical Interface (Interactive 3D Box Tuner in Browser)")
        print(f"  [{global_opt}] Global Blind Docking (Unrestrained Surface Sampling)")
        print()

        try:
            choice_raw = input(f"Select option [1-{global_opt}] (default: 1): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nUsing Pocket 1.")
            choice_raw = "1"

        if not choice_raw or choice_raw == "1":
            selected_pocket = pockets[0]
            selection_method = "user_selected"
        elif choice_raw.lower() in ("g", "gui", str(gui_opt)):
            # Launch 3D GUI Box Tuner
            try:
                from core.pocket_gui import launch_pocket_gui
                doc = gemmi.read_structure(str(rec_path))
                gui_pocket, gui_method = launch_pocket_gui(
                    pockets=pockets,
                    doc=doc,
                    protein_pdb=str(rec_path),
                    max_box_dimension=max_box_dimension,
                )
                selected_pocket = gui_pocket
                selection_method = "graphical_gui"
            except Exception as e:
                logger.warning(f"Could not launch graphical GUI: {e}. Falling back to Pocket 1.")
                selected_pocket = pockets[0]
                selection_method = "best_score"
        elif choice_raw.lower() in ("global", "blind", str(global_opt)):
            restraints_file = out_dir / "restraints.list"
            with open(restraints_file, "w", encoding="utf-8") as f:
                f.write("# Global blind docking\n")
            return {
                "mode": "global",
                "restraints_path": str(restraints_file),
                "restraints_count": 0,
                "num_restraints": 0,
                "restraints": [],
                "selected_pocket": None,
                "selection_method": "global",
            }
        else:
            try:
                idx = int(choice_raw) - 1
                if 0 <= idx < len(pockets):
                    selected_pocket = pockets[idx]
                    selection_method = "user_selected"
            except Exception:
                selected_pocket = pockets[0]

    # Convert chosen pocket / box to validated LightDock restraints
    restr_path, restr_list = convert_pocket_to_lightdock_restraints(
        receptor_pdb=rec_path,
        selected_pocket=selected_pocket,
        output_dir=out_dir,
    )

    return {
        "mode": "pocket_guided",
        "restraints_path": str(restr_path),
        "restraints_count": len(restr_list),
        "num_restraints": len(restr_list),
        "restraints": restr_list,
        "selected_pocket": selected_pocket,
        "selection_method": selection_method,
    }
