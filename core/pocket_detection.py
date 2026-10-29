"""
DrugLab Pocket Detection Module.

Detects binding pockets on prepared proteins using geometric grid-based cavity
detection or co-crystallized ligand coordinates. Defines the AutoDock Vina
docking box with safeguards against unrealistically large boxes.

Features:
    - Box dimension clamping (max_box_dimension, default 30 Å)
    - Multiple pocket selection modes (best_score, largest_volume, ligand_guided, manual)
    - Interactive pocket selection for manual/automatic workflows
    - Reduced default padding (5 Å) for realistic docking boxes
"""

import os
import time
import json
import logging
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import gemmi

from utils.logger import setup_logger, log_step
from utils.io import ensure_dir, write_json, validate_file_exists

logger = setup_logger("pocket_detection", level="INFO")


def detect_pockets(
    protein_pdb: str,
    output_dir: str,
    method: str = "geometric",
    selection_mode: str = "best_score",
    reference_ligand: str | None = None,
    min_pocket_size: float = 100.0,
    max_pockets: int = 5,
    padding: float = 5.0,
    max_box_dimension: float = 30.0,
    warn_large_boxes: bool = True,
    manual_center: tuple[float, float, float] | None = None,
    manual_size: tuple[float, float, float] | None = None,
    interactive: bool = False,
) -> dict:
    """
    Detects pockets on the protein and calculates a docking box.

    Args:
        protein_pdb: Path to the cleaned protein PDB file.
        output_dir: Path to the output directory.
        method: "geometric", "ligand_based", or "manual".
        selection_mode: "best_score", "largest_volume", "ligand_guided", "manual".
        reference_ligand: Path to reference ligand file (SDF/PDB) for ligand-based mode.
        min_pocket_size: Minimum volume in cubic Angstroms for geometric detection.
        max_pockets: Maximum number of pockets to return in geometric mode.
        padding: Padding in Angstroms to add around the pocket for the docking box.
        max_box_dimension: Maximum allowed box dimension per axis (Å). Clamp if exceeded.
        warn_large_boxes: Log warnings when box dimensions are clamped.
        manual_center: (x, y, z) coordinates for manual box center.
        manual_size: (sx, sy, sz) dimensions for manual box size.
        interactive: If True, prompt the user to select a pocket.

    Returns:
        Dict containing pocket definitions, best pocket, docking box, and output paths.
    """
    start_time = time.time()
    log_step(logger, "Pocket Detection", "started", {"protein": protein_pdb, "method": method})

    validate_file_exists(protein_pdb, "Protein PDB")

    # Ensure output directories exist
    pocket_dir = os.path.join(output_dir, "pocket_detection")
    ensure_dir(pocket_dir)

    pockets_json_path = os.path.join(pocket_dir, "pockets.json")
    best_pocket_json_path = os.path.join(pocket_dir, "best_pocket.json")
    selected_pocket_json_path = os.path.join(pocket_dir, "selected_pocket.json")
    manual_pocket_json_path = os.path.join(pocket_dir, "manual_pocket.json")
    docking_box_json_path = os.path.join(pocket_dir, "docking_box.json")
    pocket_pdb_path = os.path.join(pocket_dir, "pocket_residues.pdb")
    report_path = os.path.join(pocket_dir, "pocket_report.json")

    # Load structure with gemmi
    doc = gemmi.read_structure(protein_pdb)

    pockets = []
    best_pocket = None
    selected_pocket = None
    selection_method = "best_score"
    docking_box = None
    box_warnings = []

    if method == "manual":
        if manual_center is None or manual_size is None:
            raise ValueError("manual_center and manual_size must be specified in 'manual' mode.")
        logger.info(f"Manual pocket mode: center={manual_center}, size={manual_size}")

        # Clamp manual size (allow user to specify up to 60.0 Å for custom pockets)
        manual_max = max(max_box_dimension, 60.0)
        clamped_size, warnings = _clamp_box_size(
            list(manual_size), manual_max, warn_large_boxes
        )
        box_warnings.extend(warnings)

        residues = extract_pocket_residues(doc, np.array(manual_center), radius=12.0)

        pocket = {
            "pocket_id": 0,
            "center": list(manual_center),
            "size": clamped_size,
            "volume": float(np.prod(clamped_size)),
            "residues": residues,
            "score": 100.0
        }
        pockets = [pocket]
        best_pocket = pocket
        selected_pocket = pocket
        selection_method = "manual"
        docking_box = {
            "center": list(manual_center),
            "size": clamped_size
        }
        # Write manual_pocket.json
        write_json(pocket, manual_pocket_json_path)

    elif method == "ligand_based":
        if not reference_ligand:
            raise ValueError("reference_ligand must be specified in 'ligand_based' mode.")
        logger.info(f"Ligand-based pocket mode using reference: {reference_ligand}")
        validate_file_exists(reference_ligand, "Reference ligand")

        lig_coords = load_ligand_coords(reference_ligand)
        center = np.mean(lig_coords, axis=0)
        min_l = np.min(lig_coords, axis=0)
        max_l = np.max(lig_coords, axis=0)

        size = max_l - min_l + 2.0 * padding
        size = np.maximum(size, [15.0, 15.0, 15.0])

        # Clamp
        size_list = [round(float(x), 3) for x in size]
        clamped_size, warnings = _clamp_box_size(size_list, max_box_dimension, warn_large_boxes)
        box_warnings.extend(warnings)

        residues = extract_pocket_residues(doc, center, radius=10.0)

        pocket = {
            "pocket_id": 0,
            "center": [round(float(x), 3) for x in center],
            "size": clamped_size,
            "volume": float(round(np.prod(clamped_size), 2)),
            "residues": residues,
            "score": 100.0
        }
        pockets = [pocket]
        best_pocket = pocket
        selected_pocket = pocket
        selection_method = "best_score"
        docking_box = {
            "center": pocket["center"],
            "size": pocket["size"]
        }

    elif method == "geometric":
        logger.info("Geometric pocket mode: performing grid-based cavity clustering...")
        pockets = geometric_pocket_detection(
            doc, min_pocket_size, max_pockets, padding, max_box_dimension, warn_large_boxes
        )

        if not pockets:
            logger.warning("No pockets detected. Falling back to center-of-mass default.")
            atoms_pos = extract_protein_atoms(doc)
            center = np.mean(atoms_pos, axis=0)
            size = [20.0, 20.0, 20.0]
            residues = extract_pocket_residues(doc, center, radius=10.0)
            pocket = {
                "pocket_id": 0,
                "center": [round(x, 3) for x in center],
                "size": list(size),
                "volume": 8000.0,
                "residues": residues,
                "score": 0.0
            }
            pockets = [pocket]

        # Apply selection mode (non-interactive default)
        best_pocket = _select_pocket(
            pockets, selection_mode, reference_ligand, interactive=False
        )
        selected_pocket = best_pocket
        selection_method = "best_score"

        if interactive:
            selected_pocket, selection_method = prompt_pocket_confirmation_with_selection(
                pockets, max_box_dimension=max_box_dimension, doc=doc, protein_pdb=protein_pdb
            )

        docking_box = {
            "center": selected_pocket["center"],
            "size": selected_pocket["size"]
        }
    else:
        raise ValueError(f"Unknown pocket detection method: {method}")

    # Collect any box warnings from the selected pocket
    for pocket in pockets:
        _, pw = _clamp_box_size(pocket["size"], max_box_dimension, warn_large_boxes=False)
        if pw:
            box_warnings.append(f"Pocket {pocket.get('pocket_id', '?')}: original size would exceed {max_box_dimension} Å")

    # Write pocket residues PDB
    logger.info(f"Extracting pocket residues ({len(selected_pocket['residues'])} residues)...")
    write_pocket_pdb(protein_pdb, pocket_pdb_path, selected_pocket["residues"])

    # Write output JSON files
    write_json(pockets, pockets_json_path)
    write_json(best_pocket, best_pocket_json_path)
    write_json(selected_pocket, selected_pocket_json_path)
    write_json(docking_box, docking_box_json_path)

    duration = time.time() - start_time
    metrics = {
        "method": method,
        "selection_mode": selection_mode,
        "selection_method": selection_method,
        "max_box_dimension": max_box_dimension,
        "pockets": pockets,
        "best_pocket": best_pocket,
        "selected_pocket": selected_pocket,
        "docking_box": docking_box,
        "pocket_pdb": pocket_pdb_path,
        "box_warnings": box_warnings,
        "duration": round(duration, 2),
        "status": "completed"
    }
    write_json(metrics, report_path)

    log_step(logger, "Pocket Detection", "completed", {
        "selection_mode": selection_mode,
        "selection_method": selection_method,
        "selected_center": selected_pocket["center"],
        "box_size": selected_pocket["size"],
        "residues_count": len(selected_pocket["residues"]),
        "warnings": len(box_warnings),
        "duration": f"{duration:.1f}s"
    })

    return metrics


# ──────────────────────────────────────────────────────────────
# Box Size Safeguards
# ──────────────────────────────────────────────────────────────

def _clamp_box_size(
    size: list[float],
    max_dim: float,
    warn_large_boxes: bool = True,
) -> tuple[list[float], list[str]]:
    """
    Clamp box dimensions to max_dim per axis.

    Returns:
        Tuple of (clamped_size, list of warning messages).
    """
    warnings = []
    clamped = []

    for i, dim in enumerate(size):
        axis = ["X", "Y", "Z"][i] if i < 3 else str(i)
        if dim > max_dim:
            if warn_large_boxes:
                msg = (f"Docking box {axis}-axis ({dim:.1f} Å) exceeds max_box_dimension "
                       f"({max_dim:.1f} Å). Clamped to {max_dim:.1f} Å.")
                warnings.append(msg)
                logger.warning(msg)
            clamped.append(round(max_dim, 3))
        else:
            clamped.append(round(dim, 3))

    return clamped, warnings


# ──────────────────────────────────────────────────────────────
# Pocket Selection
# ──────────────────────────────────────────────────────────────

def _select_pocket(
    pockets: list[dict],
    mode: str = "best_score",
    reference_ligand: str | None = None,
    interactive: bool = False,
) -> dict:
    """
    Select a pocket based on the specified mode (non-interactive).

    Args:
        pockets: List of detected pockets.
        mode: Selection mode.
        reference_ligand: Reference ligand path for 'ligand_guided' mode.
        interactive: Ignored (kept for backward compatibility).

    Returns:
        Selected pocket dict.
    """
    if not pockets:
        raise ValueError("No pockets available for selection.")

    if len(pockets) == 1:
        return pockets[0]

    if mode == "best_score":
        # Already sorted by score descending from detection
        return pockets[0]

    elif mode == "largest_volume":
        return max(pockets, key=lambda p: p.get("volume", 0))

    elif mode == "ligand_guided" and reference_ligand:
        lig_coords = load_ligand_coords(reference_ligand)
        lig_center = np.mean(lig_coords, axis=0)
        # Find pocket closest to ligand center
        best = min(pockets, key=lambda p: np.linalg.norm(
            np.array(p["center"]) - lig_center
        ))
        return best

    else:
        return pockets[0]


# ──────────────────────────────────────────────────────────────
# Interactive Pocket Selection (New)
# ──────────────────────────────────────────────────────────────

def display_top_pockets(pockets: list[dict], n: int = 3) -> None:
    """
    Display the top N detected pockets in a rich, formatted layout.

    Shows for each pocket: ID, Score, Volume, Residues count,
    Center coordinates (X/Y/Z), and Docking Box dimensions (X/Y/Z).
    """
    display_pockets = pockets[:n]

    print()
    print("=" * 60)
    print("Detected Pockets")
    print("=" * 16)
    print()

    for i, p in enumerate(display_pockets):
        pid = p.get("pocket_id", i)
        score = p.get("score", 0)
        vol = p.get("volume", 0)
        n_res = len(p.get("residues", []))
        center = p.get("center", [0, 0, 0])
        size = p.get("size", [0, 0, 0])

        print(f"[{pid + 1}] Pocket {pid + 1}")
        print(f"    Score: {score:.1f}")
        print(f"    Volume: {vol:.0f} \u00c5\u00b3")
        print(f"    Residues: {n_res}")
        print()
        print(f"    Center:")
        print(f"      X = {center[0]:.3f}")
        print(f"      Y = {center[1]:.3f}")
        print(f"      Z = {center[2]:.3f}")
        print()
        print(f"    Size:")
        print(f"      X = {size[0]:.1f}")
        print(f"      Y = {size[1]:.1f}")
        print(f"      Z = {size[2]:.1f}")
        print()

        if i < len(display_pockets) - 1:
            print("    ---")
            print()

    print("=" * 60)


def _collect_manual_pocket(
    doc,
    max_box_dimension: float = 30.0,
    warn_large_boxes: bool = True,
) -> dict:
    """
    Prompt the user for manual docking box coordinates.

    Validates:
      - Coordinates are numeric
      - Sizes are positive
      - Sizes do not exceed max_box_dimension (clamped with warning)

    Returns:
        Pocket dict with center, size, volume, residues, score.
    """
    print()
    print("Define Pocket Manually")
    print("=" * 22)
    print()

    fields_center = [
        ("Center X", "center_x"),
        ("Center Y", "center_y"),
        ("Center Z", "center_z"),
    ]
    fields_size = [
        ("Size X", "size_x"),
        ("Size Y", "size_y"),
        ("Size Z", "size_z"),
    ]

    coords = {}
    for label, key in fields_center:
        while True:
            print(f"{label}: ", end="", flush=True)
            try:
                raw = input().strip()
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return None
            try:
                coords[key] = float(raw)
                break
            except ValueError:
                print("  [!] Please enter a valid number.")

    print()
    for label, key in fields_size:
        while True:
            print(f"{label}: ", end="", flush=True)
            try:
                raw = input().strip()
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return None
            try:
                val = float(raw)
                if val <= 0:
                    print("  [!] Size must be positive.")
                    continue
                coords[key] = val
                break
            except ValueError:
                print("  [!] Please enter a valid number.")

    center = [coords["center_x"], coords["center_y"], coords["center_z"]]
    size = [coords["size_x"], coords["size_y"], coords["size_z"]]

    # Clamp to max_box_dimension (allow user to specify up to 60.0 Å for custom pockets)
    manual_max = max(max_box_dimension, 60.0)
    clamped_size, warnings = _clamp_box_size(size, manual_max, warn_large_boxes)
    for w in warnings:
        print(f"  [!] {w}")

    # Extract residues near the manual center
    residues = []
    if doc is not None:
        residues = extract_pocket_residues(doc, np.array(center), radius=12.0)

    pocket = {
        "pocket_id": -1,
        "center": [round(c, 3) for c in center],
        "size": clamped_size,
        "volume": float(round(np.prod(clamped_size), 2)),
        "residues": residues,
        "score": 0.0,
    }
    return pocket


def interactive_pocket_selection(
    pockets: list[dict],
    max_box_dimension: float = 30.0,
    doc=None,
    protein_pdb: str = None,
) -> tuple[dict, str]:
    """
    Display top 3 detected pockets and prompt the user to select one,
    launch the interactive 3D graphical tuner, or define a pocket manually.

    Args:
        pockets: List of detected pockets (sorted by score descending).
        max_box_dimension: Maximum allowed box dimension per axis.
        doc: gemmi Structure object (needed for manual pocket residue extraction).
        protein_pdb: Path to the cleaned protein PDB file (for 3D viewer).

    Returns:
        Tuple of (selected_pocket, selection_method) where selection_method
        is one of: "best_score", "user_selected", "manual", "graphical_gui".
    """
    display_pockets = pockets[:3]
    display_top_pockets(pockets, n=3)

    gui_option = len(display_pockets) + 1
    manual_option = len(display_pockets) + 2

    print()
    print("Select Pocket:")
    print()
    for i, p in enumerate(display_pockets):
        pid = p.get("pocket_id", i)
        print(f"  [{pid + 1}] Use Pocket {pid + 1}")
    print(f"  [{gui_option}] Graphical Interface (Interactive 3D Box Tuner)")
    print(f"  [{manual_option}] Define Pocket Manually (CLI Text Input)")
    print()
    print("Selection [1]: ", end="", flush=True)

    try:
        raw = input().strip()
    except (EOFError, KeyboardInterrupt):
        print("\nUsing Pocket 1.")
        return pockets[0], "best_score"

    if not raw:
        return pockets[0], "best_score"

    if raw.lower() in ("g", "gui", "graphical"):
        from core.pocket_gui import launch_pocket_gui
        return launch_pocket_gui(pockets, doc, protein_pdb, max_box_dimension)

    if raw.lower() in ("m", "manual"):
        manual = _collect_manual_pocket(doc, max_box_dimension)
        if manual is None:
            print("Manual entry cancelled. Using Pocket 1.")
            return pockets[0], "best_score"
        return manual, "manual"

    try:
        choice = int(raw)
    except ValueError:
        print("Invalid selection. Using Pocket 1.")
        return pockets[0], "best_score"

    # Graphical GUI option
    if choice == gui_option:
        from core.pocket_gui import launch_pocket_gui
        return launch_pocket_gui(pockets, doc, protein_pdb, max_box_dimension)

    # Manual pocket entry
    if choice == manual_option:
        manual = _collect_manual_pocket(doc, max_box_dimension)
        if manual is None:
            print("Manual entry cancelled. Using Pocket 1.")
            return pockets[0], "best_score"
        return manual, "manual"

    # Pocket selection by index
    idx = choice - 1
    if 0 <= idx < len(display_pockets):
        selected = display_pockets[idx]
        method = "best_score" if idx == 0 else "user_selected"
        print(f"\nSelected Pocket {idx + 1} (score={selected.get('score', 0):.1f})")
        return selected, method

    print("Invalid selection. Using Pocket 1.")
    return pockets[0], "best_score"


def prompt_pocket_confirmation_with_selection(
    pockets: list[dict],
    max_box_dimension: float = 30.0,
    doc=None,
    protein_pdb: str = None,
) -> tuple[dict, str]:
    """
    For automatic/larger workflows: display top 3 pockets, then ask
    if the user wants the best-ranked pocket, graphical tuner, or list selection.

    Args:
        pockets: List of detected pockets (sorted by score descending).
        max_box_dimension: Maximum allowed box dimension per axis.
        doc: gemmi Structure object.
        protein_pdb: Path to the cleaned protein PDB file.

    Returns:
        Tuple of (selected_pocket, selection_method).
    """
    display_top_pockets(pockets, n=3)

    print()
    print("Use best-ranked pocket?")
    print()
    print("  [Y] Yes")
    print("  [N] No (Select from list or manual entry)")
    print("  [G] Graphical Interface (Interactive 3D Box Tuner)")
    print()
    print("Choice [Y/n/g]: ", end="", flush=True)

    try:
        answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nUsing best-ranked pocket.")
        return pockets[0], "best_score"

    if answer in ("g", "gui", "graphical"):
        from core.pocket_gui import launch_pocket_gui
        return launch_pocket_gui(pockets, doc, protein_pdb, max_box_dimension)

    if answer in ("n", "no"):
        # Show full selection menu
        return _pocket_selection_menu(pockets, max_box_dimension, doc, protein_pdb)

    # Accept best pocket
    return pockets[0], "best_score"


def _pocket_selection_menu(
    pockets: list[dict],
    max_box_dimension: float = 30.0,
    doc=None,
    protein_pdb: str = None,
) -> tuple[dict, str]:
    """
    Show the pocket selection menu (without re-displaying pocket details).
    Used after the user declines the best-ranked pocket in a pipeline workflow.
    """
    display_pockets = pockets[:3]
    gui_option = len(display_pockets) + 1
    manual_option = len(display_pockets) + 2

    print()
    print("Select Pocket:")
    print()
    for i, p in enumerate(display_pockets):
        pid = p.get("pocket_id", i)
        print(f"  [{pid + 1}] Pocket {pid + 1}")
    print(f"  [{gui_option}] Graphical Interface (Interactive 3D Box Tuner)")
    print(f"  [{manual_option}] Define Pocket Manually (CLI Text Input)")
    print()
    print("Selection [1]: ", end="", flush=True)

    try:
        raw = input().strip()
    except (EOFError, KeyboardInterrupt):
        print("\nUsing Pocket 1.")
        return pockets[0], "best_score"

    if not raw:
        return pockets[0], "best_score"

    if raw.lower() in ("g", "gui", "graphical"):
        from core.pocket_gui import launch_pocket_gui
        return launch_pocket_gui(pockets, doc, protein_pdb, max_box_dimension)

    if raw.lower() in ("m", "manual"):
        manual = _collect_manual_pocket(doc, max_box_dimension)
        if manual is None:
            print("Manual entry cancelled. Using Pocket 1.")
            return pockets[0], "best_score"
        return manual, "manual"

    try:
        choice = int(raw)
    except ValueError:
        print("Invalid selection. Using Pocket 1.")
        return pockets[0], "best_score"

    if choice == gui_option:
        from core.pocket_gui import launch_pocket_gui
        return launch_pocket_gui(pockets, doc, protein_pdb, max_box_dimension)

    if choice == manual_option:
        manual = _collect_manual_pocket(doc, max_box_dimension)
        if manual is None:
            print("Manual entry cancelled. Using Pocket 1.")
            return pockets[0], "best_score"
        return manual, "manual"

    idx = choice - 1
    if 0 <= idx < len(display_pockets):
        selected = display_pockets[idx]
        method = "best_score" if idx == 0 else "user_selected"
        print(f"\nSelected Pocket {idx + 1} (score={selected.get('score', 0):.1f})")
        return selected, method

    print("Invalid selection. Using Pocket 1.")
    return pockets[0], "best_score"


def display_selected_pocket_summary(pocket: dict, selection_method: str) -> None:
    """
    Print a summary of the selected docking pocket before docking begins.

    Gives the user confidence that the intended pocket is being used.
    """
    method_labels = {
        "best_score": "Best Score (Auto)",
        "user_selected": "User Selected",
        "manual": "Manual",
        "graphical_gui": "Graphical Interface (3D Tuner)",
    }
    method_label = method_labels.get(selection_method, selection_method)

    pocket_id = pocket.get("pocket_id", "?")
    if isinstance(pocket_id, int) and pocket_id >= 0:
        pocket_id_display = pocket_id + 1
    elif selection_method == "graphical_gui":
        pocket_id_display = "Custom (Graphical Tuner)"
    else:
        pocket_id_display = "Manual"

    center = pocket.get("center", [0, 0, 0])
    size = pocket.get("size", [0, 0, 0])

    print()
    print("=" * 60)
    print("Selected Docking Pocket")
    print("=" * 23)
    print()
    print(f"  Pocket ID: {pocket_id_display}")
    print(f"  Selection Method: {method_label}")
    print()
    print(f"  Center:")
    print(f"    X = {center[0]:.3f}")
    print(f"    Y = {center[1]:.3f}")
    print(f"    Z = {center[2]:.3f}")
    print()
    print(f"  Size:")
    print(f"    X = {size[0]:.1f}")
    print(f"    Y = {size[1]:.1f}")
    print(f"    Z = {size[2]:.1f}")
    print()
    print("=" * 60)
    print()


# ──────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────

def extract_protein_atoms(doc) -> np.ndarray:
    """Extract coordinates of all non-hydrogen protein atoms."""
    coords = []
    for model in doc:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element.name != 'H':
                        coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
    return np.array(coords)


def load_ligand_coords(filepath: str) -> np.ndarray:
    """Load reference ligand atom coordinates from PDB or SDF file."""
    if filepath.endswith(".sdf"):
        from rdkit import Chem
        suppl = Chem.SDMolSupplier(filepath, removeHs=True)
        if not suppl or len(suppl) == 0 or suppl[0] is None:
            raise ValueError(f"Could not read ligand from SDF: {filepath}")
        mol = suppl[0]
        return mol.GetConformer().GetPositions()
    elif filepath.endswith((".pdb", ".pdbqt")):
        doc = gemmi.read_structure(filepath)
        coords = []
        for model in doc:
            for chain in model:
                for res in chain:
                    for atom in res:
                        coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
        if not coords:
            raise ValueError(f"No atoms found in reference ligand PDB/PDBQT: {filepath}")
        return np.array(coords)
    else:
        raise ValueError(f"Unsupported reference ligand format (must be SDF, PDB, or PDBQT): {filepath}")


def extract_pocket_residues(doc, center: np.ndarray, radius: float = 10.0) -> list[str]:
    """Extract residues within a given radius of a center point."""
    pocket_res = set()
    center = np.array(center)
    for model in doc:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    pos = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                    if np.linalg.norm(pos - center) <= radius:
                        res_id = f"{chain.name}:{residue.name}:{residue.seqid.num}"
                        pocket_res.add(res_id)
                        break
    return sorted(list(pocket_res))


def write_pocket_pdb(input_pdb: str, output_pdb: str, pocket_residues: list[str]) -> None:
    """Extract lines matching pocket residues and write to a new PDB file."""
    res_tuples = set()
    for r in pocket_residues:
        parts = r.split(":")
        if len(parts) >= 3:
            res_tuples.add((parts[0], parts[2].strip()))

    with open(input_pdb, "r") as fin, open(output_pdb, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")):
                chain = line[21].strip()
                res_num = line[22:26].strip()
                if (chain, res_num) in res_tuples:
                    fout.write(line)
            elif line.startswith("CONECT") or line.startswith("MASTER") or line.startswith("END"):
                fout.write(line)


def geometric_pocket_detection(
    doc,
    min_pocket_size: float,
    max_pockets: int,
    padding: float,
    max_box_dimension: float = 30.0,
    warn_large_boxes: bool = True,
) -> list[dict]:
    """
    Find pockets using a grid-based cavity detection algorithm.
    Clusters solvent-accessible, buried grid points using DBSCAN.
    """
    atoms_pos = extract_protein_atoms(doc)
    if len(atoms_pos) == 0:
        return []

    # 1. Bounding Box
    min_coords = np.min(atoms_pos, axis=0)
    max_coords = np.max(atoms_pos, axis=0)

    # 2. Grid Generation (2.0 Å spacing)
    grid_x = np.arange(min_coords[0], max_coords[0], 2.0)
    grid_y = np.arange(min_coords[1], max_coords[1], 2.0)
    grid_z = np.arange(min_coords[2], max_coords[2], 2.0)

    gx, gy, gz = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    grid_points = np.vstack([gx.ravel(), gy.ravel(), gz.ravel()]).T

    logger.debug(f"Generated grid of {len(grid_points)} points")

    # Build KDTree for quick distance lookup
    tree = KDTree(atoms_pos)

    # 3. Filter grid points (d >= 2.6 Å to be solvent-accessible cavity, and d <= 6.0 Å to be near surface)
    distances, _ = tree.query(grid_points, k=1)
    valid_mask = (distances >= 2.6) & (distances <= 6.0)
    candidates = grid_points[valid_mask]

    logger.debug(f"Filtered to {len(candidates)} cavity candidate points")
    if len(candidates) == 0:
        return []

    # 4. Filter by buriedness (at least 14 protein atoms within 8.0 Å)
    num_neighbors = np.array([len(x) for x in tree.query_ball_point(candidates, r=8.0)])
    buried_mask = num_neighbors >= 14
    cavity_points = candidates[buried_mask]
    cavity_neighbors = num_neighbors[buried_mask]

    logger.debug(f"Identified {len(cavity_points)} buried cavity points")
    if len(cavity_points) == 0:
        return []

    # 5. Cluster cavity points with DBSCAN (eps=3.5 Å, min_samples=5)
    db = DBSCAN(eps=3.5, min_samples=5).fit(cavity_points)
    labels = db.labels_

    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise

    detected_pockets = []
    for label_id in unique_labels:
        cluster_mask = labels == label_id
        cluster_points = cavity_points[cluster_mask]

        # Calculate pocket properties
        center = np.mean(cluster_points, axis=0)
        min_p = np.min(cluster_points, axis=0)
        max_p = np.max(cluster_points, axis=0)

        # Pocket volume (grid spacing cubed times points count)
        volume = len(cluster_points) * (2.0 ** 3)
        if volume < min_pocket_size:
            continue

        # Calculate box size with padding
        size = max_p - min_p + 2.0 * padding
        # Enforce minimum size of 15.0 Å
        size = np.maximum(size, [15.0, 15.0, 15.0])

        # Clamp to max_box_dimension
        size_list = [round(float(x), 3) for x in size]
        clamped_size, _ = _clamp_box_size(size_list, max_box_dimension, warn_large_boxes)

        # Calculate buriedness score
        avg_neighbors = np.mean(cavity_neighbors[cluster_mask])
        score = len(cluster_points) * avg_neighbors

        # Extract residues lining the pocket
        residues = extract_pocket_residues(doc, center, radius=10.0)

        detected_pockets.append({
            "center": [round(float(x), 3) for x in center],
            "size": clamped_size,
            "volume": round(float(volume), 2),
            "residues": residues,
            "score": round(float(score), 2)
        })

    # Sort pockets by score (druggability) descending
    detected_pockets = sorted(detected_pockets, key=lambda p: p["score"], reverse=True)

    # Assign IDs and cap count
    final_pockets = []
    for idx, p in enumerate(detected_pockets[:max_pockets]):
        p["pocket_id"] = idx
        final_pockets.append(p)

    logger.info(f"Detected {len(final_pockets)} pockets passing volume criteria")
    return final_pockets
