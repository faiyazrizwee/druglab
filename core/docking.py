"""
DrugLab Molecular Docking Module.

Performs molecular docking using AutoDock Vina. Supports both the Python
bindings (``import vina``) and a subprocess-based CLI fallback.
"""

import os
import time
import json
import logging
import subprocess
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem

from utils.logger import setup_logger, log_step, suppress_stdout_stderr, is_verbose
from utils.io import (
    ensure_dir,
    write_json,
    read_sdf,
    write_sdf,
    write_csv,
    validate_file_exists,
    list_files,
)
from utils.chemistry import get_mol_name

logger = setup_logger("docking", level="INFO")

# Detect Vina availability
_HAS_VINA_PYTHON = False
try:
    from vina import Vina
    _HAS_VINA_PYTHON = True
    logger.debug("Vina Python bindings available.")
except ImportError:
    logger.debug("Vina Python bindings not available; will use CLI fallback.")


def run_docking(
    protein_pdbqt: str,
    ligands_path: str,
    docking_box: dict | str,
    output_dir: str,
    exhaustiveness: int = 32,
    n_poses: int = 10,
    energy_range: float = 3.0,
    seed: int = 42,
    n_cpus: int = 0,
    score_only: bool = False,
    admet_csv: str = None,
) -> dict:
    """
    Run molecular docking with AutoDock Vina.

    Args:
        protein_pdbqt: Path to prepared protein PDBQT file.
        ligands_path: Path to SDF file of prepared ligands, or directory of
                      PDBQT files.
        docking_box: Dict with ``center`` [x,y,z] and ``size`` [sx,sy,sz],
                     or path to a JSON file containing the same.
        output_dir: Path to the output directory.
        exhaustiveness: Vina exhaustiveness parameter.
        n_poses: Number of docking poses to generate.
        energy_range: Maximum energy difference from best pose (kcal/mol).
        seed: Random seed for reproducibility.
        n_cpus: Number of CPUs (0 = auto-detect).
        score_only: If True, score provided poses without docking.

    Returns:
        Dict containing docking results and output file paths.
    """
    start_time = time.time()
    log_step(logger, "Molecular Docking", "started", {
        "protein": protein_pdbqt,
        "ligands": ligands_path,
        "exhaustiveness": exhaustiveness,
    })

    validate_file_exists(protein_pdbqt, "Protein PDBQT")

    # Parse docking box
    box = _parse_docking_box(docking_box)
    center = box["center"]
    size = box["size"]
    logger.info(f"Docking box: center={center}, size={size}")

    # Output directories
    dock_dir = os.path.join(output_dir, "docking")
    poses_dir = os.path.join(dock_dir, "poses")
    ensure_dir(dock_dir)
    ensure_dir(poses_dir)

    results_csv_path = os.path.join(dock_dir, "docking_results.csv")
    best_poses_sdf_path = os.path.join(dock_dir, "best_poses.sdf")
    report_path = os.path.join(dock_dir, "docking_report.json")

    # Load ligands — either from SDF (convert to PDBQT) or from PDBQT directory
    ligand_entries = _prepare_ligand_entries(ligands_path, dock_dir)
    if not ligand_entries:
        raise ValueError(
            f"No ligands found/provided for docking in path '{ligands_path}'. "
            "Please check that upstream screening stages (drug-likeness, ADMET) "
            "didn't filter out all molecules, or that the input library is not empty."
        )
    logger.info(f"Prepared {len(ligand_entries)} ligands for docking.")

    # Run docking for each ligand
    docking_results = []
    best_pose_mols = []

    for idx, entry in enumerate(ligand_entries):
        name = entry["name"]
        pdbqt_path = entry["pdbqt_path"]
        logger.info(f"Docking [{idx+1}/{len(ligand_entries)}]: {name}")

        try:
            pose_output = os.path.join(poses_dir, f"{name}_poses.pdbqt")

            result = _dock_single_ligand(
                protein_pdbqt=protein_pdbqt,
                ligand_pdbqt=pdbqt_path,
                center=center,
                size=size,
                output_path=pose_output,
                exhaustiveness=exhaustiveness,
                n_poses=n_poses,
                energy_range=energy_range,
                seed=seed,
                n_cpus=n_cpus,
                score_only=score_only,
            )

            best_affinity = result["best_affinity"]
            all_affinities = result["affinities"]

            docking_results.append({
                "name": name,
                "smiles": entry.get("smiles", ""),
                "best_affinity": round(best_affinity, 3),
                "all_affinities": [round(a, 3) for a in all_affinities],
                "n_poses": len(all_affinities),
                "pose_file": pose_output,
                "status": "completed",
            })

            logger.info(f"  → Best affinity: {best_affinity:.2f} kcal/mol "
                        f"({len(all_affinities)} poses)")

        except Exception as e:
            logger.error(f"Docking failed for {name}: {e}")
            docking_results.append({
                "name": name,
                "smiles": entry.get("smiles", ""),
                "best_affinity": 0.0,
                "all_affinities": [],
                "n_poses": 0,
                "pose_file": "",
                "status": f"failed: {e}",
            })

    # Build results CSV
    df = pd.DataFrame([{
        "name": r["name"],
        "smiles": r["smiles"],
        "best_affinity": r["best_affinity"],
        "n_poses": r["n_poses"],
        "pose_file": r["pose_file"],
        "status": r["status"],
    } for r in docking_results])
    write_csv(df, results_csv_path)

    # Create best_poses.sdf from the top-scoring PDBQT files
    _create_best_poses_sdf(docking_results, best_poses_sdf_path)

    # ── Figure Generation ──
    try:
        from reports.generator import generate_docking_figures
        plot_df = df.copy()
        if admet_csv and os.path.exists(admet_csv):
            admet_df = pd.read_csv(admet_csv)
            plot_df = plot_df.merge(admet_df, on="name", how="left")
            
        generated_figs = generate_docking_figures(plot_df, output_dir)
    except Exception as e:
        logger.error(f"Failed to generate docking figures: {e}")
        generated_figs = []

    duration = time.time() - start_time
    completed = sum(1 for r in docking_results if r["status"] == "completed")
    failed = len(docking_results) - completed

    metrics = {
        "total_ligands": len(ligand_entries),
        "completed": completed,
        "failed": failed,
        "results_csv": results_csv_path,
        "best_poses_sdf": best_poses_sdf_path,
        "poses_dir": poses_dir,
        "docking_results": docking_results,
        "generated_figures": generated_figs,
        "duration": round(duration, 2),
        "status": "completed" if completed > 0 else "failed",
    }

    write_json(metrics, report_path)
    
    log_dict = {
        "docked": completed,
        "failed": failed,
        "results_csv": results_csv_path,
        "duration": f"{duration:.1f}s",
    }
    
    if generated_figs:
        fig_msg = "\nGenerated Figures:\n"
        for f in generated_figs:
            fig_msg += f"  ✓ {os.path.basename(f)}\n"
        log_dict["figures"] = fig_msg

    log_step(logger, "Molecular Docking", "completed", log_dict)
    
    if generated_figs:
        print(f"\nGenerated {len(generated_figs)} figures:")
        print(f"{os.path.join(output_dir, 'figures', 'docking')}/")
        for f in generated_figs:
            print(f"  ✓ {os.path.basename(f)}")
        print()

    return metrics


# ──────────────────────────────────────────────────────────────
# Internal: Parse docking box
# ──────────────────────────────────────────────────────────────

def _parse_docking_box(box) -> dict:
    """
    Parse docking box from dict or JSON file path.

    Accepts three formats:
      1. JSON file path (str) → loaded and parsed as dict
      2. Dict with "center" and "size" lists:
         {"center": [x, y, z], "size": [sx, sy, sz]}
      3. Flat dict from manual entry:
         {"center_x": x, "center_y": y, "center_z": z,
          "size_x": sx, "size_y": sy, "size_z": sz}
    """
    if isinstance(box, str):
        validate_file_exists(box, "Docking box JSON")
        with open(box, "r") as f:
            box = json.load(f)

    # Convert flat coordinate format from manual mode
    if "center_x" in box and "center" not in box:
        box = {
            "center": [box["center_x"], box["center_y"], box["center_z"]],
            "size":   [box["size_x"],   box["size_y"],   box["size_z"]],
        }

    if "center" not in box or "size" not in box:
        raise ValueError(
            f"Docking box must contain 'center' and 'size' keys. Got: {list(box.keys())}"
        )
    return box


# ──────────────────────────────────────────────────────────────
# Internal: Prepare ligand entries
# ──────────────────────────────────────────────────────────────

def _prepare_ligand_entries(ligands_path: str, dock_dir: str) -> list[dict]:
    """
    Prepare ligand entries for docking. Handles:
    - SDF file → convert each molecule to PDBQT via Meeko
    - Directory of PDBQT files → use directly
    """
    entries = []
    temp_pdbqt_dir = os.path.join(dock_dir, "ligands_pdbqt_temp")

    if os.path.isfile(ligands_path):
        ext = Path(ligands_path).suffix.lower()
        if ext == ".sdf":
            ensure_dir(temp_pdbqt_dir)
            mols = read_sdf(ligands_path)
            for idx, mol in enumerate(mols):
                if mol is None:
                    continue
                name = get_mol_name(mol, idx)
                smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
                pdbqt_path = os.path.join(temp_pdbqt_dir, f"{name}.pdbqt")

                try:
                    _mol_to_pdbqt(mol, pdbqt_path)
                    entries.append({
                        "name": name,
                        "smiles": smiles,
                        "pdbqt_path": pdbqt_path,
                    })
                except Exception as e:
                    logger.warning(f"Failed to convert {name} to PDBQT: {e}")

        elif ext == ".pdbqt":
            name = Path(ligands_path).stem
            entries.append({
                "name": name,
                "smiles": "",
                "pdbqt_path": ligands_path,
            })
        else:
            raise ValueError(f"Unsupported ligand file format: {ext}")

    elif os.path.isdir(ligands_path):
        pdbqt_files = list_files(ligands_path, extensions=[".pdbqt"])
        for pf in pdbqt_files:
            name = Path(pf).stem
            entries.append({
                "name": name,
                "smiles": "",
                "pdbqt_path": pf,
            })

    else:
        raise FileNotFoundError(f"Ligands path not found: {ligands_path}")

    return entries


def _mol_to_pdbqt(mol, output_path: str) -> None:
    """Convert an RDKit Mol to PDBQT using Meeko."""
    import meeko

    mol_h = Chem.AddHs(mol)
    # Ensure 3D coordinates exist
    if mol_h.GetNumConformers() == 0:
        from rdkit.Chem import AllChem
        AllChem.EmbedMolecule(mol_h, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol_h)

    preparer = meeko.MoleculePreparation()
    mol_setups = preparer.prepare(mol_h)
    if not mol_setups:
        raise ValueError("Meeko preparation returned empty setup.")

    writer = meeko.PDBQTWriterLegacy()
    pdbqt_string, is_ok, err = writer.write_string(mol_setups[0])
    if not is_ok:
        raise ValueError(f"Meeko writer error: {err}")

    with open(output_path, "w") as f:
        f.write(pdbqt_string)


# ──────────────────────────────────────────────────────────────
# Internal: Dock a single ligand
# ──────────────────────────────────────────────────────────────

def _dock_single_ligand(
    protein_pdbqt: str,
    ligand_pdbqt: str,
    center: list[float],
    size: list[float],
    output_path: str,
    exhaustiveness: int,
    n_poses: int,
    energy_range: float,
    seed: int,
    n_cpus: int,
    score_only: bool,
) -> dict:
    """Dock a single ligand and return results."""
    if _HAS_VINA_PYTHON:
        res = _dock_vina_python(
            protein_pdbqt, ligand_pdbqt, center, size, output_path,
            exhaustiveness, n_poses, energy_range, seed, n_cpus, score_only,
        )
    else:
        res = _dock_vina_cli(
            protein_pdbqt, ligand_pdbqt, center, size, output_path,
            exhaustiveness, n_poses, energy_range, seed, n_cpus, score_only,
        )

    if not score_only and os.path.exists(output_path):
        _split_pdbqt_poses(output_path)
        
    return res


def _dock_vina_python(
    protein_pdbqt, ligand_pdbqt, center, size, output_path,
    exhaustiveness, n_poses, energy_range, seed, n_cpus, score_only,
) -> dict:
    """Dock using Vina Python bindings."""
    verbosity = 1 if is_verbose() else 0
    with suppress_stdout_stderr():
        v = Vina(sf_name="vina", cpu=n_cpus if n_cpus > 0 else 0, seed=seed, verbosity=verbosity)
        v.set_receptor(protein_pdbqt)
        v.set_ligand_from_file(ligand_pdbqt)
        v.compute_vina_maps(
            center=center,
            box_size=size,
        )

        if score_only:
            energy = v.score()
            return {"best_affinity": energy[0], "affinities": [energy[0]]}

        v.dock(
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            min_rmsd=1.0,
        )

        v.write_poses(output_path, n_poses=n_poses, overwrite=True)

    # Parse energies from poses
    affinities = _parse_pdbqt_affinities(output_path)
    best = affinities[0] if affinities else 0.0

    return {"best_affinity": best, "affinities": affinities}


def _dock_vina_cli(
    protein_pdbqt, ligand_pdbqt, center, size, output_path,
    exhaustiveness, n_poses, energy_range, seed, n_cpus, score_only,
) -> dict:
    """Dock using Vina CLI as subprocess."""
    vina_cmd = shutil.which("vina")
    if not vina_cmd:
        raise FileNotFoundError(
            "AutoDock Vina not found. Install via conda or ensure 'vina' is on PATH."
        )

    cmd = [
        vina_cmd,
        "--receptor", protein_pdbqt,
        "--ligand", ligand_pdbqt,
        "--center_x", str(center[0]),
        "--center_y", str(center[1]),
        "--center_z", str(center[2]),
        "--size_x", str(size[0]),
        "--size_y", str(size[1]),
        "--size_z", str(size[2]),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(n_poses),
        "--energy_range", str(energy_range),
        "--seed", str(seed),
        "--out", output_path,
    ]

    if n_cpus > 0:
        cmd.extend(["--cpu", str(n_cpus)])

    if score_only:
        cmd.append("--score_only")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        raise RuntimeError(f"Vina CLI failed:\n{result.stderr}")

    affinities = _parse_pdbqt_affinities(output_path)
    best = affinities[0] if affinities else 0.0

    return {"best_affinity": best, "affinities": affinities}


# ──────────────────────────────────────────────────────────────
# Internal: Parse PDBQT output
# ──────────────────────────────────────────────────────────────

def _parse_pdbqt_affinities(pdbqt_path: str) -> list[float]:
    """Extract binding affinities from a multi-model PDBQT output file."""
    affinities = []
    if not os.path.isfile(pdbqt_path):
        return affinities

    with open(pdbqt_path, "r") as f:
        for line in f:
            if line.startswith("REMARK VINA RESULT:"):
                parts = line.split()
                try:
                    affinity = float(parts[3])
                    affinities.append(affinity)
                except (IndexError, ValueError):
                    pass

    return affinities


def _split_pdbqt_poses(pdbqt_path: str) -> None:
    """
    Split a multi-model PDBQT file into individual model files within a directory.
    Also copies the multi-model file (containing all poses) as all_pose.pdbqt.
    """
    if not os.path.isfile(pdbqt_path):
        return

    # E.g. apigenin_poses.pdbqt -> apigenin_poses/
    base_dir = os.path.splitext(pdbqt_path)[0]
    ensure_dir(base_dir)

    models = []
    current_model_lines = []
    in_model = False

    with open(pdbqt_path, "r") as f:
        for line in f:
            if line.startswith("MODEL"):
                in_model = True
                current_model_lines = [line]
            elif line.startswith("ENDMDL"):
                current_model_lines.append(line)
                models.append(current_model_lines)
                in_model = False
            elif in_model:
                current_model_lines.append(line)

    for i, model_lines in enumerate(models):
        model_idx = i + 1
        model_path = os.path.join(base_dir, f"model_{model_idx}.pdbqt")
        with open(model_path, "w") as out:
            out.writelines(model_lines)
        
    # Copy the whole pdbqt_path (containing all poses) to all_pose.pdbqt in base_dir
    all_pose_path = os.path.join(base_dir, "all_pose.pdbqt")
    import shutil
    shutil.copyfile(pdbqt_path, all_pose_path)


def _create_best_poses_sdf(docking_results: list[dict], output_sdf: str) -> None:
    """
    Create a combined SDF file from the best pose of each docked ligand.

    Uses Open Babel for PDBQT → SDF conversion.
    """
    import sys

    obabel_path = shutil.which("obabel")
    if not obabel_path:
        # Try conda env
        obabel_path = os.path.join(os.path.dirname(sys.executable), "obabel")
        if not os.path.exists(obabel_path):
            logger.warning("obabel not found; skipping best_poses.sdf generation.")
            return

    temp_sdfs = []
    temp_dir = os.path.dirname(output_sdf)

    for r in docking_results:
        if r["status"] != "completed" or not r["pose_file"]:
            continue
        if not os.path.isfile(r["pose_file"]):
            continue

        temp_sdf = os.path.join(temp_dir, f"_temp_{r['name']}.sdf")
        try:
            subprocess.run(
                [obabel_path, r["pose_file"], "-O", temp_sdf, "-l", "1"],
                capture_output=True, text=True, check=True,
            )
            if os.path.isfile(temp_sdf) and os.path.getsize(temp_sdf) > 0:
                temp_sdfs.append(temp_sdf)
        except Exception as e:
            logger.debug(f"Could not convert pose for {r['name']}: {e}")

    # Concatenate SDFs
    if temp_sdfs:
        with open(output_sdf, "w") as out:
            for tsdf in temp_sdfs:
                with open(tsdf, "r") as inp:
                    out.write(inp.read())
        logger.info(f"Created best poses SDF with {len(temp_sdfs)} molecules.")

    # Cleanup
    for tsdf in temp_sdfs:
        try:
            os.remove(tsdf)
        except OSError:
            pass


def prompt_exhaustiveness_selection() -> int:
    """Interactively prompt user to select AutoDock Vina exhaustiveness."""
    print("Select Exhaustiveness:\n")
    print("[1] 8 (Fast / Default)")
    print("[2] 16 (Balanced)")
    print("[3] 32 (Thorough)")
    print("[4] 64 (Very Thorough)")
    print()
    try:
        raw = input("Selection: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nUsing 8 (Fast / Default).")
        return 8

    if not raw:
        return 8

    try:
        choice = int(raw)
    except ValueError:
        print("Invalid selection. Using 8 (Fast / Default).")
        return 8

    mapping = {1: 8, 2: 16, 3: 32, 4: 64}
    if choice not in mapping:
        print("Invalid selection. Using 8 (Fast / Default).")
        return 8

    return mapping[choice]

