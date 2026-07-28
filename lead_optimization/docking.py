"""
DrugLab Lead Optimization — Analogue Docking Wrapper.

Wraps ``core.docking.run_docking()`` to dock generated analogues
against the same protein target and docking box used for discovery.
"""

import os
from core.docking import run_docking
from utils.logger import setup_logger
from utils.io import validate_file_exists

logger = setup_logger("optimization_docking", level="INFO")


def dock_analogues(
    protein_pdbqt: str,
    analogues_sdf: str,
    docking_box: dict | str,
    output_dir: str,
    exhaustiveness: int = 32,
    n_poses: int = 10,
    energy_range: float = 3.0,
    seed: int = 42,
    n_cpus: int = 0,
) -> dict:
    """
    Dock generated analogues against the protein target.

    Results are written to a separate ``analogue_docking/`` sub-directory.

    Args:
        protein_pdbqt: Path to prepared protein PDBQT.
        analogues_sdf: Path to the analogues SDF file.
        docking_box: Docking box dict or path to docking_box.json.
        output_dir: Output directory.
        exhaustiveness: Vina exhaustiveness.
        n_poses: Number of poses per ligand.
        energy_range: Max energy range from best pose.
        seed: Random seed.
        n_cpus: Number of CPUs (0 = auto).

    Returns:
        Dict with docking results.
    """
    validate_file_exists(protein_pdbqt, "Protein PDBQT")
    validate_file_exists(analogues_sdf, "Analogues SDF")

    # Use a separate sub-directory
    dock_output = os.path.join(output_dir, "analogue_docking")
    os.makedirs(dock_output, exist_ok=True)

    return run_docking(
        protein_pdbqt=protein_pdbqt,
        ligands_path=analogues_sdf,
        docking_box=docking_box,
        output_dir=dock_output,
        exhaustiveness=exhaustiveness,
        n_poses=n_poses,
        energy_range=energy_range,
        seed=seed,
        n_cpus=n_cpus,
    )
