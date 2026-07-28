"""
DrugLab Protein Preparation Module.

Cleans raw PDB files, fixes missing residues/atoms, adds hydrogens at a specified pH,
removes water/heterogens, and converts the prepared protein to PDBQT format for docking.
"""

import os
import io
import time
import logging
from pathlib import Path
try:
    from pdbfixer import PDBFixer
    import openmm
    from openmm.app import PDBFile
except ImportError:
    # Optional dependencies not installed; protein preparation will be unavailable.
    PDBFixer = None
    openmm = None
    PDBFile = None


try:
    from openbabel import pybel
except ImportError:
    pybel = None  # OpenBabel not available; functions using it will raise if called

from utils.logger import setup_logger, log_step
from utils.io import ensure_dir, write_json

logger = setup_logger("protein_preparation", level="INFO")

# Standard residues recognized by PDBFixer
from pdbfixer.pdbfixer import proteinResidues, dnaResidues, rnaResidues
STANDARD_RESIDUES = set(proteinResidues).union(dnaResidues).union(rnaResidues).union({'N', 'UNK'})


def prepare_protein(
    input_pdb: str,
    output_dir: str,
    chain_id: str | None = None,
    remove_water: bool = True,
    remove_heterogens: bool = True,
    add_hydrogens: bool = True,
    fix_missing_residues: bool = True,
    fix_missing_atoms: bool = True,
    ph: float = 7.4,
    keep_ligands: list[str] | None = None,
) -> dict:
    """
    Cleans a raw PDB file and converts it to PDBQT format.

    Args:
        input_pdb: Path to the input raw PDB file.
        output_dir: Path to the output directory.
        chain_id: Chain ID to select (None keeps all chains).
        remove_water: If True, water molecules are removed.
        remove_heterogens: If True, non-standard residues are removed (except keep_ligands).
        add_hydrogens: If True, adds missing hydrogens at specified pH.
        fix_missing_residues: If True, rebuilds missing loops/residues.
        fix_missing_atoms: If True, rebuilds missing heavy atoms of existing residues.
        ph: pH value at which to protonate the protein.
        keep_ligands: List of 3-letter HET residue codes to keep.

    Returns:
        Dict containing preparation metadata and output file paths.
    """
    start_time = time.time()
    log_step(logger, "Protein Preparation", "started", {"input": input_pdb, "output_dir": output_dir})

    if keep_ligands is None:
        keep_ligands = []

    # Ensure output directories exist
    prep_dir = os.path.join(output_dir, "protein_preparation")
    ensure_dir(prep_dir)

    clean_pdb_path = os.path.join(prep_dir, "protein_clean.pdb")
    pdbqt_path = os.path.join(prep_dir, "protein.pdbqt")
    report_path = os.path.join(prep_dir, "preparation_report.json")

    # Metrics dictionary to collect info
    metrics = {
        "input_pdb": input_pdb,
        "clean_pdb": clean_pdb_path,
        "pdbqt": pdbqt_path,
        "removed_waters": 0,
        "removed_heterogens": [],
        "fixed_residues": [],
        "fixed_atoms": [],
        "chains": [],
        "selected_chain": chain_id,
        "num_residues": 0,
        "status": "failed",
    }

    try:
        # Load PDB
        logger.info(f"Loading PDB: {input_pdb}")
        fixer = PDBFixer(filename=input_pdb)

        # Force OpenMM to use the Reference platform (safer, robust, no OpenCL compiler hangs)
        try:
            fixer.platform = openmm.Platform.getPlatformByName("Reference")
            logger.debug("Using OpenMM Reference platform")
        except Exception as plat_err:
            logger.warning(f"Could not explicitly set Reference platform: {plat_err}")

        # Record initial chains
        initial_chains = [c.id for c in fixer.topology.chains()]
        metrics["chains"] = list(set(initial_chains))
        logger.info(f"PDB loaded. Chains found: {metrics['chains']}")

        # 1. Chain Selection
        if chain_id:
            logger.info(f"Selecting chain: {chain_id}")
            chains_list = list(fixer.topology.chains())
            to_remove = [i for i, c in enumerate(chains_list) if c.id != chain_id]
            if len(to_remove) == len(chains_list):
                raise ValueError(f"Selected chain '{chain_id}' not found in PDB. Available: {metrics['chains']}")
            fixer.removeChains(to_remove)
            logger.info(f"Removed other chains. Remaining chains: {[c.id for c in fixer.topology.chains()]}")

        # 2. Water / Heterogen Removal
        # We perform custom heterogen removal to respect keep_ligands
        keep_set = set(STANDARD_RESIDUES)
        if not remove_water:
            keep_set.add("HOH")
            keep_set.add("WAT")
        for lig in keep_ligands:
            keep_set.add(lig.upper())

        to_delete = []
        waters_count = 0
        removed_hets = []

        for residue in fixer.topology.residues():
            if residue.name not in keep_set:
                to_delete.append(residue)
                if residue.name in ("HOH", "WAT"):
                    waters_count += 1
                else:
                    removed_hets.append(residue.name)

        if to_delete:
            logger.info(f"Deleting {len(to_delete)} unwanted residues (water count: {waters_count}, heterogen codes: {list(set(removed_hets))})")
            modeller = openmm.app.Modeller(fixer.topology, fixer.positions)
            modeller.delete(to_delete)
            fixer.topology = modeller.topology
            fixer.positions = modeller.positions
            metrics["removed_waters"] = waters_count
            metrics["removed_heterogens"] = list(set(removed_hets))

        # 3. Missing Residues / Atoms
        # Find missing residues first
        fixer.findMissingResidues()
        
        # If we do not want to fix missing residues (rebuilding massive loops),
        # we clear the dict.
        if not fix_missing_residues:
            logger.info("Skipping rebuilding of completely missing residues/loops (clearing missingResidues)")
            fixer.missingResidues = {}
        else:
            if fixer.missingResidues:
                # Log the missing residues
                chains = list(fixer.topology.chains())
                missing_res_desc = [f"chain {chains[k[0]].id} index {k[1]} ({v})" for k, v in fixer.missingResidues.items()]
                logger.info(f"Found {len(fixer.missingResidues)} missing residue segments to rebuild: {missing_res_desc[:5]}...")
                metrics["fixed_residues"] = [f"{chains[k[0]].id}:{k[1]}" for k in fixer.missingResidues.keys()]

        # Find non-standard residues and replace them (e.g. MSE to MET)
        fixer.findNonstandardResidues()
        if fixer.nonstandardResidues:
            logger.info(f"Replacing non-standard residues: {fixer.nonstandardResidues}")
            fixer.replaceNonstandardResidues()

        # Find missing atoms of existing residues
        fixer.findMissingAtoms()
        if fixer.missingAtoms:
            logger.info(f"Found missing atoms in {len(fixer.missingAtoms)} residues.")
            metrics["fixed_atoms"] = [f"{r.chain.id}:{r.name}{r.id}" for r in fixer.missingAtoms.keys()]
        
        if not fix_missing_atoms:
            logger.info("Skipping missing atom rebuilding (clearing missingAtoms)")
            fixer.missingAtoms = {}
            fixer.missingTerminals = {}

        # Add the missing atoms & run minimization
        if fixer.missingAtoms or (fix_missing_residues and fixer.missingResidues):
            logger.info("Adding missing atoms and performing local energy minimization...")
            fixer.addMissingAtoms()
            logger.info("Missing atoms added successfully.")

        # 4. Add Hydrogens
        if add_hydrogens:
            logger.info(f"Adding missing hydrogens at pH {ph}...")
            fixer.addMissingHydrogens(ph)
            logger.info("Hydrogens added.")

        # Save Cleaned PDB
        logger.info(f"Writing cleaned PDB to {clean_pdb_path}")
        with open(clean_pdb_path, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)

        # Record final residue count
        metrics["num_residues"] = len(list(fixer.topology.residues()))

        # 5. Convert to PDBQT using obabel CLI (avoids python pybel hangs)
        logger.info(f"Converting cleaned PDB to PDBQT format at {pdbqt_path}...")
        import sys
        import subprocess
        obabel_dir = os.path.dirname(sys.executable)
        obabel_path = os.path.join(obabel_dir, "obabel")
        if not os.path.exists(obabel_path):
            obabel_path = "obabel"  # fall back to system PATH
        
        logger.debug(f"Running obabel command: {obabel_path} {clean_pdb_path} -O {pdbqt_path} -xr")
        subprocess.run(
            [obabel_path, clean_pdb_path, "-O", pdbqt_path, "-xr"],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info("PDBQT conversion completed successfully.")

        duration = time.time() - start_time
        metrics["status"] = "completed"
        metrics["duration"] = round(duration, 2)
        write_json(metrics, report_path)
        log_step(logger, "Protein Preparation", "completed", {
            "clean_pdb": clean_pdb_path,
            "pdbqt": pdbqt_path,
            "residues": metrics["num_residues"],
            "duration": f"{duration:.1f}s"
        })
        return metrics

    except Exception as e:
        duration = time.time() - start_time
        metrics["status"] = "failed"
        metrics["error"] = str(e)
        metrics["duration"] = round(duration, 2)
        write_json(metrics, report_path)
        log_step(logger, "Protein Preparation", "failed", {"error": str(e)})
        logger.exception("Protein preparation failed")
        raise
