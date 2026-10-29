"""
DrugLab Ligand Preparation Module.

Loads ligands from SDF/SMILES, standardizes them (removes salts, handles tautomers),
generates 3D coordinates and conformers, computes properties, and converts to PDBQT
format using Meeko for docking.
"""

import os
import time
import logging
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
import meeko

from utils.logger import setup_logger, log_step
from utils.io import (
    ensure_dir,
    write_json,
    read_sdf,
    write_sdf,
    list_files,
    validate_file_exists,
)
from utils.chemistry import (
    sanitize_mol,
    remove_salts,
    calculate_basic_properties,
    get_mol_name,
    set_mol_name,
    embed_conformer_robust,
)
from utils.quarantine import quarantine_item

logger = setup_logger("ligand_preparation", level="INFO")


def prepare_ligands(
    input_path: str,
    output_dir: str,
    add_hydrogens: bool = True,
    generate_3d: bool = True,
    num_conformers: int = 1,
    energy_minimize: bool = True,
    ph: float = 7.4,
    max_tautomers: int = 1,
    convert_to_pdbqt: bool = True,
) -> dict:
    """
    Prepares a library of ligands for molecular docking.

    Args:
        input_path: Path to SDF, directory of SDFs, or SMILES file (.smi, .txt).
        output_dir: Path to the output directory.
        add_hydrogens: If True, adds explicit hydrogens.
        generate_3d: If True, embeds the molecule in 3D.
        num_conformers: Number of 3D conformers to generate per ligand/tautomer.
        energy_minimize: If True, performs force field energy minimization.
        ph: pH value for protonation (currently standard Chem.AddHs is used).
        max_tautomers: Maximum number of tautomers to generate per ligand.
        convert_to_pdbqt: If True, converts prepared ligands to PDBQT format.

    Returns:
        Dict containing preparation report and output file paths.
    """
    start_time = time.time()
    log_step(logger, "Ligand Preparation", "started", {"input": input_path, "output_dir": output_dir})

    # Output paths
    prep_dir = os.path.join(output_dir, "ligand_preparation")
    pdbqt_dir = os.path.join(prep_dir, "ligands_pdbqt")
    ensure_dir(prep_dir)
    if convert_to_pdbqt:
        ensure_dir(pdbqt_dir)

    combined_sdf_path = os.path.join(prep_dir, "prepared_ligands.sdf")
    report_path = os.path.join(prep_dir, "preparation_report.json")

    # Load initial ligands
    raw_mols = load_ligands(input_path)
    logger.info(f"Loaded {len(raw_mols)} raw molecules from input.")

    prepared_mols = []
    failed_ligands = []
    prepared_records = []

    for idx, mol in enumerate(raw_mols):
        if mol is None:
            continue
        
        orig_name = get_mol_name(mol, idx)
        from utils.chemistry import sanitize_molecule_name, get_mol_cid
        name = sanitize_molecule_name(orig_name)
        cid = get_mol_cid(mol)
        logger.info(f"Processing ligand [{idx+1}/{len(raw_mols)}]: {orig_name} -> {name} (CID: {cid})")

        try:
            # 1. Clean and remove salts
            mol = remove_salts(mol)
            mol = sanitize_mol(mol)
            if mol is None:
                raise ValueError("Sanitization failed after salt removal.")

            # 2. Tautomer generation
            if max_tautomers > 1:
                try:
                    enumerator = rdMolStandardize.TautomerEnumerator()
                    tautomers = list(enumerator.Enumerate(mol))
                    tautomers = tautomers[:max_tautomers]
                except Exception as taut_err:
                    logger.warning(f"Tautomer enumeration failed for {name}: {taut_err}. Using original structure.")
                    tautomers = [mol]
            else:
                tautomers = [mol]

            # 3. Process each tautomer (add Hs, embed 3D, minimize)
            for t_idx, t_mol in enumerate(tautomers):
                t_name = f"{name}_t{t_idx}" if len(tautomers) > 1 else name
                set_mol_name(t_mol, t_name)

                # Add hydrogens
                if add_hydrogens:
                    t_mol = Chem.AddHs(t_mol)

                # Generate 3D coordinates
                if generate_3d:
                    t_mol, embed_ok = embed_conformer_robust(
                        t_mol,
                        num_confs=num_conformers,
                        random_seed=42,
                        energy_minimize=energy_minimize,
                    )
                    if not embed_ok:
                        logger.warning(f"3D coordinate generation used 2D fallback for {t_name}")

                # Calculate basic descriptors
                props = calculate_basic_properties(t_mol)
                for pk, pv in props.items():
                    t_mol.SetProp(pk, str(pv))

                prepared_mols.append(t_mol)

                # 4. Convert to PDBQT using Meeko
                pdbqt_path = None
                if convert_to_pdbqt:
                    try:
                        pdbqt_filename = f"{t_name}.pdbqt"
                        pdbqt_path = os.path.join(pdbqt_dir, pdbqt_filename)
                        
                        preparer = meeko.MoleculePreparation()
                        mol_setups = preparer.prepare(t_mol)
                        if not mol_setups:
                            raise ValueError("Meeko preparation returned empty setup.")
                        
                        writer = meeko.PDBQTWriterLegacy()
                        pdbqt_string, is_ok, err = writer.write_string(mol_setups[0])
                        if not is_ok:
                            raise ValueError(f"Meeko writer error: {err}")
                            
                        with open(pdbqt_path, "w") as f_pdbqt:
                            f_pdbqt.write(pdbqt_string)
                            
                    except Exception as meeko_err:
                        logger.warning(f"PDBQT conversion failed for {t_name}: {meeko_err}")
                        pdbqt_path = None

                prepared_records.append({
                    "name": t_name,
                    "original_name": orig_name,
                    "cid": cid,
                    "smiles": props["SMILES"],
                    "mw": props["MW"],
                    "logp": props["LogP"],
                    "pdbqt_path": pdbqt_path
                })

        except Exception as e:
            logger.error(f"Failed to prepare ligand {name}: {e}")
            quarantine_item(
                output_dir=output_dir,
                stage="ligand_preparation",
                mol_id=name,
                reason=str(e),
                details={"original_name": orig_name, "cid": cid},
            )
            failed_ligands.append({
                "name": name,
                "error": str(e)
            })

    # Write combined SDF
    if prepared_mols:
        write_sdf(prepared_mols, combined_sdf_path)
    else:
        combined_sdf_path = None

    duration = time.time() - start_time
    metrics = {
        "total_input": len(raw_mols),
        "total_prepared": len(prepared_mols),
        "failed_count": len(failed_ligands),
        "failed": failed_ligands,
        "prepared_ligands": prepared_records,
        "combined_sdf": combined_sdf_path,
        "duration": round(duration, 2),
        "status": "completed" if len(prepared_mols) > 0 else "failed"
    }

    write_json(metrics, report_path)
    log_step(logger, "Ligand Preparation", "completed", {
        "prepared_count": len(prepared_mols),
        "failed_count": len(failed_ligands),
        "sdf_output": combined_sdf_path,
        "duration": f"{duration:.1f}s"
    })

    # Write mapping file in ligands_pdbqt/mapping.csv
    if convert_to_pdbqt and prepared_records:
        mapping_path = os.path.join(pdbqt_dir, "mapping.csv")
        import csv
        try:
            with open(mapping_path, "w", newline="", encoding="utf-8") as f_map:
                writer = csv.writer(f_map)
                writer.writerow(["Original_Name", "CID", "PDBQT_File"])
                for record in prepared_records:
                    if record.get("pdbqt_path"):
                        writer.writerow([
                            record.get("original_name", "N/A"),
                            record.get("cid", "N/A"),
                            os.path.basename(record.get("pdbqt_path"))
                        ])
            logger.info(f"Wrote mapping file to {mapping_path}")
        except Exception as map_err:
            logger.warning(f"Failed to write mapping file: {map_err}")

    return metrics


def load_ligands_from_file(filepath: str) -> list[Chem.Mol]:
    """Helper to load molecules from a single file based on its extension."""
    ext = Path(filepath).suffix.lower()
    mols = []
    base_name = Path(filepath).stem
    
    if ext == ".sdf":
        mols = read_sdf(filepath)
    elif ext in (".smi", ".smiles", ".txt", ".tsv", ".csv"):
        mols = load_ligands_from_smiles_file(filepath)
    elif ext == ".mol":
        mol = Chem.MolFromMolFile(filepath, removeHs=False, sanitize=True)
        if mol is not None:
            mol.SetProp("_source_filename", base_name)
            mol.SetProp("_source_format", "mol")
            mols.append(mol)
    elif ext == ".mol2":
        mol = Chem.MolFromMol2File(filepath, removeHs=False, sanitize=True)
        if mol is not None:
            mol.SetProp("_source_filename", base_name)
            mol.SetProp("_source_format", "mol2")
            mols.append(mol)
    elif ext == ".pdb":
        logger.warning("\nWARNING:\nPDB ligand chemistry may be incomplete.\nPlease verify bond orders.\n")
        mol = Chem.MolFromPDBFile(filepath, removeHs=False)
        if mol is not None:
            mol.SetProp("_source_filename", base_name)
            mol.SetProp("_source_format", "pdb")
            # Overwrite _Name to prioritize filename over PDB Header
            mol.SetProp("_Name", base_name)
            mols.append(mol)
    else:
        raise ValueError(f"Unsupported ligand file format: {ext}")
        
    return mols


def load_ligands(input_path: str) -> list[Chem.Mol]:
    """
    Loads molecules from a single file, directory of files, or a SMILES file.
    """
    mols = []
    
    if os.path.isdir(input_path):
        logger.info(f"Loading ligands from directory: {input_path}")
        extensions = [".sdf", ".mol", ".mol2", ".pdb", ".smi", ".smiles", ".txt", ".tsv", ".csv"]
        ligand_files = list_files(input_path, extensions=extensions)
        for f in ligand_files:
            try:
                mols.extend(load_ligands_from_file(f))
            except Exception as e:
                logger.warning(f"Failed to load ligand file {f}: {e}")
                
    elif os.path.isfile(input_path):
        mols = load_ligands_from_file(input_path)
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")
        
    return mols


def load_ligands_from_smiles_file(filepath: str) -> list[Chem.Mol]:
    """
    Loads molecules from a SMILES file containing: <smiles> <name>
    """
    validate_file_exists(filepath, "SMILES file")
    mols = []
    base_name = Path(filepath).stem
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith(("#", "//")):
                continue
            
            parts = line.split()
            smiles = parts[0]
            name = parts[1] if len(parts) > 1 else f"ligand_{idx+1}"
            
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    set_mol_name(mol, name)
                    mol.SetProp("_source_filename", base_name)
                    mol.SetProp("_source_format", "smi")
                    mols.append(mol)
                else:
                    logger.warning(f"Failed to parse SMILES string on line {idx+1}: {smiles}")
            except Exception as e:
                logger.warning(f"Error parsing SMILES on line {idx+1}: {e}")
                
    return mols
