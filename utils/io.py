"""
DrugLab File I/O Utilities.

Centralizes all file read/write operations for molecular formats
(SDF, PDB, PDBQT), tabular data (CSV), and structured data (JSON/YAML).
Every module uses these helpers instead of raw file operations.

Usage:
    from utils.io import read_sdf, write_sdf, ensure_dir, validate_file_exists
    mols = read_sdf("ligands/library.sdf")
    write_sdf(mols, "output/prepared.sdf")
"""

import json
import os
from pathlib import Path

import pandas as pd
import yaml

from utils.logger import setup_logger

logger = setup_logger("io", level="INFO")


# ──────────────────────────────────────────────────────────────
# SDF (Structure-Data File) I/O
# ──────────────────────────────────────────────────────────────

def read_sdf(filepath: str) -> list:
    """
    Read an SDF file and return a list of RDKit Mol objects.

    Invalid/unparseable entries are skipped with a warning.

    Args:
        filepath: Path to the SDF file.

    Returns:
        List of RDKit Mol objects (only valid molecules).

    Raises:
        FileNotFoundError: If filepath doesn't exist.
    """
    from rdkit import Chem

    validate_file_exists(filepath, "SDF file")

    if os.path.getsize(filepath) == 0:
        logger.info(f"SDF file {filepath} is empty. Returning empty list.")
        return []

    supplier = Chem.SDMolSupplier(filepath, removeHs=False, sanitize=True)
    mols = []
    skipped = 0
    base_name = Path(filepath).stem

    for i, mol in enumerate(supplier):
        if mol is None:
            skipped += 1
            logger.warning(f"Skipped invalid molecule at index {i} in {filepath}")
            continue
        mol.SetProp("_source_filename", base_name)
        mol.SetProp("_source_format", "sdf")
        mols.append(mol)

    logger.info(f"Read {len(mols)} molecules from {filepath}" +
                (f" (skipped {skipped} invalid)" if skipped else ""))
    return mols


def write_sdf(
    mols: list,
    filepath: str,
    properties: list[str] | None = None,
) -> str:
    """
    Write RDKit Mol objects to an SDF file.

    Args:
        mols: List of RDKit Mol objects.
        filepath: Output SDF file path.
        properties: Optional list of property names to include.
                   If None, all properties on the Mol are written.

    Returns:
        The output filepath.
    """
    from rdkit import Chem

    ensure_dir(os.path.dirname(filepath))

    writer = Chem.SDWriter(filepath)

    if properties:
        writer.SetProps(properties)

    for mol in mols:
        if mol is not None:
            writer.write(mol)

    writer.close()
    logger.info(f"Wrote {len(mols)} molecules to {filepath}")
    return filepath


def read_sdf_to_smiles(filepath: str) -> list[dict]:
    """
    Read an SDF file and extract SMILES + molecule name.

    Returns:
        List of dicts: [{"name": str, "smiles": str}, ...]
    """
    from rdkit import Chem

    mols = read_sdf(filepath)
    results = []
    for mol in mols:
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{len(results)}"
        smiles = Chem.MolToSmiles(mol)
        results.append({"name": name, "smiles": smiles})
    return results


# ──────────────────────────────────────────────────────────────
# PDB I/O
# ──────────────────────────────────────────────────────────────

def read_pdb(filepath: str) -> str:
    """
    Read a PDB file and return its contents as a string.

    Args:
        filepath: Path to the PDB file.

    Returns:
        PDB file contents as string.

    Raises:
        FileNotFoundError: If filepath doesn't exist.
    """
    validate_file_exists(filepath, "PDB file")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    logger.debug(f"Read PDB file: {filepath} ({len(content)} bytes)")
    return content


def write_pdb(content: str, filepath: str) -> str:
    """
    Write PDB content string to a file.

    Args:
        content: PDB file contents.
        filepath: Output file path.

    Returns:
        The output filepath.
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    logger.debug(f"Wrote PDB file: {filepath}")
    return filepath


# ──────────────────────────────────────────────────────────────
# PDBQT I/O
# ──────────────────────────────────────────────────────────────

def read_pdbqt(filepath: str) -> str:
    """
    Read a PDBQT file and return its contents as a string.

    Args:
        filepath: Path to the PDBQT file.

    Returns:
        PDBQT file contents as string.

    Raises:
        FileNotFoundError: If filepath doesn't exist.
    """
    validate_file_exists(filepath, "PDBQT file")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    logger.debug(f"Read PDBQT file: {filepath} ({len(content)} bytes)")
    return content


def write_pdbqt(content: str, filepath: str) -> str:
    """
    Write PDBQT content to a file.

    Args:
        content: PDBQT file contents.
        filepath: Output file path.

    Returns:
        The output filepath.
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    logger.debug(f"Wrote PDBQT file: {filepath}")
    return filepath


# ──────────────────────────────────────────────────────────────
# CSV I/O
# ──────────────────────────────────────────────────────────────

def read_csv(filepath: str) -> pd.DataFrame:
    """
    Read a CSV file into a Pandas DataFrame.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with the CSV contents.

    Raises:
        FileNotFoundError: If filepath doesn't exist.
    """
    validate_file_exists(filepath, "CSV file")
    df = pd.read_csv(filepath)
    logger.debug(f"Read CSV: {filepath} ({len(df)} rows, {len(df.columns)} columns)")
    return df


def write_csv(df: pd.DataFrame, filepath: str, index: bool = False) -> str:
    """
    Write a DataFrame to a CSV file.

    Args:
        df: DataFrame to write.
        filepath: Output file path.
        index: Whether to include the index column.

    Returns:
        The output filepath.
    """
    ensure_dir(os.path.dirname(filepath))
    df.to_csv(filepath, index=index)
    logger.info(f"Wrote CSV: {filepath} ({len(df)} rows)")
    return filepath


# ──────────────────────────────────────────────────────────────
# JSON I/O
# ──────────────────────────────────────────────────────────────

def read_json(filepath: str) -> dict:
    """
    Read a JSON file and return as dict.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Parsed JSON as dict.

    Raises:
        FileNotFoundError: If filepath doesn't exist.
    """
    validate_file_exists(filepath, "JSON file")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.debug(f"Read JSON: {filepath}")
    return data


def write_json(data: dict | list, filepath: str, indent: int = 2) -> str:
    """
    Write a dict/list to a JSON file.

    Args:
        data: Data to serialize.
        filepath: Output file path.
        indent: JSON indentation level.

    Returns:
        The output filepath.
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    logger.debug(f"Wrote JSON: {filepath}")
    return filepath


# ──────────────────────────────────────────────────────────────
# YAML I/O
# ──────────────────────────────────────────────────────────────

def read_yaml(filepath: str) -> dict:
    """
    Read a YAML file and return as dict.

    Args:
        filepath: Path to the YAML file.

    Returns:
        Parsed YAML as dict.

    Raises:
        FileNotFoundError: If filepath doesn't exist.
    """
    validate_file_exists(filepath, "YAML file")
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def write_yaml(data: dict, filepath: str) -> str:
    """
    Write a dict to a YAML file.

    Args:
        data: Data to serialize.
        filepath: Output file path.

    Returns:
        The output filepath.
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    logger.debug(f"Wrote YAML: {filepath}")
    return filepath


# ──────────────────────────────────────────────────────────────
# Path & Directory Utilities
# ──────────────────────────────────────────────────────────────

def ensure_dir(dirpath: str) -> str:
    """
    Create a directory (and parents) if it doesn't exist.

    Args:
        dirpath: Directory path to create.

    Returns:
        The directory path.
    """
    if dirpath:
        Path(dirpath).mkdir(parents=True, exist_ok=True)
    return dirpath


def validate_file_exists(filepath: str, description: str = "file") -> None:
    """
    Raise FileNotFoundError with a descriptive message if the file doesn't exist.

    Args:
        filepath: Path to check.
        description: Human-readable description for the error message.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"{description} not found: {filepath}\n"
            f"  Expected a valid file path. Check the path and try again."
        )


def validate_dir_exists(dirpath: str, description: str = "directory") -> None:
    """
    Raise FileNotFoundError if the directory doesn't exist.

    Args:
        dirpath: Directory path to check.
        description: Human-readable description for the error message.

    Raises:
        FileNotFoundError: If the directory doesn't exist.
    """
    if not os.path.isdir(dirpath):
        raise FileNotFoundError(
            f"{description} not found: {dirpath}\n"
            f"  Expected a valid directory path."
        )


def resolve_output_path(output_dir: str, stage: str, filename: str) -> str:
    """
    Build an output path: output_dir/stage/filename.

    Creates the directory structure if it doesn't exist.

    Args:
        output_dir: Root output directory.
        stage: Pipeline stage name (used as subdirectory).
        filename: Output filename.

    Returns:
        Full resolved path.
    """
    stage_dir = os.path.join(output_dir, stage)
    ensure_dir(stage_dir)
    return os.path.join(stage_dir, filename)


def list_files(dirpath: str, extensions: list[str] | None = None) -> list[str]:
    """
    List files in a directory, optionally filtering by extension.

    Args:
        dirpath: Directory to list.
        extensions: Optional list of extensions to filter (e.g., [".sdf", ".mol2"]).

    Returns:
        List of full file paths, sorted alphabetically.
    """
    validate_dir_exists(dirpath)
    files = []
    for f in sorted(os.listdir(dirpath)):
        full_path = os.path.join(dirpath, f)
        if os.path.isfile(full_path):
            if extensions is None or any(f.lower().endswith(ext.lower()) for ext in extensions):
                files.append(full_path)
    return files


def get_stage_dir(output_dir: str, stage: str) -> str:
    """
    Get the directory path for a pipeline stage, creating it if needed.

    Args:
        output_dir: Root output directory.
        stage: Stage name (e.g. "protein_preparation").

    Returns:
        Full stage directory path.
    """
    stage_dir = os.path.join(output_dir, stage)
    ensure_dir(stage_dir)
    return stage_dir
