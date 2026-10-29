"""
Input validation for receptor structures and peptide sequences/structures.
"""

from __future__ import annotations

import os
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.logger import setup_logger

logger = setup_logger("input_validation", level="INFO")

STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class PeptideCandidate:
    """Represents a peptide candidate throughout the workflow."""
    id: str
    sequence: str
    source: str  # "fasta", "sequence", "structure"
    structure_path: str | None = None
    length: int = 0
    checksum: str = ""
    status: str = "pending"  # pending, filtered, prepared, docking, completed, failed
    failure_reason: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    docking_results: list[dict[str, Any]] = field(default_factory=list)
    prepared_structure_path: str | None = None
    prepared_checksum: str = ""

    def __post_init__(self):
        if not self.length and self.sequence:
            self.length = len(self.sequence)
        if not self.checksum and self.sequence:
            self.checksum = compute_sequence_checksum(self.sequence)


def compute_file_checksum(filepath: str | Path, algorithm: str = "sha256") -> str:
    """Compute cryptographic hash of a file for run manifest and provenance."""
    hasher = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_sequence_checksum(sequence: str) -> str:
    """Compute SHA256 checksum of normalized amino acid sequence."""
    return hashlib.sha256(sequence.strip().upper().encode("utf-8")).hexdigest()


def validate_protein_pdb(pdb_path: str | Path) -> dict[str, Any]:
    """
    Validate receptor PDB file structure, existence, and extract chains.
    """
    path = Path(pdb_path)
    if not path.is_file():
        raise FileNotFoundError(f"Protein PDB file not found: {pdb_path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Protein PDB file is empty: {pdb_path}")

    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("receptor", str(path))
    except Exception as e:
        raise ValueError(f"Failed to parse protein PDB structure {pdb_path}: {e}")

    chains = []
    residue_count = 0
    atom_count = 0
    models = list(structure.get_models())
    if not models:
        raise ValueError(f"No models found in PDB structure: {pdb_path}")

    model = models[0]
    for chain in model.get_chains():
        chains.append(chain.id)
        residues = [r for r in chain.get_residues() if r.id[0] == " "]
        residue_count += len(residues)
        for r in residues:
            atom_count += len(list(r.get_atoms()))

    if residue_count == 0:
        raise ValueError(f"Protein PDB structure contains no amino acid residues: {pdb_path}")

    checksum = compute_file_checksum(path)
    return {
        "path": str(path.resolve()),
        "checksum": checksum,
        "chains": chains,
        "num_residues": residue_count,
        "num_atoms": atom_count,
    }


def validate_peptide_sequence(
    sequence: str,
    allow_nonstandard: bool = False,
) -> str:
    """
    Validate single-letter amino acid sequence and return cleaned uppercase string.
    """
    cleaned = "".join(sequence.split()).upper()
    if not cleaned:
        raise ValueError("Peptide sequence cannot be empty.")

    invalid_chars = set(cleaned) - STANDARD_AMINO_ACIDS
    if invalid_chars and not allow_nonstandard:
        raise ValueError(
            f"Peptide sequence contains non-standard amino acid codes: {sorted(invalid_chars)}. "
            f"Expected standard 20 amino acids ({sorted(STANDARD_AMINO_ACIDS)})."
        )
    return cleaned


def parse_fasta_content(
    fasta_text: str,
    allow_nonstandard: bool = False,
    source_name: str = "fasta",
) -> list[PeptideCandidate]:
    """
    Parse FASTA-formatted text containing one or multiple peptide records.
    Returns a list of validated PeptideCandidate objects with unique, filesystem-safe IDs.
    """
    candidates: list[PeptideCandidate] = []
    seen_ids: set[str] = set()
    current_header = ""
    current_seq_lines: list[str] = []

    def _commit_candidate(header: str, seq_lines: list[str]):
        if not header and not seq_lines:
            return
        seq_str = "".join(seq_lines).strip()
        if not seq_str:
            return
        try:
            valid_seq = validate_peptide_sequence(seq_str, allow_nonstandard=allow_nonstandard)
        except Exception as e:
            logger.warning(f"Skipping invalid FASTA entry '{header}': {e}")
            return

        raw_id = header.split()[0] if header else f"peptide_{len(candidates) + 1}"
        # Sanitize ID to be safe for filenames and directories
        clean_id = re.sub(r"[^\w\-.]", "_", raw_id).strip("_") or f"peptide_{len(candidates) + 1}"

        # Deduplicate ID if multiple entries share the same header
        if clean_id in seen_ids:
            suffix = 2
            candidate_id = f"{clean_id}_{suffix}"
            while candidate_id in seen_ids:
                suffix += 1
                candidate_id = f"{clean_id}_{suffix}"
        else:
            candidate_id = clean_id

        seen_ids.add(candidate_id)
        candidates.append(
            PeptideCandidate(
                id=candidate_id,
                sequence=valid_seq,
                source=source_name,
                length=len(valid_seq),
                checksum=compute_sequence_checksum(valid_seq),
            )
        )

    for line in fasta_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_header or current_seq_lines:
                _commit_candidate(current_header, current_seq_lines)
            current_header = line[1:].strip()
            current_seq_lines = []
        else:
            current_seq_lines.append(line)

    if current_header or current_seq_lines:
        _commit_candidate(current_header, current_seq_lines)

    return candidates


def parse_fasta_file(
    fasta_path: str | Path,
    allow_nonstandard: bool = False,
) -> list[PeptideCandidate]:
    """
    Parse a FASTA file or directory of FASTA files containing one or multiple peptides.
    """
    path = Path(fasta_path)
    if not path.exists():
        raise FileNotFoundError(f"Peptide FASTA path not found: {fasta_path}")

    # If directory, find and parse all FASTA files
    if path.is_dir():
        fasta_files = sorted(
            [f for f in path.glob("*") if f.suffix.lower() in (".fasta", ".fa", ".faa", ".fna")]
        )
        if not fasta_files:
            raise ValueError(f"No FASTA files (.fasta, .fa, .faa) found in directory: {fasta_path}")
        all_candidates: list[PeptideCandidate] = []
        for ff in fasta_files:
            with open(ff, "r", encoding="utf-8") as f:
                cands = parse_fasta_content(f.read(), allow_nonstandard=allow_nonstandard, source_name=ff.name)
                all_candidates.extend(cands)
        if not all_candidates:
            raise ValueError(f"No valid FASTA records found in directory: {fasta_path}")
        return all_candidates

    if path.stat().st_size == 0:
        raise ValueError(f"Peptide FASTA file is empty: {fasta_path}")

    with open(path, "r", encoding="utf-8") as f:
        candidates = parse_fasta_content(f.read(), allow_nonstandard=allow_nonstandard, source_name=path.name)

    if not candidates:
        raise ValueError(f"No valid FASTA records found in {fasta_path}")

    return candidates


def validate_peptide_structure(
    structure_path: str | Path,
) -> PeptideCandidate:
    """
    Validate a user-supplied peptide PDB structure and extract its sequence.
    """
    path = Path(structure_path)
    if not path.is_file():
        raise FileNotFoundError(f"Peptide structure file not found: {structure_path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Peptide structure file is empty: {structure_path}")

    from Bio.PDB import PDBParser, Polypeptide
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("peptide", str(path))
    except Exception as e:
        raise ValueError(f"Failed to parse peptide PDB structure {structure_path}: {e}")

    # Extract sequence from peptide chain
    residues = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] == " " and Polypeptide.is_aa(res, standard=True):
                    residues.append(res)
        break

    if not residues:
        raise ValueError(f"No standard amino acid residues found in peptide structure: {structure_path}")

    sequence = "".join([Polypeptide.three_to_one(r.get_resname()) for r in residues])
    checksum = compute_file_checksum(path)
    cid = path.stem

    return PeptideCandidate(
        id=cid,
        sequence=sequence,
        source="structure",
        structure_path=str(path.resolve()),
        length=len(sequence),
        checksum=checksum,
    )
