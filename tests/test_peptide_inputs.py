"""
Unit tests for input validation (sequences, FASTA, PDB structures).
"""

import pytest
from pathlib import Path
from core.peptide_docking.input_validation import (
    validate_peptide_sequence,
    parse_fasta_file,
    validate_protein_pdb,
    compute_sequence_checksum,
    compute_file_checksum,
    PeptideCandidate,
)


def test_validate_peptide_sequence_canonical():
    seq = "acdefghiklmnpqrstvwy"
    valid = validate_peptide_sequence(seq)
    assert valid == "ACDEFGHIKLMNPQRSTVWY"


def test_validate_peptide_sequence_invalid():
    with pytest.raises(ValueError, match="non-standard amino acid"):
        validate_peptide_sequence("ACDEF123")

    with pytest.raises(ValueError, match="cannot be empty"):
        validate_peptide_sequence("   ")


def test_parse_fasta_file(tmp_path):
    fasta_path = tmp_path / "peptides.fasta"
    fasta_path.write_text("""
>peptide_A Description of A
ACDEFGH
>peptide_B
KLMNPQRSTV
""")
    candidates = parse_fasta_file(fasta_path)
    assert len(candidates) == 2
    assert candidates[0].id == "peptide_A"
    assert candidates[0].sequence == "ACDEFGH"
    assert candidates[0].length == 7
    assert candidates[1].id == "peptide_B"
    assert candidates[1].sequence == "KLMNPQRSTV"
    assert candidates[1].length == 10


def test_parse_multi_fasta_library_resilience(tmp_path):
    """Test multi-peptide FASTA with special characters, duplicates, and multiline sequences."""
    fasta_path = tmp_path / "multi_library.fasta"
    fasta_path.write_text("""
>sp|P12345|PEP_A first lead
ACDE
FGHI
>dup_id duplicate 1
WYVIK
>dup_id duplicate 2
GLWSKI
>invalid_one
XYZ123BAD
>final_pep
AAGKAALGAVSEAV
""")
    candidates = parse_fasta_file(fasta_path)
    # The invalid sequence is skipped; the 4 valid peptides are parsed
    assert len(candidates) == 4
    # Special character sanitization
    assert "sp_P12345_PEP_A" in candidates[0].id
    assert candidates[0].sequence == "ACDEFGHI"
    # Deduplication
    assert candidates[1].id == "dup_id"
    assert candidates[2].id == "dup_id_2"
    assert candidates[3].id == "final_pep"


def test_parse_fasta_directory(tmp_path):
    """Test loading peptides from a directory containing multiple FASTA files."""
    d = tmp_path / "fasta_dir"
    d.mkdir()
    (d / "f1.fasta").write_text(">pep1\nACDEF\n")
    (d / "f2.fa").write_text(">pep2\nWYVIK\n>pep3\nGLWSK\n")

    candidates = parse_fasta_file(d)
    assert len(candidates) == 3
    ids = [c.id for c in candidates]
    assert "pep1" in ids
    assert "pep2" in ids
    assert "pep3" in ids


def test_validate_protein_pdb_real(tmp_path):
    # Test on a real PDB if available in project, e.g. protein/3DWT.pdb
    pdb_path = Path("protein/3DWT.pdb")
    if pdb_path.is_file():
        val = validate_protein_pdb(pdb_path)
        assert len(val["chains"]) > 0
        assert val["num_residues"] > 0
        assert len(val["checksum"]) == 64
