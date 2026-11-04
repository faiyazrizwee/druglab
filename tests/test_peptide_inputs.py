"""
Unit tests for input validation (sequences, FASTA, PDB structures).
"""

import pytest
from pathlib import Path
from core.peptide_docking.input_validation import (
    validate_peptide_sequence,
    parse_fasta_file,
    validate_protein_pdb,
    validate_peptide_structure,
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


def test_validate_peptide_structure(tmp_path):
    # Test synthetic peptide PDB with standard ALA-CYS-ASP
    p = tmp_path / "test_pep.pdb"
    p.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  C   ALA A   1       2.000   1.200   0.000  1.00  0.00           C\n"
        "ATOM      4  O   ALA A   1       1.500   2.200   0.000  1.00  0.00           O\n"
        "ATOM      5  N   CYS A   2       3.200   1.100   0.000  1.00  0.00           N\n"
        "ATOM      6  CA  CYS A   2       4.100   2.200   0.000  1.00  0.00           C\n"
        "ATOM      7  C   CYS A   2       5.500   1.800   0.000  1.00  0.00           C\n"
        "ATOM      8  O   CYS A   2       6.200   2.600   0.000  1.00  0.00           O\n"
    )
    cand = validate_peptide_structure(p)
    assert cand.id == "test_pep"
    assert cand.sequence == "AC"
    assert cand.length == 2
    assert cand.source == "structure"
    assert cand.checksum != ""


def test_validate_peptide_structure_real_chain_b():
    pdb_path = Path("ligands/peptide_chainB.pdb")
    if pdb_path.is_file():
        cand = validate_peptide_structure(pdb_path)
        assert cand.id == "peptide_chainB"
        assert cand.length == 31
        assert cand.sequence == "DIFCGETCAFIPCITHVPGTCSCKSKVCYFN"

