"""
Unit tests for peptide and receptor preparation.
"""

import os
from pathlib import Path
from core.peptide_docking.input_validation import PeptideCandidate
from core.peptide_docking.peptide_preparation import (
    generate_peptide_3d_structure,
    prepare_peptide_candidate,
)
from core.peptide_docking.protein_preparation import prepare_receptor_for_lightdock


def test_generate_peptide_3d_structure(tmp_path):
    out_pdb = tmp_path / "test_pep.pdb"
    res = generate_peptide_3d_structure(
        sequence="ACDEF",
        output_pdb_path=out_pdb,
        chain_id="B",
        num_conformers=1,
        minimize=True,
    )
    assert out_pdb.is_file()
    assert res["num_conformers_sampled"] >= 1
    assert res["energy_kcal_mol"] is not None

    # Check PDB content has chain B and standard atoms
    content = out_pdb.read_text()
    assert "ATOM" in content
    assert " B " in content or " B  " in content
    assert "ALA B" in content or "ALA" in content


def test_generate_long_peptide_3d_structure(tmp_path):
    """Ensure long peptides (>30 AA) embed and optimize reliably without distance geometry failure."""
    long_seq = "GLWSKIKEVGKEAAKAAAKAAGKAALGAVSEAV"
    out_pdb = tmp_path / "long_pep.pdb"
    res = generate_peptide_3d_structure(
        sequence=long_seq,
        output_pdb_path=out_pdb,
        chain_id="B",
        num_conformers=1,
        minimize=True,
    )
    assert out_pdb.is_file()
    assert out_pdb.stat().st_size > 1000
    assert res["num_conformers_sampled"] >= 1


def test_prepare_peptide_candidate(tmp_path):
    cand = PeptideCandidate(id="p1", sequence="WVALY", source="sequence")
    prep_cand = prepare_peptide_candidate(cand, output_dir=tmp_path)
    assert prep_cand.status == "prepared"
    assert os.path.isfile(prep_cand.prepared_structure_path)
    assert len(prep_cand.prepared_checksum) == 64


def test_prepare_receptor_for_lightdock(tmp_path):
    # Use existing PDB if available
    pdb_path = Path("protein/3DWT.pdb")
    if pdb_path.is_file():
        res = prepare_receptor_for_lightdock(
            input_pdb=pdb_path,
            output_dir=tmp_path,
            remove_waters=True,
        )
        assert os.path.isfile(res["prepared_path"])
        assert res["num_residues"] > 0
