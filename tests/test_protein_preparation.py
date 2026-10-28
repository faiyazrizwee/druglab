"""
Tests for core.protein_preparation module.
Verifies robust handling of non-standard residues, missing residues, and PDBQT generation.
"""

import os
import pytest
from core.protein_preparation import prepare_protein


def test_prepare_protein_6lu7(tmp_path):
    pdb_path = "protein/6LU7.pdb"
    if not os.path.exists(pdb_path):
        pytest.skip(f"{pdb_path} not found")

    out_dir = str(tmp_path / "6lu7_prep")
    result = prepare_protein(
        input_pdb=pdb_path,
        output_dir=out_dir,
        chain_id="A",
        remove_water=True,
        remove_heterogens=True,
        add_hydrogens=True,
        fix_missing_residues=True,
        fix_missing_atoms=True,
    )

    assert result["status"] == "completed"
    assert os.path.exists(result["clean_pdb"])
    assert os.path.exists(result["pdbqt"])
    assert os.path.getsize(result["pdbqt"]) > 0
    assert result["num_residues"] > 0


def test_prepare_protein_all_chains_with_heterogen_residues(tmp_path):
    """
    Test 6LU7 without chain filtering.
    6LU7 has non-standard peptidomimetic inhibitor residues (02J, PJE, 010) on Chain C.
    This ensures that PDBFixer does not crash on non-standard SEQRES entries.
    """
    pdb_path = "protein/6LU7.pdb"
    if not os.path.exists(pdb_path):
        pytest.skip(f"{pdb_path} not found")

    out_dir = str(tmp_path / "6lu7_all_chains")
    result = prepare_protein(
        input_pdb=pdb_path,
        output_dir=out_dir,
        chain_id=None,
        remove_water=True,
        remove_heterogens=True,
        add_hydrogens=True,
        fix_missing_residues=True,
        fix_missing_atoms=True,
    )

    assert result["status"] == "completed"
    assert os.path.exists(result["clean_pdb"])
    assert os.path.exists(result["pdbqt"])
