"""
Unit tests for binding-site selection and restraint formatting.
"""

import pytest
from pathlib import Path
from Bio.PDB import PDBParser
from core.peptide_docking.config import BindingSiteConfig
from core.peptide_docking.binding_site import (
    validate_residue_identifier,
    setup_binding_site_and_restraints,
)
from core.peptide_docking.peptide_preparation import generate_peptide_3d_structure


def test_validate_residue_identifier(tmp_path):
    # Generate a small peptide structure to test residue identifier resolution
    pdb_path = tmp_path / "pep.pdb"
    generate_peptide_3d_structure("ACDEF", pdb_path, chain_id="B")

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("pep", str(pdb_path))

    # Valid format: B.ALA.1
    chain, resname, resnum, canon = validate_residue_identifier("B.ALA.1", struct)
    assert chain == "B"
    assert resname == "ALA"
    assert resnum == 1
    assert canon == "B.ALA.1"

    # Non-existent residue number
    with pytest.raises(ValueError, match="not found in prepared structure"):
        validate_residue_identifier("B.ALA.99", struct)

    # Malformed format
    with pytest.raises(ValueError, match="Invalid residue identifier format"):
        validate_residue_identifier("B", struct)


def test_setup_binding_site_restraints(tmp_path):
    rec_pdb = tmp_path / "rec.pdb"
    pep_pdb = tmp_path / "pep.pdb"
    generate_peptide_3d_structure("ACDEFGHIKL", rec_pdb, chain_id="A")
    generate_peptide_3d_structure("WYV", pep_pdb, chain_id="B")

    # Explicit restraints
    cfg = BindingSiteConfig(
        mode="explicit",
        receptor_residues=["A.ALA.1", "A.CYS.2"],
        peptide_residues=["B.TRP.1"],
    )
    res = setup_binding_site_and_restraints(
        receptor_pdb=rec_pdb,
        peptide_pdb=pep_pdb,
        config=cfg,
        output_dir=tmp_path,
    )
    assert res.mode == "explicit"
    assert res.restraints_path is not None
    assert Path(res.restraints_path).is_file()

    content = Path(res.restraints_path).read_text()
    assert "R A.ALA.1" in content
    assert "R A.CYS.2" in content
    assert "L B.TRP.1" in content


def test_setup_binding_site_global(tmp_path):
    rec_pdb = tmp_path / "rec.pdb"
    pep_pdb = tmp_path / "pep.pdb"
    generate_peptide_3d_structure("ACDEF", rec_pdb, chain_id="A")
    generate_peptide_3d_structure("WYV", pep_pdb, chain_id="B")

    cfg = BindingSiteConfig(mode="global")
    res = setup_binding_site_and_restraints(
        receptor_pdb=rec_pdb,
        peptide_pdb=pep_pdb,
        config=cfg,
        output_dir=tmp_path,
    )
    assert res.mode == "global"
    assert res.restraints_path is None
    assert len(res.receptor_restraints) == 0
