"""
Tests for shared chemistry utilities and medicinal chemistry metrics.
"""

import pytest
from rdkit import Chem
from utils.chemistry import (
    smiles_to_mol,
    mol_to_smiles,
    embed_conformer_robust,
    calculate_ligand_efficiency,
    calculate_lipophilic_efficiency,
    calculate_binding_efficiency_index,
    calculate_basic_properties,
)


def test_smiles_conversion(sample_smiles):
    for name, smi in sample_smiles.items():
        mol = smiles_to_mol(smi)
        assert mol is not None
        canon = mol_to_smiles(mol)
        assert isinstance(canon, str)
        assert len(canon) > 0


def test_invalid_smiles():
    with pytest.raises(ValueError):
        smiles_to_mol("INVALID_NOT_A_SMILES_STRING")


def test_embed_conformer_robust():
    mol = smiles_to_mol("CC(=O)Oc1ccccc1C(=O)O")
    mol = Chem.AddHs(mol)
    mol_3d, ok = embed_conformer_robust(mol, num_confs=1, random_seed=42)
    assert ok is True
    assert mol_3d.GetNumConformers() >= 1


def test_ligand_efficiency():
    # -6.5 kcal/mol with 13 heavy atoms = 6.5 / 13 = 0.5 kcal/mol/atom
    le = calculate_ligand_efficiency(binding_affinity_kcal_mol=-6.5, heavy_atom_count=13)
    assert le == 0.5

    # Zero heavy atoms edge case
    assert calculate_ligand_efficiency(-7.0, 0) == 0.0


def test_lipophilic_efficiency():
    # pKd ~ -(-8.22) / 1.37 = 6.00. LogP = 1.0 -> LLE = 5.0
    lle = calculate_lipophilic_efficiency(binding_affinity_kcal_mol=-8.22, logp=1.0)
    assert pytest.approx(lle, rel=1e-2) == 5.0


def test_binding_efficiency_index():
    # pKd = 6.0, MW = 300 -> BEI = 6.0 / 0.3 = 20.0
    bei = calculate_binding_efficiency_index(binding_affinity_kcal_mol=-8.22, mw=300.0)
    assert pytest.approx(bei, rel=1e-2) == 20.0


def test_calculate_basic_properties(sample_mols):
    for mol in sample_mols:
        props = calculate_basic_properties(mol)
        assert "MW" in props
        assert "LogP" in props
        assert "HBD" in props
        assert "HBA" in props
        assert "TPSA" in props
        assert props["MW"] > 0
