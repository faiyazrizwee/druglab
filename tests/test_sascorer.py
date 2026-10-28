"""
Tests for Synthetic Accessibility Scorer (SAScore).
"""

import pytest
from utils.chemistry import smiles_to_mol
from utils.sascorer import calculate_sascore


def test_sascore_simple_drugs():
    # Simple, highly accessible commercial drugs should have low SAScore (< 3.5)
    aspirin = smiles_to_mol("CC(=O)Oc1ccccc1C(=O)O")
    score_asp = calculate_sascore(aspirin)
    assert 1.0 <= score_asp <= 3.5

    paracetamol = smiles_to_mol("CC(=O)Nc1ccc(O)cc1")
    score_par = calculate_sascore(paracetamol)
    assert 1.0 <= score_par <= 3.5


def test_sascore_complex_molecules():
    # Highly complex natural product / macrocycle with many chiral centers
    # Paclitaxel (Taxol)
    taxol_smiles = (
        "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C"
    )
    taxol = smiles_to_mol(taxol_smiles)
    score_taxol = calculate_sascore(taxol)
    # Taxol should have a significantly higher SAScore (> 5.0)
    assert score_taxol >= 5.0


def test_sascore_none_handling():
    assert calculate_sascore(None) == 10.0
