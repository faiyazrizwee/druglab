"""
DrugLab Pytest Configuration and Shared Fixtures.
"""

import os
import shutil
import tempfile
import pytest
import pandas as pd
from rdkit import Chem

from utils.chemistry import smiles_to_mol, set_mol_name


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs, cleaned up after test."""
    d = tempfile.mkdtemp(prefix="druglab_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_smiles():
    """Return dictionary of standard drug molecules as SMILES."""
    return {
        "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "paracetamol": "CC(=O)Nc1ccc(O)cc1",
        "caffeine": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
        "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    }


@pytest.fixture
def sample_mols(sample_smiles):
    """Return list of RDKit Mol objects."""
    mols = []
    for name, smi in sample_smiles.items():
        m = smiles_to_mol(smi)
        set_mol_name(m, name)
        mols.append(m)
    return mols


@pytest.fixture
def mock_docking_results_df():
    """Return sample docking results DataFrame."""
    return pd.DataFrame([
        {
            "name": "aspirin",
            "smiles": "CC(=O)Oc1ccccc1C(=O)O",
            "best_affinity": -6.8,
            "HeavyAtoms": 13,
            "MW": 180.16,
            "LogP": 1.19,
            "n_poses": 9,
            "pose_file": "aspirin_poses.pdbqt",
            "status": "completed",
        },
        {
            "name": "paracetamol",
            "smiles": "CC(=O)Nc1ccc(O)cc1",
            "best_affinity": -5.9,
            "HeavyAtoms": 11,
            "MW": 151.16,
            "LogP": 0.46,
            "n_poses": 9,
            "pose_file": "paracetamol_poses.pdbqt",
            "status": "completed",
        },
        {
            "name": "caffeine",
            "smiles": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
            "best_affinity": -6.2,
            "HeavyAtoms": 14,
            "MW": 194.19,
            "LogP": -0.07,
            "n_poses": 9,
            "pose_file": "caffeine_poses.pdbqt",
            "status": "completed",
        },
    ])


@pytest.fixture
def mock_admet_df():
    """Return sample ADMET results DataFrame."""
    return pd.DataFrame([
        {"name": "aspirin", "admet_risk": 15.0, "status": "PASS"},
        {"name": "paracetamol", "admet_risk": 20.0, "status": "PASS"},
        {"name": "caffeine", "admet_risk": 10.0, "status": "PASS"},
    ])


@pytest.fixture
def mock_druglikeness_df():
    """Return sample druglikeness descriptors DataFrame."""
    return pd.DataFrame([
        {"name": "aspirin", "MW": 180.16, "LogP": 1.19, "druglikeness_violations": 0, "status": "PASS"},
        {"name": "paracetamol", "MW": 151.16, "LogP": 0.46, "druglikeness_violations": 0, "status": "PASS"},
        {"name": "caffeine", "MW": 194.19, "LogP": -0.07, "druglikeness_violations": 0, "status": "PASS"},
    ])
