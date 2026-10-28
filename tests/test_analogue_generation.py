"""
Tests for Analogue Generation Module with SAScore Filtering.
"""

import os
import pytest
from rdkit import Chem
from utils.io import read_sdf
from lead_optimization.analogue_generation import generate_analogues


def test_generate_analogues_bioisostere(temp_dir):
    lead_file = "ligands/aspirin.sdf"
    if not os.path.exists(lead_file):
        pytest.skip("aspirin.sdf not found.")

    res = generate_analogues(
        lead_sdf=lead_file,
        output_dir=temp_dir,
        strategies=["bioisostere"],
        max_analogues=10,
        max_sascore=6.0,
    )

    assert res["status"] in ("completed", "no_analogues")
    if res["status"] == "completed":
        assert os.path.isfile(res["analogues_sdf"])
        mols = read_sdf(res["analogues_sdf"])
        assert len(mols) > 0
        for m in mols:
            assert m.HasProp("SAScore")
            sascore = float(m.GetProp("SAScore"))
            assert sascore <= 6.0
