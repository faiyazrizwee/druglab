"""
Tests for Data Fetch Utility (RCSB PDB and PubChem).
"""

import os
import urllib.error
import pytest
from utils.fetch import fetch_pdb, fetch_ligand_by_name, fetch_ligand_by_cid


def test_invalid_pdb_id():
    with pytest.raises(ValueError):
        fetch_pdb("INVALID_ID_TOO_LONG")


def test_invalid_ligand_name():
    with pytest.raises(ValueError):
        fetch_ligand_by_name("   ")


def test_invalid_cid():
    with pytest.raises(ValueError):
        fetch_ligand_by_cid("NOT_A_NUMBER")


def test_fetch_aspirin_pubchem(temp_dir):
    out_path = os.path.join(temp_dir, "aspirin.sdf")
    try:
        fetched = fetch_ligand_by_name("aspirin", output_path=out_path)
        assert os.path.isfile(fetched)
        with open(fetched, "r", encoding="utf-8") as f:
            content = f.read()
        assert "$$$$" in content or "M  END" in content
    except urllib.error.URLError as e:
        pytest.skip(f"External network unavailable for PubChem test: {e}")
