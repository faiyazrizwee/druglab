"""
Tests for Data Fetch Utility (RCSB PDB, PubChem, and AlphaFold DB).
"""

import os
import urllib.error
import pytest
from utils.fetch import (
    fetch_pdb,
    fetch_ligand_by_name,
    fetch_ligand_by_cid,
    extract_uniprot_accession,
    validate_uniprot_accession,
    fetch_alphafold,
)


def test_invalid_pdb_id():
    with pytest.raises(ValueError):
        fetch_pdb("INVALID_ID_TOO_LONG")


def test_invalid_ligand_name():
    with pytest.raises(ValueError):
        fetch_ligand_by_name("   ")


def test_invalid_cid():
    with pytest.raises(ValueError):
        fetch_ligand_by_cid("NOT_A_NUMBER")


def test_extract_uniprot_accession():
    assert extract_uniprot_accession("Q9KPW0") == "Q9KPW0"
    assert extract_uniprot_accession("q9kpw0") == "Q9KPW0"
    assert extract_uniprot_accession("https://alphafold.ebi.ac.uk/entry/Q9KPW0") == "Q9KPW0"
    assert extract_uniprot_accession("https://alphafold.ebi.ac.uk/entry/Q9KPW0/") == "Q9KPW0"
    assert extract_uniprot_accession("https://www.uniprot.org/uniprotkb/Q9KPW0/entry") == "Q9KPW0"
    assert extract_uniprot_accession("P04637") == "P04637"
    assert extract_uniprot_accession("A0A024RBG1") == "A0A024RBG1"
    assert extract_uniprot_accession("not_an_accession") is None
    assert extract_uniprot_accession("") is None


def test_validate_uniprot_accession():
    assert validate_uniprot_accession("Q9KPW0") is True
    assert validate_uniprot_accession("P04637") is True
    assert validate_uniprot_accession("A0A024RBG1") is True
    assert validate_uniprot_accession("INVALID") is False
    assert validate_uniprot_accession("12345") is False
    assert validate_uniprot_accession("") is False


def test_fetch_alphafold_invalid_format():
    with pytest.raises(ValueError, match="Invalid UniProt accession"):
        fetch_alphafold("NOT_A_VALID_ACCESSION_STRING")


def test_fetch_alphafold_not_found():
    # Q00000 returns 404 from AlphaFold DB
    with pytest.raises(ValueError, match="not found in the AlphaFold Database"):
        fetch_alphafold("Q00000")


def test_fetch_alphafold_live_q9kpw0(tmp_path):
    dest = str(tmp_path / "test_af.pdb")
    try:
        res = fetch_alphafold("https://alphafold.ebi.ac.uk/entry/Q9KPW0", output_path=dest)
        assert os.path.isfile(res["output_path"])
        assert res["accession"] == "Q9KPW0"
        assert res["entry_id"] == "AF-Q9KPW0-F1"
        assert res["gene"] == "bamA"
        assert res["is_alphafold"] is True
        assert res["mean_plddt"] > 50.0
        assert "plddt_very_high_pct" in res
    except (urllib.error.URLError, RuntimeError) as e:
        pytest.skip(f"External network unavailable for AlphaFold API: {e}")


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
