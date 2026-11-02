"""
Unit tests for structure component inspection and transparent review proposals.
"""

import os
import pytest
from core.structure_inspection import (
    inspect_structure_components,
    is_alphafold_model,
    calculate_plddt_stats,
)


def test_is_alphafold_model():
    assert is_alphafold_model("protein/7DTD.pdb") is False
    assert is_alphafold_model("protein/7NRE.pdb") is False
    assert is_alphafold_model("protein/AF_Q9KPW0.pdb") is True
    assert is_alphafold_model("protein/AF-P04637-F1-model_v4.pdb") is True


def test_inspect_7nre_small_molecule():
    pdb_path = "protein/7NRE.pdb"
    if not os.path.exists(pdb_path):
        pytest.skip("protein/7NRE.pdb not available")

    res = inspect_structure_components(pdb_path, workflow_mode="small_molecule")
    assert res["is_alphafold"] is False
    assert res["structure_source"] == "Experimental PDB (X-ray / Cryo-EM / NMR)"
    assert set(res["chain_ids"]) == {"A", "C"}
    assert res["waters_count"] > 100
    assert "MG" in res["metals"]
    assert "UX8" in res["ligands"] or "C8E" in res["ligands"]

    # Review items
    items_by_id = {it["id"]: it for it in res["review_items"]}
    assert "waters" in items_by_id
    assert "metals" in items_by_id
    assert "ligands" in items_by_id

    # In small molecule mode, metals proposed to keep for active-site coordination
    assert items_by_id["metals"]["proposed_action"] == "keep"
    assert items_by_id["waters"]["proposed_action"] == "remove"
    assert items_by_id["ligands"]["proposed_action"] == "remove"

    # Scientific rationale and consequences are clearly documented
    for it in res["review_items"]:
        assert len(it["scientific_rationale"]) > 10
        assert len(it["docking_consequences"]) > 10


def test_inspect_7nre_peptide_docking():
    pdb_path = "protein/7NRE.pdb"
    if not os.path.exists(pdb_path):
        pytest.skip("protein/7NRE.pdb not available")

    res = inspect_structure_components(pdb_path, workflow_mode="peptide")
    items_by_id = {it["id"]: it for it in res["review_items"]}

    # In peptide mode with LightDock, metals proposed to remove due to scoring force field limitations
    assert items_by_id["metals"]["proposed_action"] == "remove"
    assert "LightDock" in items_by_id["metals"]["scientific_rationale"] or "force field" in items_by_id["metals"]["scientific_rationale"]
    assert items_by_id["waters"]["proposed_action"] == "remove_bulk"


def test_inspect_alphafold_model(tmp_path):
    # Create minimal mock AlphaFold PDB structure with pLDDT in B-factor column
    mock_af_pdb = tmp_path / "AF_mock.pdb"
    mock_content = (
        "HEADER    ALPHAFOLD PREDICTION MODEL\n"
        "TITLE     MODEL OF TEST PROTEIN\n"
        "ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 95.00           N\n"
        "ATOM      2  CA  ALA A   1      11.000  10.000  10.000  1.00 95.00           C\n"
        "ATOM      3  C   ALA A   1      11.500  11.000  10.000  1.00 95.00           C\n"
        "ATOM      4  O   ALA A   1      11.000  12.000  10.000  1.00 95.00           O\n"
        "ATOM      5  N   GLY A   2      12.500  11.000  10.000  1.00 45.00           N\n"
        "ATOM      6  CA  GLY A   2      13.000  12.000  10.000  1.00 45.00           C\n"
        "ATOM      7  C   GLY A   2      14.000  12.000  10.000  1.00 45.00           C\n"
        "ATOM      8  O   GLY A   2      14.500  13.000  10.000  1.00 45.00           O\n"
        "END\n"
    )
    with open(mock_af_pdb, "w") as f:
        f.write(mock_content)

    res = inspect_structure_components(mock_af_pdb, workflow_mode="small_molecule")
    assert res["is_alphafold"] is True
    assert res["alphafold_caveats"] is not None
    assert "AlphaFold Database" in res["alphafold_caveats"]["disclaimer"]

    # Verify pLDDT distribution: 1 very high (>90), 1 very low (<50)
    plddt = res["alphafold_caveats"]["plddt_stats"]
    assert plddt["has_plddt"] is True
    assert plddt["total_evaluated_residues"] == 2
    assert plddt["very_high_count"] == 1
    assert plddt["very_low_count"] == 1
    assert plddt["mean_plddt"] == 70.0
