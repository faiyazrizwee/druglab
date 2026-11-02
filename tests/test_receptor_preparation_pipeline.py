"""
Unit tests for transparent receptor preparation pipeline and reporting.
"""

import os
import hashlib
import pytest
from core.structure_inspection import inspect_structure_components
from core.structure_review_gui import review_structure_preparation
from core.receptor_preparation_pipeline import prepare_receptor_transparently


def _checksum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def test_prepare_receptor_transparently_small_molecule(tmp_path):
    orig_path = "protein/7NRE.pdb"
    if not os.path.exists(orig_path):
        pytest.skip(f"{orig_path} not found")

    initial_sum = _checksum(orig_path)
    out_dir = str(tmp_path / "sm_prep")

    insp = inspect_structure_components(orig_path, workflow_mode="small_molecule")
    decisions = review_structure_preparation(orig_path, insp, launch_gui=False, auto_confirm=True)

    res = prepare_receptor_transparently(
        input_pdb=orig_path,
        output_dir=out_dir,
        workflow_mode="small_molecule",
        decisions=decisions,
        inspection_data=insp,
    )

    # 1. Original file is unchanged
    assert _checksum(orig_path) == initial_sum

    # 2. Output files exist
    assert os.path.isfile(res["prepared_path"])
    assert res["prepared_path"].endswith("_prepared.pdb")
    assert os.path.isfile(res["pdbqt_path"])
    assert os.path.isfile(res["report_html"])
    assert os.path.isfile(res["report_json"])
    assert os.path.isfile(res["report_tsv"])

    # 3. HTML report contains audit details
    with open(res["report_html"], "r", encoding="utf-8") as f:
        html = f.read()
    assert "DrugLab Receptor Preparation Report" in html
    assert "Component Preparation Decisions" in html
    assert "Original Structure (Unmodified)" in html

    # 4. TSV report is formatted correctly
    with open(res["report_tsv"], "r", encoding="utf-8") as f:
        tsv_lines = f.readlines()
    assert tsv_lines[0].startswith("Component\tAction\tScientific_Rationale\tDocking_Impact")
    assert len(tsv_lines) > 1


def test_prepare_receptor_transparently_peptide(tmp_path):
    orig_path = "protein/7NRE.pdb"
    if not os.path.exists(orig_path):
        pytest.skip(f"{orig_path} not found")

    initial_sum = _checksum(orig_path)
    out_dir = str(tmp_path / "pep_prep")

    insp = inspect_structure_components(orig_path, workflow_mode="peptide")
    decisions = review_structure_preparation(orig_path, insp, launch_gui=False, auto_confirm=True)

    res = prepare_receptor_transparently(
        input_pdb=orig_path,
        output_dir=out_dir,
        workflow_mode="peptide",
        decisions=decisions,
        inspection_data=insp,
    )

    # Original file is unchanged
    assert _checksum(orig_path) == initial_sum

    # Output files exist
    assert os.path.isfile(res["prepared_path"])
    assert os.path.isfile(res["report_html"])
    assert os.path.isfile(res["report_json"])
    assert os.path.isfile(res["report_tsv"])
    # Peptide mode does not require PDBQT
    assert res.get("pdbqt_path") is None


def test_prepare_receptor_chain_selection_options(tmp_path):
    orig_path = "protein/7NRE.pdb"
    if not os.path.exists(orig_path):
        pytest.skip(f"{orig_path} not found")

    import gemmi
    insp = inspect_structure_components(orig_path, workflow_mode="peptide")

    # 1. Select Chain A only
    out_dir_a = str(tmp_path / "chain_a_only")
    res_a = prepare_receptor_transparently(
        input_pdb=orig_path,
        output_dir=out_dir_a,
        workflow_mode="peptide",
        decisions={"selected_chains": ["A"]},
        inspection_data=insp,
    )
    st_a = gemmi.read_structure(res_a["prepared_path"])
    assert [c.name for c in st_a[0]] == ["A"]

    # 2. Select Chain C only
    out_dir_c = str(tmp_path / "chain_c_only")
    res_c = prepare_receptor_transparently(
        input_pdb=orig_path,
        output_dir=out_dir_c,
        workflow_mode="peptide",
        decisions={"selected_chains": ["C"]},
        inspection_data=insp,
    )
    st_c = gemmi.read_structure(res_c["prepared_path"])
    assert [c.name for c in st_c[0]] == ["C"]

    # 3. Select both Chain A and Chain C
    out_dir_ac = str(tmp_path / "chain_ac")
    res_ac = prepare_receptor_transparently(
        input_pdb=orig_path,
        output_dir=out_dir_ac,
        workflow_mode="peptide",
        decisions={"selected_chains": ["A", "C"]},
        inspection_data=insp,
    )
    st_ac = gemmi.read_structure(res_ac["prepared_path"])
    assert set([c.name for c in st_ac[0]]) == {"A", "C"}

