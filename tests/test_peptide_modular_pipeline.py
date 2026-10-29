"""
Unit and integration tests for DrugLab 2.0 modular peptide pipeline additions:
- Protein preparation (biological assemblies, inspection, selective water/metal/ligand retention)
- Peptide preparation (geometry checks, zwitterionic default, terminal caps, cyclization)
- Pocket detection (3D Box GUI tuner conversion to LightDock restraints)
- Lead selection (multi-objective ranking)
- Interactive reporting controls
"""

from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import pytest

from core.peptide_docking.protein_preparation import (
    inspect_protein_structure,
    prepare_receptor_for_lightdock,
)
from core.peptide_docking.peptide_preparation import (
    check_peptide_geometry_and_stereochemistry,
    apply_terminal_chemistry,
    prepare_peptide_candidate,
)
from core.peptide_docking.input_validation import PeptideCandidate
from core.peptide_docking.pocket_detection import (
    extract_residues_in_bounding_box,
    convert_pocket_to_lightdock_restraints,
    interactive_peptide_pocket_detection,
)
from core.peptide_docking.lead_selection import rank_and_select_peptide_leads
from core.peptide_docking.reporting import generate_peptide_html_report


@pytest.fixture
def sample_receptor_pdb():
    path = Path("protein/7DTD.pdb")
    if not path.is_file():
        pytest.skip("protein/7DTD.pdb not found")
    return str(path.resolve())


def test_inspect_protein_structure(sample_receptor_pdb):
    insp = inspect_protein_structure(sample_receptor_pdb)
    assert "available_chain_ids" in insp
    assert len(insp["available_chain_ids"]) >= 1
    assert "total_residues" in insp
    assert insp["total_residues"] > 0
    assert "missing_residues_count" in insp
    assert "alternate_conformations_count" in insp


def test_prepare_receptor_for_lightdock_selective_retention(sample_receptor_pdb, tmp_path):
    out_dir = tmp_path / "prep_prot"
    res = prepare_receptor_for_lightdock(
        input_pdb=sample_receptor_pdb,
        output_dir=out_dir,
        chains=["B"],
        water_policy="remove_bulk",
        keep_metals=True,
        keep_cofactors=True,
    )
    assert os.path.isfile(res["prepared_path"])
    assert "B" in res["chains"]
    assert res["prepared_checksum"] is not None


def test_peptide_geometry_and_stereochemistry(tmp_path):
    from core.peptide_docking.peptide_preparation import generate_peptide_3d_structure
    res = generate_peptide_3d_structure("ACDEFGH", tmp_path / "test_pep.pdb")
    geom = res["geometry"]
    assert geom["valid_geometry"] is True
    assert geom["cis_peptide_bonds"] == 0
    assert geom["steric_clashes_count"] == 0


def test_peptide_terminal_chemistry_zwitterion():
    from rdkit import Chem
    mol = Chem.MolFromSequence("ACDEFGH")
    mod_mol, chem_desc = apply_terminal_chemistry(mol, sequence_length=7, n_term_cap="zwitterion", c_term_cap="zwitterion")
    assert "Zwitterionic" in chem_desc or "zwitterion" in chem_desc.lower()


def test_peptide_preparation_conformer_and_pdb(tmp_path):
    cand = PeptideCandidate(id="pep_prep_test", sequence="ACDEFGH", source="sequence")
    res = prepare_peptide_candidate(
        candidate=cand,
        output_dir=tmp_path / "pep_out",
        n_term_cap="ace",
        c_term_cap="nhe",
    )
    assert res.prepared_structure_path is not None
    assert os.path.isfile(res.prepared_structure_path)
    term_chem = res.properties.get("terminal_chemistry", "")
    assert "ACE" in term_chem or "Acetylated" in term_chem
    assert "NHE" in term_chem or "Amidated" in term_chem


def test_pocket_to_lightdock_restraints_conversion(sample_receptor_pdb, tmp_path):
    mock_pocket = {
        "center": [111.0, 113.9, 115.9],
        "size": [30.0, 30.0, 30.0],
    }
    restr_path, restr_list = convert_pocket_to_lightdock_restraints(
        receptor_pdb=sample_receptor_pdb,
        selected_pocket=mock_pocket,
        output_dir=tmp_path / "pocket_out",
    )
    assert os.path.isfile(restr_path)
    assert len(restr_list) > 0
    assert all("." in r for r in restr_list)
    with open(restr_path, "r", encoding="utf-8") as f:
        file_lines = [l.strip() for l in f if l.startswith("R ")]
    assert len(file_lines) == len(restr_list)


def test_interactive_peptide_pocket_detection_auto_confirm(sample_receptor_pdb, tmp_path):
    res = interactive_peptide_pocket_detection(
        receptor_pdb=sample_receptor_pdb,
        output_dir=tmp_path / "pkt_auto",
        auto_confirm=True,
    )
    assert res.get("mode") in ("pocket_guided", "global")
    assert res.get("restraints_path") is not None
    assert os.path.isfile(res["restraints_path"])


def test_rank_and_select_peptide_leads(tmp_path):
    cands_df = pd.DataFrame([
        {
            "candidate_id": "pep_A",
            "sequence": "ACDEFGH",
            "best_lightdock_score": 12.5,
            "contacts_count": 22,
            "hbonds_count": 5,
            "clashes_count": 0,
            "peptide_toxicity_risk": "Low risk (non-hemolytic / safe)",
        },
        {
            "candidate_id": "pep_B",
            "sequence": "WYVIKAA",
            "best_lightdock_score": 8.0,
            "contacts_count": 10,
            "hbonds_count": 1,
            "clashes_count": 3,
            "peptide_toxicity_risk": "High hemolytic risk",
        },
    ])
    poses_df = pd.DataFrame([
        {"candidate_id": "pep_A", "pose_rank": 1, "lightdock_score": 12.5},
        {"candidate_id": "pep_B", "pose_rank": 1, "lightdock_score": 8.0},
    ])

    leads_df = rank_and_select_peptide_leads(
        candidates_df=cands_df,
        poses_df=poses_df,
        output_dir=tmp_path / "leads_out",
        top_n=2,
    )
    assert not leads_df.empty
    assert len(leads_df) == 2
    assert "composite_lead_score" in leads_df.columns
    # Lower composite score is better
    assert leads_df.iloc[0]["candidate_id"] == "pep_A"
    assert (tmp_path / "leads_out" / "lead_peptides.csv").is_file()


def test_report_generation_with_leads_and_viewer_controls(tmp_path):
    manifest = {
        "run_id": "test_run_123",
        "title": "Unit Test Report",
        "lightdock_version": "0.9.4",
        "preset": "fast",
        "scoring_function": "fastdfire",
    }
    cands_df = pd.DataFrame([
        {
            "candidate_id": "pep_lead",
            "sequence": "ACDEFGH",
            "length": 7,
            "best_lightdock_score": 12.5,
            "contacts": 15,
            "hbonds": 4,
            "clashes": 0,
            "status": "completed",
        }
    ])
    leads_df = pd.DataFrame([
        {
            "rank": 1,
            "candidate_id": "pep_lead",
            "sequence": "ACDEFGH",
            "composite_score": 5.4,
            "lightdock_score": 12.5,
            "contacts": 15,
            "hbonds": 4,
            "clashes": 0,
            "hemolytic_risk": "Low (Safe)",
            "selection_notes": "Ideal binding geometry",
        }
    ])
    rep_path = generate_peptide_html_report(
        output_dir=tmp_path,
        manifest=manifest,
        candidates_df=cands_df,
        leads_df=leads_df,
    )
    assert os.path.isfile(rep_path)
    with open(rep_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Focus Peptide" in content
    assert "updateComplexStyles" in content
    assert "Top Lead Candidates" in content
    assert "pep_lead" in content
