"""
Tests for HTML and JSON Report Generator.
"""

import os
import json
import pandas as pd
from utils.io import ensure_dir, write_csv
from utils.quarantine import quarantine_item
from reports.generator import generate_report


def test_generate_report_full(temp_dir):
    # Setup mock output tree
    dock_dir = os.path.join(temp_dir, "docking")
    lead_dir = os.path.join(temp_dir, "lead_selection")
    ensure_dir(dock_dir)
    ensure_dir(lead_dir)

    dock_df = pd.DataFrame([
        {
            "name": "compound_1",
            "smiles": "CC(=O)Oc1ccccc1C(=O)O",
            "best_affinity": -7.5,
            "ligand_efficiency": 0.577,
            "lipophilic_efficiency": 4.28,
            "MW": 180.16,
            "LogP": 1.19,
            "composite_score": 0.88,
            "rank": 1,
        },
        {
            "name": "compound_2",
            "smiles": "CC(=O)Nc1ccc(O)cc1",
            "best_affinity": -6.2,
            "ligand_efficiency": 0.564,
            "lipophilic_efficiency": 4.07,
            "MW": 151.16,
            "LogP": 0.46,
            "composite_score": 0.76,
            "rank": 2,
        },
    ])
    write_csv(dock_df, os.path.join(dock_dir, "docking_results.csv"))
    write_csv(dock_df, os.path.join(lead_dir, "ranking.csv"))
    write_csv(dock_df, os.path.join(lead_dir, "top_hits.csv"))

    # Add a quarantine record
    quarantine_item(temp_dir, "ligand_prep", "bad_mol", "Parsing error", "C123")

    report_html = generate_report(output_dir=temp_dir)
    assert os.path.isfile(report_html)

    # Check that file has HTML content and key elements
    with open(report_html, "r", encoding="utf-8") as f:
        content = f.read()

    assert "<!DOCTYPE html>" in content
    assert "compound_1" in content
    assert "Quarantined Molecule Diagnostics" in content
    assert "bad_mol" in content
    assert ".chart-grid" in content

    # Check JSON summary
    summary_json = os.path.join(temp_dir, "reports", "summary_report.json")
    assert os.path.isfile(summary_json)
    with open(summary_json, "r", encoding="utf-8") as jf:
        data = json.load(jf)
    assert data["kpis"]["top_lead_name"] == "compound_1"
    assert data["quarantined_count"] == 1


def test_3d_viewer_and_chart_controls(temp_dir):
    dock_dir = os.path.join(temp_dir, "docking")
    poses_dir = os.path.join(dock_dir, "poses")
    dl_dir = os.path.join(temp_dir, "druglikeness")
    ensure_dir(poses_dir)
    ensure_dir(dl_dir)

    # Mock PDBQT pose file with 2 models
    sample_pdbqt = """MODEL 1
REMARK VINA RESULT:    -8.500      0.000      0.000
ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00    +0.000 C
ENDMDL
MODEL 2
REMARK VINA RESULT:    -7.800      1.200      2.100
ATOM      1  C   UNL     1       1.000   1.000   1.000  1.00  0.00    +0.000 C
ENDMDL
"""
    with open(os.path.join(poses_dir, "aspirin_poses.pdbqt"), "w") as f:
        f.write(sample_pdbqt)

    # Mock druglikeness descriptors
    dl_df = pd.DataFrame([
        {"name": "aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O", "MW": 180.16, "LogP": 1.19, "status": "PASS"}
    ])
    write_csv(dl_df, os.path.join(dl_dir, "descriptors.csv"))

    report_html = generate_report(output_dir=temp_dir)
    with open(report_html, "r", encoding="utf-8") as f:
        content = f.read()

    # Verify 3D controls and data
    assert "aspirin" in content
    assert "btn_bg_mode" in content
    assert "toggleBackground" in content
    assert "Cartoon + Surface" in content
    assert "Ball &amp; Stick" in content
    assert "chk_pocket_residues" in content
    assert "focusLigand" in content
    # Verify chemical space grid and rules
    assert "Chemical Space (MW vs cLogP)" in content
    assert "Lipinski MW" in content
    assert "Lipinski cLogP" in content
    assert "gridcolor" in content


def test_generate_report_empty(temp_dir):
    # Tests report generation with empty/minimal output directory
    report_html = generate_report(output_dir=temp_dir)
    assert os.path.isfile(report_html)
    with open(report_html, "r", encoding="utf-8") as f:
        content = f.read()
    assert "<!DOCTYPE html>" in content


def test_generate_summary_figures(temp_dir):
    from reports.generator import generate_summary_figures
    counts = {
        "Input Ligands": 50,
        "Drug-Like": 40,
        "ADMET Passed": 30,
        "Docked": 30,
        "Selected Leads": 5,
    }
    figs = generate_summary_figures(counts, temp_dir)
    assert len(figs) == 1
    assert os.path.isfile(figs[0])
    assert os.path.getsize(figs[0]) > 0

