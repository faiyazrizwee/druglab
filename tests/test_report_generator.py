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

    # Check JSON summary
    summary_json = os.path.join(temp_dir, "reports", "summary_report.json")
    assert os.path.isfile(summary_json)
    with open(summary_json, "r", encoding="utf-8") as jf:
        data = json.load(jf)
    assert data["kpis"]["top_lead_name"] == "compound_1"
    assert data["quarantined_count"] == 1


def test_generate_report_empty(temp_dir):
    # Tests report generation with empty/minimal output directory
    report_html = generate_report(output_dir=temp_dir)
    assert os.path.isfile(report_html)
    with open(report_html, "r", encoding="utf-8") as f:
        content = f.read()
    assert "<!DOCTYPE html>" in content
