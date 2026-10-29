"""
Unit tests for interactive HTML report generation.
"""

from pathlib import Path
import pandas as pd
from core.peptide_docking.reporting import generate_peptide_html_report


def test_generate_peptide_html_report(tmp_path):
    manifest = {
        "run_id": "test_run_123",
        "title": "Unit Test Peptide Report",
        "total_peptides": 2,
        "passed_filtering": 1,
        "completed_docking": 1,
        "lightdock_version": "0.9.4",
        "scoring_function": "fastdfire",
        "preset": "fast",
        "runtime_sec": 12.34,
    }

    cands_df = pd.DataFrame([{
        "candidate_id": "pep_1",
        "sequence": "ACDEFGH",
        "length": 7,
        "best_lightdock_score": -18.5,
        "scoring_function": "fastdfire",
        "contacts": 14,
        "hbonds": 2,
        "clashes": 0,
        "status": "completed",
    }])

    admet_df = pd.DataFrame([{
        "candidate_id": "pep_1",
        "sequence": "ACDEFGH",
        "molecular_weight_da": 780.2,
        "isoelectric_point_pi": 4.5,
        "net_charge_ph74": -1.2,
        "gravy_hydropathicity": 0.1,
        "instability_index": 25.0,
        "predicted_stability": "stable",
        "boman_index_kcal_mol": 1.5,
    }])

    report_path = generate_peptide_html_report(
        output_dir=tmp_path,
        manifest=manifest,
        candidates_df=cands_df,
        admet_df=admet_df,
        output_filename="test_report.html",
    )

    assert Path(report_path).is_file()
    html_text = Path(report_path).read_text()
    assert "DrugLab 2.0 — Protein–Peptide Docking Report" in html_text
    assert "pep_1" in html_text
    assert "ACDEFGH" in html_text
    assert "-18.50" in html_text
    assert "3Dmol" in html_text
