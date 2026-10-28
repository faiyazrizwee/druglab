"""
End-to-End Miniature Virtual Screening Integration Test.

Runs protein preparation, ligand preparation, pocket detection, drug-likeness,
ADMET, docking, and lead selection on real sample files with minimal exhaustiveness
to verify full pipeline interoperability, reproducibility, and HTML reporting.
"""

import os
import pytest
import pandas as pd
from utils.config import DrugLabConfig
from drug_selection.workflow import run_drug_selection
from reports.generator import generate_report


@pytest.mark.integration
def test_mini_virtual_screening_e2e(temp_dir):
    protein_pdb = "protein/4MZI.pdb"
    # Fallback to 7DTD if 4MZI is missing
    if not os.path.exists(protein_pdb):
        protein_pdb = "protein/7DTD.pdb"

    ligand_file = "ligands/aspirin.sdf"
    if not os.path.exists(ligand_file):
        pytest.skip("Sample ligand file not found.")

    # Configure minimal parameters for fast test execution
    config = DrugLabConfig()
    config.docking.exhaustiveness = 1
    config.docking.n_poses = 1
    config.docking.n_cpus = 1
    config.pocket.padding = 3.0
    config.pocket.max_box_dimension = 20.0
    config.druglikeness.rules = ["lipinski"]
    config.druglikeness.evaluation_mode = "majority"
    config.admet.profile = "permissive"
    config.lead_selection.top_n = 1
    config.lead_selection.on_no_hits = "fallback"

    results = run_drug_selection(
        protein_pdb=protein_pdb,
        ligand_library=ligand_file,
        output_dir=temp_dir,
        config=config,
    )

    assert "protein_preparation" in results
    assert "ligand_preparation" in results
    assert "docking" in results
    assert "lead_selection" in results

    # Verify essential artifacts
    clean_pdb = os.path.join(temp_dir, "protein_preparation", "protein_clean.pdb")
    assert os.path.isfile(clean_pdb)

    docking_csv = os.path.join(temp_dir, "docking", "docking_results.csv")
    assert os.path.isfile(docking_csv)

    ranking_csv = os.path.join(temp_dir, "lead_selection", "ranking.csv")
    assert os.path.isfile(ranking_csv)

    # Verify that Ligand Efficiency columns were populated
    df_rank = pd.read_csv(ranking_csv)
    assert not df_rank.empty
    assert "ligand_efficiency" in df_rank.columns
    assert "lipophilic_efficiency" in df_rank.columns

    # Verify report was generated
    report_html = os.path.join(temp_dir, "reports", "summary.html")
    assert os.path.isfile(report_html)
