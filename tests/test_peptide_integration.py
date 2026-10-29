"""
Real LightDock end-to-end integration smoke test.
Marked with @pytest.mark.integration (deselected during fast CI with '-m "not integration"').
"""

import os
import shutil
import pytest
from pathlib import Path

from core.peptide_docking.config import load_peptide_config
from core.peptide_docking.pipeline import run_peptide_docking_pipeline
from core.peptide_docking.lightdock_runner import find_lightdock_executable


@pytest.mark.integration
def test_real_lightdock_smoke_docking(tmp_path):
    """
    Execute a minimal real LightDock docking simulation with 1 swarm and 2 steps.
    Validates end-to-end process execution, output file generation, and HTML report.
    """
    try:
        find_lightdock_executable("lightdock3.py")
    except FileNotFoundError:
        pytest.skip("LightDock is not installed in the environment.")

    # Target protein fixture: protein/3DWT.pdb or synthesize minimal receptor
    target_pdb = Path("protein/3DWT.pdb")
    if not target_pdb.is_file():
        pytest.skip("Receptor fixture protein/3DWT.pdb not present.")

    # Configure ultra-fast smoke test
    cfg = load_peptide_config(
        overrides={
            "lightdock": {
                "preset": "fast",
                "swarms": 1,
                "steps": 2,
                "glowworms": 5,
                "scoring_function": "fastdfire",
            },
            "output": {"directory": str(tmp_path)},
        }
    )

    res = run_peptide_docking_pipeline(
        protein_input=str(target_pdb),
        peptide_sequence="ACDEF",
        output_dir=str(tmp_path),
        config=cfg,
    )

    assert "run_dir" in res
    run_dir = Path(res["run_dir"])
    assert run_dir.is_dir()

    manifest_file = run_dir / "run_manifest.json"
    assert manifest_file.is_file()

    report_file = run_dir / "report.html"
    assert report_file.is_file()

    results_tsv = run_dir / "docking_results.tsv"
    assert results_tsv.is_file()


@pytest.mark.integration
def test_multi_peptide_fasta_docking_pipeline(tmp_path):
    """
    Validate end-to-end pipeline ingestion and screening of a multi-peptide FASTA file.
    """
    try:
        find_lightdock_executable("lightdock3.py")
    except FileNotFoundError:
        pytest.skip("LightDock is not installed in the environment.")

    target_pdb = Path("protein/3DWT.pdb")
    if not target_pdb.is_file():
        pytest.skip("Receptor fixture protein/3DWT.pdb not present.")

    # Create multi-peptide FASTA
    fasta_file = tmp_path / "multi_peptides.fasta"
    fasta_file.write_text("""
>peptide_lead_1 First candidate
ACDEF
>peptide_lead_2 Second candidate
WYVIK
""")

    cfg = load_peptide_config(
        overrides={
            "lightdock": {
                "preset": "fast",
                "swarms": 1,
                "steps": 2,
                "glowworms": 5,
                "scoring_function": "fastdfire",
            },
            "output": {"directory": str(tmp_path)},
        }
    )

    res = run_peptide_docking_pipeline(
        protein_input=str(target_pdb),
        peptide_fasta=str(fasta_file),
        output_dir=str(tmp_path),
        config=cfg,
    )

    assert res["manifest"]["total_peptides"] == 2
    assert res["manifest"]["completed_docking"] == 2
    c_df = res["candidates_df"]
    assert len(c_df) == 2
    assert set(c_df["candidate_id"]) == {"peptide_lead_1", "peptide_lead_2"}
