"""
Unit tests for CLI commands and backwards compatibility of `druglab run`.
"""

import sys
from unittest.mock import patch, MagicMock
import pytest

import druglab


def test_cli_run_small_molecule_default(monkeypatch):
    """Ensure `druglab run` defaults to small-molecule workflow without breaking existing flags."""
    mock_run_drug_selection = MagicMock()
    monkeypatch.setattr("drug_selection.workflow.run_drug_selection", mock_run_drug_selection)
    monkeypatch.setattr("utils.logger.print_banner", lambda: None)

    test_args = ["druglab", "run", "--protein", "protein/3DWT.pdb", "--library", "ligands/"]
    monkeypatch.setattr(sys, "argv", test_args)

    druglab.main()
    assert mock_run_drug_selection.called
    kwargs = mock_run_drug_selection.call_args[1]
    assert kwargs["protein_pdb"] == "protein/3DWT.pdb"
    assert kwargs["ligand_library"] == "ligands/"


def test_cli_run_mode_peptide(monkeypatch):
    """Ensure `druglab run --mode peptide` dispatches to peptide pipeline."""
    mock_peptide_pipeline = MagicMock(return_value={"run_dir": "results/test", "report_html": None})
    monkeypatch.setattr("core.peptide_docking.pipeline.run_peptide_docking_pipeline", mock_peptide_pipeline)

    test_args = [
        "druglab", "run",
        "--mode", "peptide",
        "--protein", "protein/3DWT.pdb",
        "--peptide-seq", "ACDEF",
        "--preset", "fast",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    druglab.main()
    assert mock_peptide_pipeline.called
    kwargs = mock_peptide_pipeline.call_args[1]
    assert kwargs["protein_input"] == "protein/3DWT.pdb"
    assert kwargs["peptide_sequence"] == "ACDEF"


def test_cli_peptide_subcommands(monkeypatch):
    """Ensure `druglab peptide run` subcommand functions identically."""
    mock_peptide_pipeline = MagicMock(return_value={"run_dir": "results/test", "report_html": None})
    monkeypatch.setattr("core.peptide_docking.pipeline.run_peptide_docking_pipeline", mock_peptide_pipeline)

    test_args = [
        "druglab", "peptide", "run",
        "--protein", "protein/3DWT.pdb",
        "--sequence", "WYVIK",
        "--admet-provider", "peplife",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    druglab.main()
    assert mock_peptide_pipeline.called
    kwargs = mock_peptide_pipeline.call_args[1]
    assert kwargs["protein_input"] == "protein/3DWT.pdb"
    assert kwargs["peptide_sequence"] == "WYVIK"
    assert kwargs["config"].admet.provider == "peplife"
