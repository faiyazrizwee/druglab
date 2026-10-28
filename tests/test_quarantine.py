"""
Tests for Quarantine Logger.
"""

import os
from utils.quarantine import (
    quarantine_item,
    load_quarantine,
    get_quarantine_summary,
    get_quarantine_path,
)


def test_quarantine_logging(temp_dir):
    # Log two items
    quarantine_item(
        output_dir=temp_dir,
        stage="ligand_preparation",
        mol_id="bad_ligand_1",
        reason="Sanitization error: invalid valence",
        smiles="C(=O)(C)(C)(C)(C)",
    )
    quarantine_item(
        output_dir=temp_dir,
        stage="docking",
        mol_id="bad_ligand_2",
        reason="Vina crash: zero conformers",
        details={"pdbqt": "bad_ligand_2.pdbqt"},
    )

    q_file = get_quarantine_path(temp_dir)
    assert os.path.isfile(q_file)

    records = load_quarantine(temp_dir)
    assert len(records) == 2
    assert records[0]["mol_id"] == "bad_ligand_1"
    assert records[0]["stage"] == "ligand_preparation"
    assert records[1]["stage"] == "docking"

    summary = get_quarantine_summary(temp_dir)
    assert summary["total_quarantined"] == 2
    assert summary["by_stage"]["ligand_preparation"] == 1
    assert summary["by_stage"]["docking"] == 1
