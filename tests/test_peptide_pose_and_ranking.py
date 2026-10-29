"""
Unit tests for structural pose analysis and ranking tables.
"""

from pathlib import Path
from core.peptide_docking.input_validation import PeptideCandidate
from core.peptide_docking.pose_analysis import analyze_docked_pose
from core.peptide_docking.ranking import rank_peptide_candidates
from core.peptide_docking.peptide_preparation import generate_peptide_3d_structure


def test_analyze_docked_pose(tmp_path):
    # Create complex PDB with Chain A (receptor) and Chain B (peptide)
    rec_pdb = tmp_path / "rec.pdb"
    pep_pdb = tmp_path / "pep.pdb"
    complex_pdb = tmp_path / "complex.pdb"

    generate_peptide_3d_structure("ACDEF", rec_pdb, chain_id="A")
    generate_peptide_3d_structure("WYV", pep_pdb, chain_id="B")

    with open(complex_pdb, "w") as out_f:
        out_f.write(rec_pdb.read_text())
        out_f.write(pep_pdb.read_text())

    metrics = analyze_docked_pose(
        pose_pdb_path=complex_pdb,
        receptor_chain="A",
        peptide_chain="B",
    )
    assert "clashes_count" in metrics
    assert "contacts_count" in metrics
    assert "hbonds_count" in metrics
    assert "salt_bridges_count" in metrics
    assert "interface_receptor_residues" in metrics


def test_rank_peptide_candidates(tmp_path):
    c1 = PeptideCandidate(id="p1", sequence="ACDEF", source="sequence", status="completed")
    c1.docking_results = [{
        "rank": 1,
        "swarm_id": 0,
        "glowworm_id": 1,
        "score": -25.4,
        "scoring_function": "fastdfire",
        "contacts_count": 15,
        "hbonds_count": 3,
        "clashes_count": 0,
        "status": "valid",
    }]

    c2 = PeptideCandidate(id="p2", sequence="KLMNP", source="sequence", status="completed")
    c2.docking_results = [{
        "rank": 1,
        "swarm_id": 1,
        "glowworm_id": 2,
        "score": -10.2,
        "scoring_function": "fastdfire",
        "contacts_count": 8,
        "hbonds_count": 1,
        "clashes_count": 2,
        "status": "valid",
    }]

    c3 = PeptideCandidate(id="p3", sequence="WYV", source="sequence", status="filtered", failure_reason="too short")

    df_cands, df_poses = rank_peptide_candidates([c1, c2, c3], output_dir=tmp_path)

    assert len(df_cands) == 3
    # Top ranked should be c1 because -25.4 < -10.2
    assert df_cands.iloc[0]["candidate_id"] == "p1"
    assert df_cands.iloc[1]["candidate_id"] == "p2"
    assert df_cands.iloc[2]["status"] == "filtered"
    assert (tmp_path / "docking_results.tsv").is_file()
    assert (tmp_path / "pose_metrics.tsv").is_file()
