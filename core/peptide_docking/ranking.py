"""
Candidate and pose ranking for peptide docking.
Combines LightDock native scoring with structural pose validation metrics.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import pandas as pd

from core.peptide_docking.input_validation import PeptideCandidate
from utils.io import ensure_dir
from utils.logger import setup_logger

logger = setup_logger("peptide_ranking", level="INFO")


def rank_peptide_candidates(
    candidates: list[PeptideCandidate],
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rank peptide candidates and their docking poses.

    Generates:
        1. docking_results.tsv: Best pose summary per candidate
        2. pose_metrics.tsv: All generated poses with structural metrics
    """
    all_pose_records: list[dict[str, Any]] = []
    best_candidate_records: list[dict[str, Any]] = []

    for cand in candidates:
        if cand.status == "completed" and cand.docking_results:
            poses = cand.docking_results
            # Sort poses by rank / score
            # Lower score is generally better for DFIRE/CPYDOCK (energies)
            sorted_poses = sorted(poses, key=lambda p: p.get("rank", 999))
            best_pose = sorted_poses[0]

            for p in sorted_poses:
                rec = {
                    "candidate_id": cand.id,
                    "sequence": cand.sequence,
                    "pose_rank": p.get("rank", 1),
                    "swarm_id": p.get("swarm_id"),
                    "glowworm_id": p.get("glowworm_id"),
                    "lightdock_score": p.get("score"),
                    "scoring_function": p.get("scoring_function", "fastdfire"),
                    "clashes": p.get("clashes_count", 0),
                    "contacts": p.get("contacts_count", 0),
                    "hbonds": p.get("hbonds_count", 0),
                    "salt_bridges": p.get("salt_bridges_count", 0),
                    "restraint_satisfaction_rate": p.get("restraint_satisfaction_rate", 1.0),
                    "interface_receptor_residues": ";".join(p.get("interface_receptor_residues", [])),
                    "interface_peptide_residues": ";".join(p.get("interface_peptide_residues", [])),
                    "pose_pdb_path": p.get("pdb_path", ""),
                    "status": p.get("status", "valid"),
                }
                all_pose_records.append(rec)

            best_candidate_records.append({
                "candidate_id": cand.id,
                "sequence": cand.sequence,
                "length": cand.length,
                "best_lightdock_score": best_pose.get("score"),
                "scoring_function": best_pose.get("scoring_function", "fastdfire"),
                "contacts": best_pose.get("contacts_count", 0),
                "hbonds": best_pose.get("hbonds_count", 0),
                "clashes": best_pose.get("clashes_count", 0),
                "salt_bridges": best_pose.get("salt_bridges_count", 0),
                "restraint_satisfaction_rate": best_pose.get("restraint_satisfaction_rate", 1.0),
                "best_pose_pdb": best_pose.get("pdb_path", ""),
                "status": "completed",
            })
        else:
            # Candidate was filtered or failed
            best_candidate_records.append({
                "candidate_id": cand.id,
                "sequence": cand.sequence,
                "length": cand.length,
                "best_lightdock_score": None,
                "scoring_function": None,
                "contacts": None,
                "hbonds": None,
                "clashes": None,
                "salt_bridges": None,
                "restraint_satisfaction_rate": None,
                "best_pose_pdb": None,
                "status": cand.status,
                "failure_reason": cand.failure_reason,
            })

    # Sort successful candidates by best_lightdock_score ascending (lower energy/DFIRE = better)
    df_candidates = pd.DataFrame(best_candidate_records)
    if not df_candidates.empty and "best_lightdock_score" in df_candidates.columns:
        # Separate completed from others
        completed_mask = df_candidates["status"] == "completed"
        df_comp = df_candidates[completed_mask].sort_values("best_lightdock_score", ascending=True)
        df_other = df_candidates[~completed_mask]
        df_candidates = pd.concat([df_comp, df_other], ignore_index=True)
        # Assign overall rank
        df_candidates["overall_rank"] = range(1, len(df_candidates) + 1)

    df_poses = pd.DataFrame(all_pose_records)

    if output_dir:
        out_dir = Path(ensure_dir(output_dir))
        cand_path = out_dir / "docking_results.tsv"
        poses_path = out_dir / "pose_metrics.tsv"
        df_candidates.to_csv(cand_path, sep="\t", index=False)
        df_poses.to_csv(poses_path, sep="\t", index=False)
        logger.info(f"Docking summary written to: {cand_path}")
        logger.info(f"Detailed pose metrics written to: {poses_path}")

    return df_candidates, df_poses
