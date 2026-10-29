"""
DrugLab Quarantine Logger.

Records invalid, corrupt, or failed molecules during screening into a
persistent quarantine.json file so that the pipeline continues uninterrupted
while maintaining full auditability of failed compounds.
"""

import os
import json
import time
from datetime import datetime
from typing import Any
from utils.logger import setup_logger
from utils.io import ensure_dir

logger = setup_logger("quarantine", level="INFO")


def get_quarantine_path(output_dir: str) -> str:
    """Return the canonical path to quarantine.json in output_dir."""
    return os.path.join(output_dir, "quarantine.json")


def quarantine_item(
    output_dir: str,
    stage: str,
    mol_id: str,
    reason: str,
    smiles: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Log a failed or malformed molecule to quarantine.json.

    Args:
        output_dir: The run's output directory.
        stage: Pipeline stage where failure occurred (e.g. 'ligand_prep', 'docking').
        mol_id: Molecule name, index, or identifier.
        reason: Description of the error or failure mode.
        smiles: Canonical SMILES string if available.
        details: Optional dict of extra diagnostic data (traceback, parameters).

    Returns:
        The quarantine entry dict.
    """
    ensure_dir(output_dir)
    quarantine_file = get_quarantine_path(output_dir)

    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "stage": stage,
        "mol_id": str(mol_id),
        "smiles": smiles,
        "reason": str(reason),
        "details": details or {},
    }

    records = []
    if os.path.exists(quarantine_file):
        try:
            with open(quarantine_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    records = data
        except Exception as e:
            logger.warning(f"Failed to read existing quarantine file {quarantine_file}: {e}")

    records.append(entry)

    try:
        tmp_file = quarantine_file + ".tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        os.replace(tmp_file, quarantine_file)
    except Exception as e:
        logger.error(f"Failed to write quarantine entry for {mol_id}: {e}")

    logger.warning(f"Quarantined molecule [{mol_id}] at stage [{stage}]: {reason}")
    return entry


def load_quarantine(output_dir: str) -> list[dict[str, Any]]:
    """Load all quarantine records from output_dir."""
    quarantine_file = get_quarantine_path(output_dir)
    if not os.path.exists(quarantine_file):
        return []
    try:
        with open(quarantine_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load quarantine file {quarantine_file}: {e}")
        return []


def get_quarantine_summary(output_dir: str) -> dict[str, Any]:
    """Compute summary statistics of quarantined items."""
    records = load_quarantine(output_dir)
    by_stage = {}
    for r in records:
        s = r.get("stage", "unknown")
        by_stage[s] = by_stage.get(s, 0) + 1

    return {
        "total_quarantined": len(records),
        "by_stage": by_stage,
        "quarantine_file": get_quarantine_path(output_dir),
    }
