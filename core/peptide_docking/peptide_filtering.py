"""
Peptide size and composition filtering before molecular docking.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from core.peptide_docking.config import FilteringConfig
from core.peptide_docking.input_validation import PeptideCandidate
from utils.io import ensure_dir
from utils.logger import setup_logger

logger = setup_logger("peptide_filtering", level="INFO")


def filter_peptides(
    candidates: list[PeptideCandidate],
    config: FilteringConfig,
    output_dir: str | Path | None = None,
) -> tuple[list[PeptideCandidate], list[PeptideCandidate], pd.DataFrame]:
    """
    Filter peptide candidates by composition and molecular weight.
    Note: Length filter is removed to allow flexible peptide lengths;
    advisory warnings are emitted and recorded when peptide length is outside
    recommended bounds.
    """
    records: list[dict[str, Any]] = []
    passed: list[PeptideCandidate] = []
    filtered: list[PeptideCandidate] = []

    for cand in candidates:
        seq = cand.sequence
        length = len(seq)
        try:
            analysis = ProteinAnalysis(seq)
            mw = round(analysis.molecular_weight(), 2)
        except Exception:
            mw = None

        cand.properties["length"] = length
        cand.properties["molecular_weight"] = mw

        # Check length against recommended bounds to generate advisory warning
        length_warning = None
        if length < config.min_length_aa:
            length_warning = f"Length {length} AA < minimum recommended {config.min_length_aa} AA"
        elif length > config.max_length_aa:
            length_warning = f"Length {length} AA > maximum recommended {config.max_length_aa} AA"

        if length_warning:
            cand.properties["length_warning"] = length_warning
            logger.warning(
                f"Candidate '{cand.id}' ({seq}): {length_warning}. "
                "Length filter is removed; candidate proceeds to docking with warning."
            )

        reason = None
        status = "pass"

        if not config.enabled:
            if length_warning:
                status = "warning"
                reason = f"Warning: {length_warning}"
            cand.status = "ready"
            passed.append(cand)
        else:
            # Length filter is bypassed: only warn, do not fail/reject candidate
            if mw and mw > config.max_mw:
                status = "fail"
                reason = f"Molecular weight {mw:.1f} Da > maximum {config.max_mw:.1f} Da"
                cand.status = "filtered"
                cand.failure_reason = reason
                filtered.append(cand)
            else:
                if length_warning:
                    status = "warning"
                    reason = f"Warning: {length_warning}"
                cand.status = "ready"
                passed.append(cand)

        records.append({
            "candidate_id": cand.id,
            "sequence": seq,
            "length": length,
            "molecular_weight_da": mw,
            "status": status,
            "filter_reason": reason or "",
            "warning": length_warning or "",
        })

    df = pd.DataFrame(records)

    if output_dir:
        out_path = Path(ensure_dir(output_dir)) / "peptide_filtering.tsv"
        df.to_csv(out_path, sep="\t", index=False)
        logger.info(f"Peptide filtering results written to: {out_path}")

    warn_count = sum(1 for c in passed if "length_warning" in c.properties)
    logger.info(
        f"Peptide Filtering: Total={len(candidates)}, Passed={len(passed)} "
        f"(Warnings={warn_count}), Filtered={len(filtered)}"
    )

    return passed, filtered, df
