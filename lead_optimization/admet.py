"""
DrugLab Lead Optimization — ADMET Wrapper.

Thin wrapper around ``core.admet.screen_admet()`` with optimization-specific
defaults (typically more lenient thresholds for early-stage analogues).
"""

import os
from core.admet import screen_admet
from utils.logger import setup_logger

logger = setup_logger("optimization_admet", level="INFO")


def screen_analogues_admet(
    input_sdf: str,
    output_dir: str,
    profile: str = "balanced",
    target_type: str = "auto",
    filter_toxic: bool = True,
    filter_low_absorption: bool = True,
    thresholds: dict | None = None,
) -> dict:
    """
    ADMET screening for lead optimization analogues.

    Uses slightly relaxed defaults compared to the discovery screening,
    since analogues are early-stage candidates meant for further refinement.

    Args:
        input_sdf: Path to the analogues SDF file.
        output_dir: Output directory.
        profile: ADMET threshold profile ('conservative', 'balanced', 'permissive').
        target_type: Target type for BBB screening ('auto', 'general', 'cns', 'peripheral').
        filter_toxic: Remove toxic molecules.
        filter_low_absorption: Remove low-absorption molecules.
        thresholds: Custom ADMET thresholds (overrides defaults).

    Returns:
        Dict with ADMET screening results.
    """
    # Use a separate sub-directory to avoid overwriting discovery ADMET results
    admet_output = os.path.join(output_dir, "optimization_admet")
    os.makedirs(admet_output, exist_ok=True)

    return screen_admet(
        input_sdf=input_sdf,
        output_dir=admet_output,
        profile=profile,
        target_type=target_type,
        filter_toxic=filter_toxic,
        filter_low_absorption=filter_low_absorption,
        thresholds=thresholds,
    )
