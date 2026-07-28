"""
DrugLab Lead Optimization — Combined Screening Module.

Wraps core drug-likeness and ADMET screening for analogue evaluation.
Merges results into a single output.
"""

import os
import time
import logging
import pandas as pd

from utils.logger import setup_logger, log_step
from utils.io import ensure_dir, write_json, write_csv, validate_file_exists

from core.druglikeness import screen_druglikeness
from core.admet import screen_admet

logger = setup_logger("optimization_screening", level="INFO")


def screen_analogues(
    input_sdf: str,
    output_dir: str,
    profile: str | None = None,
    evaluation_mode: str = "majority",
    minimum_rule_pass_fraction: float = 0.7,
    brenk_mode: str = "warning",
    druglikeness_rules: list[str] | None = None,
    max_violations: int = 1,
    strict: bool = False,
    admet_profile: str = "balanced",
    admet_target_type: str = "auto",
    filter_toxic: bool = True,
    filter_low_absorption: bool = True,
    admet_thresholds: dict | None = None,
) -> dict:
    """
    Screen generated analogues through drug-likeness and ADMET filters.

    Args:
        input_sdf: Path to the analogues SDF file.
        output_dir: Path to the output directory.
        profile: Drug-likeness screening profile.
        evaluation_mode: Drug-likeness evaluation mode.
        minimum_rule_pass_fraction: Minimum pass fraction for custom mode.
        brenk_mode: Brenk filter mode ('warning' or 'fail').
        druglikeness_rules: Rules for drug-likeness screening.
        max_violations: Max violations for Lipinski/Muegge.
        strict: If True, all rules must pass (deprecated).
        admet_profile: ADMET threshold profile.
        admet_target_type: Target type for ADMET BBB screening.
        filter_toxic: Remove toxic molecules in ADMET.
        filter_low_absorption: Remove low-absorption molecules in ADMET.
        admet_thresholds: Custom ADMET thresholds.

    Returns:
        Dict with screening results and output file paths.
    """
    start_time = time.time()
    log_step(logger, "Analogue Screening", "started", {"input": input_sdf})

    validate_file_exists(input_sdf, "Analogues SDF")

    screen_dir = os.path.join(output_dir, "screening")
    ensure_dir(screen_dir)
    report_path = os.path.join(screen_dir, "screening_report.json")

    # Step 1: Drug-likeness screening
    logger.info("Running drug-likeness screening on analogues...")
    dl_result = screen_druglikeness(
        input_sdf=input_sdf,
        output_dir=screen_dir,
        profile=profile,
        rules=druglikeness_rules,
        evaluation_mode=evaluation_mode,
        minimum_rule_pass_fraction=minimum_rule_pass_fraction,
        brenk_mode=brenk_mode,
        max_violations=max_violations,
        strict=strict,
    )

    passed_dl_sdf = dl_result.get("passed_sdf")
    dl_passed_count = dl_result.get("total_passed", 0)
    logger.info(f"Drug-likeness: {dl_passed_count} passed / {dl_result.get('total_input', 0)} total")

    # Step 2: ADMET screening on drug-like-passed molecules
    admet_result = {"total_passed": 0, "total_input": 0, "status": "skipped"}
    if passed_dl_sdf and os.path.isfile(passed_dl_sdf) and dl_passed_count > 0:
        logger.info("Running ADMET screening on drug-like analogues...")
        admet_result = screen_admet(
            input_sdf=passed_dl_sdf,
            output_dir=screen_dir,
            profile=admet_profile,
            target_type=admet_target_type,
            filter_toxic=filter_toxic,
            filter_low_absorption=filter_low_absorption,
            thresholds=admet_thresholds,
        )
        logger.info(f"ADMET: {admet_result.get('total_passed', 0)} passed / "
                     f"{admet_result.get('total_input', 0)} total")
    else:
        logger.warning("No molecules passed drug-likeness; skipping ADMET screening.")

    duration = time.time() - start_time
    metrics = {
        "druglikeness": {
            "total_input": dl_result.get("total_input", 0),
            "total_passed": dl_result.get("total_passed", 0),
            "passed_sdf": dl_result.get("passed_sdf"),
            "descriptors_csv": dl_result.get("descriptors_csv"),
        },
        "admet": {
            "total_input": admet_result.get("total_input", 0),
            "total_passed": admet_result.get("total_passed", 0),
            "passed_sdf": admet_result.get("passed_sdf"),
            "admet_csv": admet_result.get("admet_csv"),
        },
        "final_passed_sdf": admet_result.get("passed_sdf") or dl_result.get("passed_sdf"),
        "duration": round(duration, 2),
        "status": "completed",
    }

    write_json(metrics, report_path)
    log_step(logger, "Analogue Screening", "completed", {
        "dl_passed": dl_result.get("total_passed", 0),
        "admet_passed": admet_result.get("total_passed", 0),
        "duration": f"{duration:.1f}s",
    })

    return metrics
