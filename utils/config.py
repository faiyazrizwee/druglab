"""
DrugLab Configuration System.

Handles loading, merging, and validating YAML configuration files.
Provides a DrugLabConfig dataclass that every module reads from.

Configuration hierarchy (highest priority wins):
    CLI flags  →  User config YAML  →  Default config YAML  →  Hardcoded defaults

Usage:
    from utils.config import load_config
    config = load_config("my_config.yml")
    print(config.docking.exhaustiveness)
"""

import copy
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

from utils.logger import setup_logger

logger = setup_logger("config", level="INFO")

# ──────────────────────────────────────────────────────────────
# Configuration Dataclasses
# ──────────────────────────────────────────────────────────────


@dataclass
class ProteinConfig:
    """Protein preparation configuration."""
    remove_water: bool = True
    remove_heterogens: bool = True
    add_hydrogens: bool = True
    fix_missing_residues: bool = False
    fix_missing_atoms: bool = True
    ph: float = 7.4
    chain_id: str | None = None
    keep_ligands: list[str] = field(default_factory=list)


@dataclass
class PocketConfig:
    """Pocket detection configuration."""
    method: str = "geometric"          # "geometric" | "ligand_based" | "manual"
    selection_mode: str = "best_score"  # "best_score" | "largest_volume" | "ligand_guided" | "manual"
    min_pocket_size: float = 100.0     # Å³
    max_pockets: int = 5
    padding: float = 5.0              # Å (reduced from 10.0 for realistic box sizes)
    max_box_dimension: float = 30.0   # Å — clamp any dimension exceeding this
    warn_large_boxes: bool = True     # Warn when boxes exceed max_box_dimension
    manual_center: list[float] | None = None  # [x, y, z]
    manual_size: list[float] | None = None    # [sx, sy, sz]
    reference_ligand: str | None = None


@dataclass
class LigandConfig:
    """Ligand preparation configuration."""
    add_hydrogens: bool = True
    generate_3d: bool = True
    num_conformers: int = 1
    energy_minimize: bool = True
    ph: float = 7.4
    max_tautomers: int = 1
    convert_to_pdbqt: bool = True


@dataclass
class DrugLikenessConfig:
    """Drug-likeness screening configuration."""
    profile: str = "recommended"       # "recommended" | "oral_drug" | "lead_like" | "fragment" | "custom"
    rules: list[str] = field(default_factory=lambda: ["lipinski"])
    evaluation_mode: str = "majority"  # "strict" | "majority" | "custom"
    minimum_rule_pass_fraction: float = 0.7
    max_violations: int = 1
    brenk_mode: str = "warning"        # "warning" | "fail"
    strict: bool = False               # DEPRECATED — use evaluation_mode instead
    custom_filters: dict | None = None


@dataclass
class ADMETConfig:
    """ADMET screening configuration."""
    profile: str = "balanced"          # "conservative" | "balanced" | "permissive"
    target_type: str = "auto"          # "auto" | "general" | "cns" | "peripheral"
    filter_toxic: bool = True
    filter_low_absorption: bool = True
    thresholds: dict | None = None
    models: list[str] | None = None


@dataclass
class DockingConfig:
    """Molecular docking configuration."""
    exhaustiveness: int = 32
    n_poses: int = 10
    energy_range: float = 3.0        # kcal/mol
    seed: int = 42
    n_cpus: int = 0                  # 0 = auto-detect
    score_only: bool = False


@dataclass
class LeadSelectionConfig:
    """Lead selection configuration."""
    top_n: int = 5
    scoring_method: str = "weighted_sum"  # "weighted_sum" | "pareto" | "topsis"
    weights: dict = field(default_factory=lambda: {
        "affinity": 0.5,
        "admet": 0.3,
        "druglikeness": 0.2,
    })
    affinity_cutoff: float = -6.0    # kcal/mol
    on_no_hits: str = "prompt"       # "prompt" | "fallback" | "abort"
    diversity_filter: bool = True
    similarity_cutoff: float = 0.7


@dataclass
class OptimizationConfig:
    """Lead optimization configuration."""
    strategies: list[str] = field(default_factory=lambda: [
        "bioisostere", "scaffold_hop", "r_group",
    ])
    max_analogues: int = 100
    similarity_range: list[float] = field(default_factory=lambda: [0.4, 0.9])
    mw_range: list[float] = field(default_factory=lambda: [150.0, 600.0])
    top_n: int = 10
    custom_fragments: str | None = None


@dataclass
class ReportConfig:
    """Report generation configuration."""
    format: str = "html"             # "html" | "json" | "csv"
    include_charts: bool = True
    include_raw_data: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    verbose: bool = False


@dataclass
class DrugLabConfig:
    """
    Master configuration dataclass for DrugLab.

    Contains nested configuration sections for every module.
    """
    project_name: str = "druglab_run"
    output_dir: str = "output"
    protein: ProteinConfig = field(default_factory=ProteinConfig)
    pocket: PocketConfig = field(default_factory=PocketConfig)
    ligand: LigandConfig = field(default_factory=LigandConfig)
    druglikeness: DrugLikenessConfig = field(default_factory=DrugLikenessConfig)
    admet: ADMETConfig = field(default_factory=ADMETConfig)
    docking: DockingConfig = field(default_factory=DockingConfig)
    lead_selection: LeadSelectionConfig = field(default_factory=LeadSelectionConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ──────────────────────────────────────────────────────────────
# Config Section Mapping
# ──────────────────────────────────────────────────────────────

# Maps YAML section names to dataclass types
_SECTION_MAP = {
    "protein_preparation": ("protein", ProteinConfig),
    "pocket_detection": ("pocket", PocketConfig),
    "ligand_preparation": ("ligand", LigandConfig),
    "druglikeness": ("druglikeness", DrugLikenessConfig),
    "admet": ("admet", ADMETConfig),
    "docking": ("docking", DockingConfig),
    "lead_selection": ("lead_selection", LeadSelectionConfig),
    "optimization": ("optimization", OptimizationConfig),
    "report": ("report", ReportConfig),
    "logging": ("logging", LoggingConfig),
}


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def load_config(
    config_path: str | None = None,
    cli_overrides: dict | None = None,
) -> DrugLabConfig:
    """
    Load and merge configuration from multiple sources.

    1. Start with hardcoded defaults (DrugLabConfig dataclass defaults).
    2. Load configs/default.yml if it exists and merge.
    3. If config_path is provided, load and deep-merge user YAML on top.
    4. Apply CLI overrides (flat dict of dotted keys, e.g. {"docking.exhaustiveness": 64}).
    5. Validate the final configuration.

    Args:
        config_path: Optional path to a user YAML config file.
        cli_overrides: Optional dict of CLI flag overrides.

    Returns:
        Fully resolved DrugLabConfig instance.

    Raises:
        FileNotFoundError: If config_path is specified but doesn't exist.
        ValueError: If configuration validation fails.
    """
    # Step 1: Start with hardcoded defaults as a dict
    config_dict = _dataclass_to_dict(DrugLabConfig())

    # Step 2: Load default.yml
    default_yml = _find_default_config()
    if default_yml:
        logger.debug(f"Loading default config: {default_yml}")
        default_data = _load_yaml(default_yml)
        config_dict = merge_configs(config_dict, default_data)

    # Step 3: Load user config
    if config_path:
        config_path = str(config_path)
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        logger.info(f"Loading user config: {config_path}")
        user_data = _load_yaml(config_path)
        config_dict = merge_configs(config_dict, user_data)

    # Step 4: Apply CLI overrides
    if cli_overrides:
        config_dict = _apply_cli_overrides(config_dict, cli_overrides)

    # Step 5: Build DrugLabConfig from merged dict
    config = _dict_to_config(config_dict)

    # Step 6: Validate
    errors = validate_config(config)
    if errors:
        for err in errors:
            logger.error(f"Config validation error: {err}")
        raise ValueError(
            f"Configuration validation failed with {len(errors)} error(s):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    logger.debug("Configuration loaded and validated successfully")
    return config


def merge_configs(base: dict, override: dict) -> dict:
    """
    Deep-merge two config dicts. Override values win on conflicts.

    Args:
        base: Base configuration dict.
        override: Override configuration dict.

    Returns:
        New merged dict (neither input is modified).
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def validate_config(config: DrugLabConfig) -> list[str]:
    """
    Validate configuration parameters.

    Returns:
        List of validation error messages (empty = valid).
    """
    errors: list[str] = []

    # Protein
    if config.protein.ph < 0 or config.protein.ph > 14:
        errors.append(f"protein.ph must be 0–14, got {config.protein.ph}")

    # Pocket
    valid_methods = ("geometric", "ligand_based", "manual")
    if config.pocket.method not in valid_methods:
        errors.append(
            f"pocket.method must be one of {valid_methods}, got '{config.pocket.method}'"
        )
    if config.pocket.method == "manual":
        if config.pocket.manual_center is None or config.pocket.manual_size is None:
            errors.append(
                "pocket.manual_center and pocket.manual_size are required "
                "when pocket.method is 'manual'"
            )
    if config.pocket.padding < 0:
        errors.append(f"pocket.padding must be >= 0, got {config.pocket.padding}")

    # Docking
    if config.docking.exhaustiveness < 1:
        errors.append(f"docking.exhaustiveness must be >= 1, got {config.docking.exhaustiveness}")
    if config.docking.n_poses < 1:
        errors.append(f"docking.n_poses must be >= 1, got {config.docking.n_poses}")
    if config.docking.energy_range <= 0:
        errors.append(f"docking.energy_range must be > 0, got {config.docking.energy_range}")

    # Drug-likeness
    valid_rules = {"lipinski", "veber", "ghose", "egan", "muegge", "lead_like", "pains", "brenk", "rule_of_3"}
    for rule in config.druglikeness.rules:
        if rule not in valid_rules:
            errors.append(f"druglikeness.rules contains unknown rule '{rule}'. Valid: {valid_rules}")

    valid_profiles = {"recommended", "oral_drug", "lead_like", "fragment", "custom"}
    if config.druglikeness.profile not in valid_profiles:
        errors.append(f"druglikeness.profile must be one of {valid_profiles}, got '{config.druglikeness.profile}'")

    valid_eval_modes = {"strict", "majority", "custom"}
    if config.druglikeness.evaluation_mode not in valid_eval_modes:
        errors.append(f"druglikeness.evaluation_mode must be one of {valid_eval_modes}, got '{config.druglikeness.evaluation_mode}'")

    if not (0.0 <= config.druglikeness.minimum_rule_pass_fraction <= 1.0):
        errors.append(f"druglikeness.minimum_rule_pass_fraction must be 0.0–1.0, got {config.druglikeness.minimum_rule_pass_fraction}")

    valid_brenk_modes = {"warning", "fail"}
    if config.druglikeness.brenk_mode not in valid_brenk_modes:
        errors.append(f"druglikeness.brenk_mode must be one of {valid_brenk_modes}, got '{config.druglikeness.brenk_mode}'")

    valid_admet_profiles = {"conservative", "balanced", "permissive"}
    if config.admet.profile not in valid_admet_profiles:
        errors.append(f"admet.profile must be one of {valid_admet_profiles}, got '{config.admet.profile}'")

    valid_target_types = {"auto", "general", "cns", "peripheral"}
    if config.admet.target_type not in valid_target_types:
        errors.append(f"admet.target_type must be one of {valid_target_types}, got '{config.admet.target_type}'")

    valid_pocket_selection = {"best_score", "largest_volume", "ligand_guided", "manual"}
    if config.pocket.selection_mode not in valid_pocket_selection:
        errors.append(f"pocket.selection_mode must be one of {valid_pocket_selection}, got '{config.pocket.selection_mode}'")

    if config.pocket.max_box_dimension <= 0:
        errors.append(f"pocket.max_box_dimension must be > 0, got {config.pocket.max_box_dimension}")

    # Lead selection
    valid_scoring = ("weighted_sum", "pareto", "topsis")
    if config.lead_selection.scoring_method not in valid_scoring:
        errors.append(
            f"lead_selection.scoring_method must be one of {valid_scoring}, "
            f"got '{config.lead_selection.scoring_method}'"
        )
    if config.lead_selection.top_n < 1:
        errors.append(f"lead_selection.top_n must be >= 1, got {config.lead_selection.top_n}")

    valid_on_no_hits = ("prompt", "fallback", "abort")
    if config.lead_selection.on_no_hits not in valid_on_no_hits:
        errors.append(
            f"lead_selection.on_no_hits must be one of {valid_on_no_hits}, "
            f"got '{config.lead_selection.on_no_hits}'"
        )

    # Optimization
    valid_strategies = {"bioisostere", "scaffold_hop", "r_group", "fragmentation"}
    for strat in config.optimization.strategies:
        if strat not in valid_strategies:
            errors.append(
                f"optimization.strategies contains unknown strategy '{strat}'. "
                f"Valid: {valid_strategies}"
            )
    if config.optimization.max_analogues < 1:
        errors.append(
            f"optimization.max_analogues must be >= 1, got {config.optimization.max_analogues}"
        )

    # Logging
    valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    if config.logging.level.upper() not in valid_levels:
        errors.append(
            f"logging.level must be one of {valid_levels}, got '{config.logging.level}'"
        )

    return errors


def save_config_snapshot(config: DrugLabConfig, output_dir: str) -> str:
    """
    Save the resolved configuration to output_dir/config_snapshot.yml.

    This records the exact parameters used for a run, enabling reproducibility.

    Args:
        config: The resolved DrugLabConfig.
        output_dir: Output directory path.

    Returns:
        Path to the saved config snapshot file.
    """
    snapshot_dir = Path(output_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / "config_snapshot.yml"

    config_dict = _dataclass_to_dict(config)
    with open(snapshot_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Config snapshot saved: {snapshot_path}")
    return str(snapshot_path)


def config_to_dict(config: DrugLabConfig) -> dict:
    """Convert a DrugLabConfig to a plain dict for serialization."""
    return _dataclass_to_dict(config)


# ──────────────────────────────────────────────────────────────
# Internal Helpers
# ──────────────────────────────────────────────────────────────

def _find_default_config() -> str | None:
    """Locate configs/default.yml relative to the project root."""
    # Search relative to this file's location (utils/ → parent → configs/)
    project_root = Path(__file__).resolve().parent.parent
    candidates = [
        project_root / "configs" / "default.yml",
        project_root / "configs" / "default.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _load_yaml(path: str) -> dict:
    """Load a YAML file and return as dict."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping at top level: {path}")
    return data


def _dataclass_to_dict(obj: Any) -> dict:
    """
    Convert a dataclass instance to a plain dict, recursively handling
    nested dataclasses.
    """
    if hasattr(obj, "__dataclass_fields__"):
        result = {}
        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)
            result[field_name] = _dataclass_to_dict(value)
        return result
    elif isinstance(obj, list):
        return [_dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    else:
        return obj


def _dict_to_config(data: dict) -> DrugLabConfig:
    """
    Convert a plain dict to a DrugLabConfig instance.

    Handles the YAML section name → dataclass attribute mapping:
        protein_preparation → protein
        pocket_detection → pocket
        ligand_preparation → ligand
    """
    # Normalize YAML section names to dataclass field names
    normalized = {}
    for key, value in data.items():
        if key in _SECTION_MAP:
            attr_name, _ = _SECTION_MAP[key]
            normalized[attr_name] = value
        elif key == "project":
            # Flatten project section
            if isinstance(value, dict):
                for pk, pv in value.items():
                    if pk == "name":
                        normalized["project_name"] = pv
                    elif pk == "output_dir":
                        normalized["output_dir"] = pv
                    else:
                        normalized[pk] = pv
            else:
                normalized["project_name"] = value
        else:
            normalized[key] = value

    # Build nested dataclasses
    config_kwargs: dict[str, Any] = {}

    for field_name, field_obj in DrugLabConfig.__dataclass_fields__.items():
        if field_name in normalized:
            value = normalized[field_name]
            field_type = field_obj.type

            # If the field type is a dataclass and value is a dict, instantiate it
            if isinstance(value, dict) and hasattr(field_type, "__dataclass_fields__"):
                # Only pass keys that are valid fields
                valid_keys = {k for k in field_type.__dataclass_fields__}
                filtered = {k: v for k, v in value.items() if k in valid_keys}
                config_kwargs[field_name] = field_type(**filtered)
            else:
                config_kwargs[field_name] = value

    return DrugLabConfig(**config_kwargs)


def _apply_cli_overrides(config_dict: dict, overrides: dict) -> dict:
    """
    Apply CLI overrides specified as dotted keys.

    Example:
        overrides = {"docking.exhaustiveness": 64, "lead_selection.top_n": 10}
    """
    result = copy.deepcopy(config_dict)
    for dotted_key, value in overrides.items():
        keys = dotted_key.split(".")
        target = result
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value
    return result
