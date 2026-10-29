"""
Unit tests for peptide docking configuration and presets.
"""

import pytest
from core.peptide_docking.config import (
    load_peptide_config,
    PeptideDockingConfig,
    ProteinConfig,
    PeptideConfig,
    FilteringConfig,
    AdmetConfig,
    BindingSiteConfig,
    LightdockConfig,
    PRESET_SAMPLING,
)


def test_default_config_loading():
    cfg = load_peptide_config()
    assert cfg.workflow_mode == "peptide"
    assert cfg.filtering.enabled is True
    assert cfg.filtering.min_length_aa == 5
    assert cfg.filtering.max_length_aa == 30
    assert cfg.lightdock.preset == "fast"
    assert cfg.lightdock.scoring_function == "fastdfire"


def test_sampling_resolution():
    cfg = PeptideDockingConfig()
    cfg.lightdock.preset = "fast"
    swarms, steps, glowworms = cfg.lightdock.resolve_sampling()
    assert swarms == PRESET_SAMPLING["fast"]["swarms"]
    assert steps == PRESET_SAMPLING["fast"]["steps"]
    assert glowworms == PRESET_SAMPLING["fast"]["glowworms"]

    # Overrides
    cfg.lightdock.swarms = 20
    cfg.lightdock.steps = 15
    s, st, g = cfg.lightdock.resolve_sampling()
    assert s == 20
    assert st == 15


def test_config_validation_failures():
    cfg = PeptideDockingConfig()

    cfg.lightdock.preset = "invalid_preset"
    with pytest.raises(ValueError, match="Invalid LightDock preset"):
        cfg.validate()

    cfg.lightdock.preset = "fast"
    cfg.lightdock.scoring_function = "unknown_scoring"
    with pytest.raises(ValueError, match="Unsupported scoring function"):
        cfg.validate()

    cfg.lightdock.scoring_function = "fastdfire"
    cfg.binding_site.mode = "invalid_mode"
    with pytest.raises(ValueError, match="Invalid binding-site mode"):
        cfg.validate()

    cfg.binding_site.mode = "auto"
    cfg.filtering.min_length_aa = 20
    cfg.filtering.max_length_aa = 10
    with pytest.raises(ValueError, match="max_length_aa must be >= min_length_aa"):
        cfg.validate()


def test_config_overrides(tmp_path):
    custom_yaml = tmp_path / "custom.yaml"
    custom_yaml.write_text("""
lightdock:
  preset: balanced
  scoring_function: dfire
filtering:
  min_length_aa: 6
  max_length_aa: 25
""")
    cfg = load_peptide_config(config_path=str(custom_yaml))
    assert cfg.lightdock.preset == "balanced"
    assert cfg.lightdock.scoring_function == "dfire"
    assert cfg.filtering.min_length_aa == 6
    assert cfg.filtering.max_length_aa == 25
