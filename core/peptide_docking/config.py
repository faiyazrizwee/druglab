"""
Configuration dataclasses, validation, and loaders for DrugLab 2.0 Peptide Docking.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


PRESET_SAMPLING = {
    "fast": {"swarms": 10, "steps": 10, "glowworms": 50},
    "balanced": {"swarms": 30, "steps": 25, "glowworms": 50},
    "thorough": {"swarms": 100, "steps": 50, "glowworms": 100},
}

SUPPORTED_SCORING_FUNCTIONS = {
    "fastdfire", "dfire", "dfire2", "cpydock", "tobi", "sd", "mj3h", "pisa", "vdw"
}

SUPPORTED_BINDING_SITE_MODES = {"auto", "explicit", "pocket_guided", "global"}
SUPPORTED_ADMET_PROVIDERS = {"local", "pepadmet", "peplife", "none"}


@dataclass
class ProteinConfig:
    input: str | None = None
    chains: list[str] = field(default_factory=list)
    remove_waters: bool = True
    water_policy: str = "remove_bulk"  # "remove_bulk" (keep interfacial), "keep_all", "remove_all"
    keep_metals: bool = True
    keep_cofactors: bool = True
    keep_heteroatoms: bool = True
    remove_hydrogens: bool = False
    biological_assembly: int | None = None
    ph: float = 7.4


@dataclass
class PeptideConfig:
    fasta: str | None = None
    sequence: str | None = None
    structure: str | None = None
    allow_nonstandard_residues: bool = False
    n_term_cap: str = "zwitterion"  # "zwitterion", "ace", "free"
    c_term_cap: str = "zwitterion"  # "zwitterion", "nhe", "free"
    is_cyclic: bool = False
    disulfide_pairs: list[tuple[int, int]] | None = None
    conformers: int = 1
    minimize: bool = True
    forcefield: str = "MMFF94"
    seed: int = 42


@dataclass
class FilteringConfig:
    enabled: bool = True
    min_length_aa: int = 5
    max_length_aa: int = 30
    max_mw: float = 4000.0
    retain_filtered_candidates: bool = True
    warn_on_length: bool = True


@dataclass
class AdmetConfig:
    enabled: bool = True
    provider: str = "local"
    reference_library_path: str | None = None
    toxicity_model_enabled: bool = True
    on_prediction_failure: str = "warn_and_continue"
    max_instability_index: float | None = None
    max_gravy: float | None = None


@dataclass
class BindingSiteConfig:
    mode: str = "auto"
    receptor_residues: list[str] = field(default_factory=list)
    peptide_residues: list[str] = field(default_factory=list)
    restraints_enabled: bool = False
    pocket_method: str = "geometric"
    reference_ligand: str | None = None
    pocket_cutoff_distance: float = 6.0


@dataclass
class LightdockConfig:
    executable: str | None = None
    preset: str = "fast"
    scoring_function: str = "fastdfire"
    swarms: int | None = None
    glowworms: int = 50
    steps: int | None = None
    anm: bool = False
    timeout_seconds: int = 3600
    seed: int = 42

    def resolve_sampling(self) -> tuple[int, int, int]:
        """Resolve effective (swarms, steps, glowworms) based on preset and overrides."""
        preset_vals = PRESET_SAMPLING.get(self.preset, PRESET_SAMPLING["fast"])
        swarms = self.swarms if self.swarms is not None else preset_vals["swarms"]
        steps = self.steps if self.steps is not None else preset_vals["steps"]
        glowworms = self.glowworms if self.glowworms is not None else preset_vals["glowworms"]
        return swarms, steps, glowworms


@dataclass
class ResourceConfig:
    max_parallel_jobs: int = 1
    max_cpu_processes: int = 4


@dataclass
class ReportConfig:
    html: bool = True
    interactive_3d: bool = True
    title: str = "DrugLab Peptide Docking Report"


@dataclass
class PeptideDockingConfig:
    workflow_mode: str = "peptide"
    protein: ProteinConfig = field(default_factory=ProteinConfig)
    peptide: PeptideConfig = field(default_factory=PeptideConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    admet: AdmetConfig = field(default_factory=AdmetConfig)
    binding_site: BindingSiteConfig = field(default_factory=BindingSiteConfig)
    lightdock: LightdockConfig = field(default_factory=LightdockConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    output_dir: str = "results"

    def validate(self) -> None:
        """Validate configuration settings and raise ValueError on invalid options."""
        if self.lightdock.preset not in PRESET_SAMPLING:
            raise ValueError(
                f"Invalid LightDock preset '{self.lightdock.preset}'. "
                f"Must be one of: {list(PRESET_SAMPLING.keys())}"
            )
        if self.lightdock.scoring_function not in SUPPORTED_SCORING_FUNCTIONS:
            raise ValueError(
                f"Unsupported scoring function '{self.lightdock.scoring_function}'. "
                f"Supported: {sorted(SUPPORTED_SCORING_FUNCTIONS)}"
            )
        if self.binding_site.mode not in SUPPORTED_BINDING_SITE_MODES:
            raise ValueError(
                f"Invalid binding-site mode '{self.binding_site.mode}'. "
                f"Must be one of: {sorted(SUPPORTED_BINDING_SITE_MODES)}"
            )
        if self.admet.provider not in SUPPORTED_ADMET_PROVIDERS:
            raise ValueError(
                f"Invalid ADMET provider '{self.admet.provider}'. "
                f"Must be one of: {sorted(SUPPORTED_ADMET_PROVIDERS)}"
            )
        if self.filtering.min_length_aa < 1:
            raise ValueError("Filtering min_length_aa must be >= 1")
        if self.filtering.max_length_aa < self.filtering.min_length_aa:
            raise ValueError("Filtering max_length_aa must be >= min_length_aa")
        if self.peptide.conformers < 1:
            raise ValueError("Peptide conformers count must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        """Convert config dataclass to dictionary."""
        return {
            "workflow": {"mode": self.workflow_mode},
            "protein": {
                "input": self.protein.input,
                "chains": self.protein.chains,
                "remove_waters": self.protein.remove_waters,
                "water_policy": self.protein.water_policy,
                "keep_metals": self.protein.keep_metals,
                "keep_cofactors": self.protein.keep_cofactors,
                "keep_heteroatoms": self.protein.keep_heteroatoms,
                "remove_hydrogens": self.protein.remove_hydrogens,
                "biological_assembly": self.protein.biological_assembly,
                "ph": self.protein.ph,
            },
            "peptide": {
                "fasta": self.peptide.fasta,
                "sequence": self.peptide.sequence,
                "structure": self.peptide.structure,
                "allow_nonstandard_residues": self.peptide.allow_nonstandard_residues,
                "n_term_cap": self.peptide.n_term_cap,
                "c_term_cap": self.peptide.c_term_cap,
                "is_cyclic": self.peptide.is_cyclic,
                "disulfide_pairs": self.peptide.disulfide_pairs,
                "conformers": self.peptide.conformers,
                "minimize": self.peptide.minimize,
                "forcefield": self.peptide.forcefield,
                "seed": self.peptide.seed,
            },
            "filtering": {
                "enabled": self.filtering.enabled,
                "min_length_aa": self.filtering.min_length_aa,
                "max_length_aa": self.filtering.max_length_aa,
                "max_mw": self.filtering.max_mw,
                "retain_filtered_candidates": self.filtering.retain_filtered_candidates,
                "warn_on_length": self.filtering.warn_on_length,
            },
            "admet": {
                "enabled": self.admet.enabled,
                "provider": self.admet.provider,
                "on_prediction_failure": self.admet.on_prediction_failure,
                "max_instability_index": self.admet.max_instability_index,
                "max_gravy": self.admet.max_gravy,
            },
            "binding_site": {
                "mode": self.binding_site.mode,
                "receptor_residues": self.binding_site.receptor_residues,
                "peptide_residues": self.binding_site.peptide_residues,
                "restraints_enabled": self.binding_site.restraints_enabled,
                "pocket_method": self.binding_site.pocket_method,
                "reference_ligand": self.binding_site.reference_ligand,
                "pocket_cutoff_distance": self.binding_site.pocket_cutoff_distance,
            },
            "lightdock": {
                "executable": self.lightdock.executable,
                "preset": self.lightdock.preset,
                "scoring_function": self.lightdock.scoring_function,
                "swarms": self.lightdock.swarms,
                "glowworms": self.lightdock.glowworms,
                "steps": self.lightdock.steps,
                "anm": self.lightdock.anm,
                "timeout_seconds": self.lightdock.timeout_seconds,
                "seed": self.lightdock.seed,
            },
            "resources": {
                "max_parallel_jobs": self.resources.max_parallel_jobs,
                "max_cpu_processes": self.resources.max_cpu_processes,
            },
            "report": {
                "html": self.report.html,
                "interactive_3d": self.report.interactive_3d,
                "title": self.report.title,
            },
            "output": {"directory": self.output_dir},
        }


def _deep_update(base: dict, update: dict) -> dict:
    """Recursively update nested dictionary."""
    for k, v in update.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_peptide_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> PeptideDockingConfig:
    """
    Load peptide docking configuration from YAML file and apply CLI/runtime overrides.
    """
    default_config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "peptide_docking.yaml"
    raw_data: dict[str, Any] = {}

    if default_config_path.is_file():
        try:
            with open(default_config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    raw_data = loaded
        except Exception as e:
            # Fall back to programmatic defaults
            pass

    if config_path and os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_loaded = yaml.safe_load(f)
            if isinstance(user_loaded, dict):
                _deep_update(raw_data, user_loaded)

    if overrides:
        _deep_update(raw_data, overrides)

    cfg = PeptideDockingConfig()

    wf_raw = raw_data.get("workflow", {})
    if isinstance(wf_raw, dict):
        cfg.workflow_mode = wf_raw.get("mode", "peptide")

    prot = raw_data.get("protein", {})
    if isinstance(prot, dict):
        cfg.protein = ProteinConfig(
            input=prot.get("input"),
            chains=prot.get("chains", []) or [],
            remove_waters=prot.get("remove_waters", True),
            water_policy=str(prot.get("water_policy", "remove_bulk")),
            keep_metals=bool(prot.get("keep_metals", True)),
            keep_cofactors=bool(prot.get("keep_cofactors", True)),
            keep_heteroatoms=bool(prot.get("keep_heteroatoms", True)),
            remove_hydrogens=prot.get("remove_hydrogens", False),
            biological_assembly=prot.get("biological_assembly"),
            ph=float(prot.get("ph", 7.4)),
        )

    pep = raw_data.get("peptide", {})
    if isinstance(pep, dict):
        cfg.peptide = PeptideConfig(
            fasta=pep.get("fasta"),
            sequence=pep.get("sequence"),
            structure=pep.get("structure"),
            allow_nonstandard_residues=pep.get("allow_nonstandard_residues", False),
            n_term_cap=str(pep.get("n_term_cap", "zwitterion")),
            c_term_cap=str(pep.get("c_term_cap", "zwitterion")),
            is_cyclic=bool(pep.get("is_cyclic", False)),
            disulfide_pairs=pep.get("disulfide_pairs"),
            conformers=int(pep.get("conformers", 1)),
            minimize=pep.get("minimize", True),
            forcefield=str(pep.get("forcefield", "MMFF94")),
            seed=int(pep.get("seed", 42)),
        )

    filt = raw_data.get("filtering", {})
    if isinstance(filt, dict):
        cfg.filtering = FilteringConfig(
            enabled=filt.get("enabled", True),
            min_length_aa=int(filt.get("min_length_aa", 5)),
            max_length_aa=int(filt.get("max_length_aa", 30)),
            max_mw=float(filt.get("max_mw", 4000.0)),
            retain_filtered_candidates=filt.get("retain_filtered_candidates", True),
            warn_on_length=bool(filt.get("warn_on_length", True)),
        )

    admet_raw = raw_data.get("admet", {})
    if isinstance(admet_raw, dict):
        raw_provider = str(admet_raw.get("provider", "local")).strip().lower()
        if raw_provider in ("pepadmet", "pep-admet"):
            raw_provider = "local"
        cfg.admet = AdmetConfig(
            enabled=admet_raw.get("enabled", True),
            provider=raw_provider,
            reference_library_path=admet_raw.get("reference_library_path"),
            toxicity_model_enabled=bool(admet_raw.get("toxicity_model_enabled", True)),
            on_prediction_failure=str(admet_raw.get("on_prediction_failure", "warn_and_continue")),
            max_instability_index=admet_raw.get("max_instability_index"),
            max_gravy=admet_raw.get("max_gravy"),
        )

    bs = raw_data.get("binding_site", {})
    if isinstance(bs, dict):
        cfg.binding_site = BindingSiteConfig(
            mode=str(bs.get("mode", "auto")),
            receptor_residues=bs.get("receptor_residues", []) or [],
            peptide_residues=bs.get("peptide_residues", []) or [],
            restraints_enabled=bool(bs.get("restraints_enabled", False)),
            pocket_method=str(bs.get("pocket_method", "geometric")),
            reference_ligand=bs.get("reference_ligand"),
            pocket_cutoff_distance=float(bs.get("pocket_cutoff_distance", 6.0)),
        )

    lgd = raw_data.get("lightdock", {})
    if isinstance(lgd, dict):
        cfg.lightdock = LightdockConfig(
            executable=lgd.get("executable"),
            preset=str(lgd.get("preset", "fast")),
            scoring_function=str(lgd.get("scoring_function", "fastdfire")),
            swarms=lgd.get("swarms"),
            glowworms=int(lgd.get("glowworms", 50)),
            steps=lgd.get("steps"),
            anm=bool(lgd.get("anm", False)),
            timeout_seconds=int(lgd.get("timeout_seconds", 3600)),
            seed=int(lgd.get("seed", 42)),
        )

    res = raw_data.get("resources", {})
    if isinstance(res, dict):
        cfg.resources = ResourceConfig(
            max_parallel_jobs=int(res.get("max_parallel_jobs", 1)),
            max_cpu_processes=int(res.get("max_cpu_processes", 4)),
        )

    rep = raw_data.get("report", {})
    if isinstance(rep, dict):
        cfg.report = ReportConfig(
            html=bool(rep.get("html", True)),
            interactive_3d=bool(rep.get("interactive_3d", True)),
            title=str(rep.get("title", "DrugLab Peptide Docking Report")),
        )

    out = raw_data.get("output", {})
    if isinstance(out, dict):
        cfg.output_dir = str(out.get("directory", "results"))

    cfg.validate()
    return cfg
