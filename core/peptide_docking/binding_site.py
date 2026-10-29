"""
Binding-site selection, pocket mapping, and LightDock restraint generation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from Bio.PDB import PDBParser, Polypeptide
from core.peptide_docking.config import BindingSiteConfig
from utils.io import ensure_dir
from utils.logger import setup_logger

logger = setup_logger("binding_site", level="INFO")


@dataclass
class BindingSiteResult:
    mode: str
    restraints_path: str | None = None
    receptor_restraints: list[str] = field(default_factory=list)
    peptide_restraints: list[str] = field(default_factory=list)
    pocket_center: tuple[float, float, float] | None = None
    pocket_residues: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def validate_residue_identifier(
    identifier: str,
    structure,
) -> tuple[str, str, int, str]:
    """
    Validate residue identifier format ('CHAIN.RESNAME.RESNUM' or 'CHAIN.RESNUM')
    against a parsed BioPython structure.

    Returns:
        tuple (chain_id, resname, resnum, canonical_identifier)
    """
    clean_id = identifier.strip()
    parts = clean_id.split(".")
    if len(parts) == 3:
        chain_id, exp_resname, resnum_str = parts
    elif len(parts) == 2:
        chain_id, resnum_str = parts
        exp_resname = None
    else:
        raise ValueError(
            f"Invalid residue identifier format '{identifier}'. "
            f"Expected 'CHAIN.RESNAME.RESNUM' (e.g. 'A.TYR.45') or 'CHAIN.RESNUM' (e.g. 'A.45')."
        )

    try:
        resnum = int(resnum_str)
    except ValueError:
        raise ValueError(f"Residue number in '{identifier}' must be an integer, got '{resnum_str}'.")

    # Check structure
    matched_residue = None
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for res in chain:
                    if res.id[1] == resnum and res.id[0] == " ":
                        matched_residue = res
                        break
        if matched_residue:
            break

    if not matched_residue:
        raise ValueError(f"Residue '{identifier}' not found in prepared structure.")

    actual_resname = matched_residue.get_resname().strip().upper()
    if exp_resname and exp_resname.upper() != actual_resname:
        logger.warning(
            f"Residue name mismatch for {identifier}: expected {exp_resname}, "
            f"found {actual_resname} in structure. Using {actual_resname}."
        )

    canonical = f"{chain_id}.{actual_resname}.{resnum}"
    return chain_id, actual_resname, resnum, canonical


def find_surface_residues_near_point(
    structure,
    center: tuple[float, float, float],
    cutoff_distance: float = 6.0,
    chain_filter: list[str] | None = None,
) -> list[str]:
    """
    Find receptor residues that have atoms within cutoff_distance of a 3D coordinate.
    """
    import numpy as np

    center_arr = np.array(center, dtype=float)
    cutoff_sq = cutoff_distance ** 2
    contact_residues = []

    for model in structure:
        for chain in model:
            if chain_filter and chain.id not in chain_filter:
                continue
            for res in chain:
                if res.id[0] != " ":
                    continue
                # Check CA or any atom
                is_contact = False
                for atom in res:
                    diff = atom.get_coord() - center_arr
                    if np.dot(diff, diff) <= cutoff_sq:
                        is_contact = True
                        break
                if is_contact:
                    resname = res.get_resname().strip().upper()
                    contact_residues.append(f"{chain.id}.{resname}.{res.id[1]}")
        break  # Use first model

    return contact_residues


def find_contact_residues_near_ligand(
    receptor_structure,
    ligand_pdb_path: str | Path,
    cutoff_distance: float = 6.0,
) -> list[str]:
    """
    Find receptor residues in contact with a reference ligand.
    """
    import numpy as np

    parser = PDBParser(QUIET=True)
    lig_struct = parser.get_structure("ligand", str(ligand_pdb_path))
    lig_coords = [atom.get_coord() for atom in lig_struct.get_atoms()]
    if not lig_coords:
        return []

    lig_arr = np.array(lig_coords)
    cutoff_sq = cutoff_distance ** 2
    contact_residues = set()

    for model in receptor_structure:
        for chain in model:
            for res in chain:
                if res.id[0] != " ":
                    continue
                for atom in res:
                    diffs = lig_arr - atom.get_coord()
                    dists_sq = np.sum(diffs ** 2, axis=1)
                    if np.any(dists_sq <= cutoff_sq):
                        resname = res.get_resname().strip().upper()
                        contact_residues.add(f"{chain.id}.{resname}.{res.id[1]}")
                        break
        break

    return sorted(list(contact_residues))


def setup_binding_site_and_restraints(
    receptor_pdb: str | Path,
    peptide_pdb: str | Path,
    config: BindingSiteConfig,
    output_dir: str | Path,
) -> BindingSiteResult:
    """
    Resolve binding-site mode, validate restraints, and generate LightDock restraints.list.
    """
    parser = PDBParser(QUIET=True)
    rec_structure = parser.get_structure("receptor", str(receptor_pdb))
    pep_structure = parser.get_structure("peptide", str(peptide_pdb))

    out_dir = Path(ensure_dir(output_dir))
    restraints_path = out_dir / "restraints.list"

    canonical_rec: list[str] = []
    canonical_pep: list[str] = []
    pocket_center = None
    details: dict[str, Any] = {}

    mode = config.mode.lower()

    if mode == "auto":
        if config.receptor_residues:
            mode = "explicit"
        elif config.reference_ligand:
            mode = "pocket_guided"
        else:
            mode = "global"

    if mode == "explicit":
        # Validate receptor residues
        for r_id in config.receptor_residues:
            try:
                _, _, _, canon = validate_residue_identifier(r_id, rec_structure)
                canonical_rec.append(canon)
            except Exception as e:
                logger.error(f"Invalid receptor restraint: {e}")
                raise

        # Validate peptide residues if any
        for p_id in config.peptide_residues:
            try:
                _, _, _, canon = validate_residue_identifier(p_id, pep_structure)
                canonical_pep.append(canon)
            except Exception as e:
                logger.error(f"Invalid peptide restraint: {e}")
                raise

    elif mode == "pocket_guided":
        if config.reference_ligand and os.path.isfile(config.reference_ligand):
            logger.info(f"Mapping contact residues from reference ligand: {config.reference_ligand}")
            contacts = find_contact_residues_near_ligand(
                rec_structure,
                config.reference_ligand,
                cutoff_distance=config.pocket_cutoff_distance,
            )
            canonical_rec.extend(contacts)
            details["reference_ligand"] = config.reference_ligand
            details["mapped_contacts_count"] = len(contacts)
        else:
            # Try geometric cavity detection if core.pocket_detection is available
            try:
                from core.pocket_detection import detect_pockets
                pockets_dir = out_dir / "pocket_detection"
                pockets = detect_pockets(
                    protein_pdb=str(receptor_pdb),
                    output_dir=str(pockets_dir),
                    method=config.pocket_method,
                )
                if pockets:
                    top_pocket = pockets[0]
                    pocket_center = tuple(top_pocket.get("center", (0.0, 0.0, 0.0)))
                    contacts = find_surface_residues_near_point(
                        rec_structure,
                        pocket_center,
                        cutoff_distance=config.pocket_cutoff_distance,
                    )
                    canonical_rec.extend(contacts)
                    details["pocket_id"] = top_pocket.get("pocket_id", 1)
                    details["pocket_center"] = pocket_center
                    details["pocket_score"] = top_pocket.get("score")
            except Exception as e:
                logger.warning(
                    f"Pocket detection could not identify pocket: {e}. Falling back to global docking."
                )
                mode = "global"

    elif mode == "global":
        logger.info("Unrestrained global docking selected. Swarms will sample receptor surface uniformly.")

    # Write restraints.list if there are restraints
    has_restraints = bool(canonical_rec or canonical_pep)
    if has_restraints:
        lines = []
        for r in canonical_rec:
            lines.append(f"R {r}")
        for p in canonical_pep:
            lines.append(f"L {p}")

        with open(restraints_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        logger.info(
            f"Wrote LightDock restraints to {restraints_path} "
            f"(Receptor: {len(canonical_rec)}, Peptide: {len(canonical_pep)})"
        )
        return BindingSiteResult(
            mode=mode,
            restraints_path=str(restraints_path.resolve()),
            receptor_restraints=canonical_rec,
            peptide_restraints=canonical_pep,
            pocket_center=pocket_center,
            details=details,
        )
    else:
        return BindingSiteResult(
            mode=mode,
            restraints_path=None,
            receptor_restraints=[],
            peptide_restraints=[],
            pocket_center=pocket_center,
            details=details,
        )
