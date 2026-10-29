"""
Structural pose analysis for protein–peptide complexes.
Calculates inter-chain contacts, clashes, hydrogen-bond candidates, salt bridges,
and interface residue mappings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import numpy as np
from Bio.PDB import PDBParser

from utils.logger import setup_logger

logger = setup_logger("pose_analysis", level="INFO")

POS_CHARGED_RES = {"ARG", "LYS", "HIS"}
NEG_CHARGED_RES = {"ASP", "GLU"}
HBOND_DONORS_ACCEPTORS = {"N", "O", "S"}


def analyze_docked_pose(
    pose_pdb_path: str | Path,
    receptor_chain: str = "A",
    peptide_chain: str = "B",
    restraints: list[str] | None = None,
    clash_cutoff: float = 2.0,
    contact_cutoff: float = 4.5,
    hbond_cutoff: float = 3.5,
    salt_bridge_cutoff: float = 4.0,
) -> dict[str, Any]:
    """
    Perform structural contact, clash, and interface analysis on a docked complex.
    """
    path = Path(pose_pdb_path)
    if not path.is_file():
        raise FileNotFoundError(f"Pose PDB file not found: {pose_pdb_path}")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", str(path))

    rec_atoms = []
    pep_atoms = []

    for model in structure:
        for chain in model:
            is_pep = (chain.id == peptide_chain)
            for res in chain:
                if res.id[0] != " ":
                    continue
                resname = res.get_resname().strip().upper()
                resnum = res.id[1]
                chain_id = chain.id
                for atom in res:
                    # Ignore hydrogens for clash / contact distances
                    element = atom.element.strip().upper() if atom.element else atom.name[0].upper()
                    if element == "H":
                        continue
                    atom_info = {
                        "name": atom.name.strip(),
                        "element": element,
                        "coord": atom.get_coord(),
                        "resname": resname,
                        "resnum": resnum,
                        "chain": chain_id,
                        "residue_id": f"{chain_id}.{resname}.{resnum}",
                    }
                    if is_pep or chain_id == peptide_chain:
                        pep_atoms.append(atom_info)
                    else:
                        rec_atoms.append(atom_info)
        break

    if not rec_atoms or not pep_atoms:
        return {
            "pose_path": str(path),
            "clashes_count": 0,
            "contacts_count": 0,
            "hbonds_count": 0,
            "salt_bridges_count": 0,
            "interface_receptor_residues": [],
            "interface_peptide_residues": [],
            "interface_receptor_count": 0,
            "interface_peptide_count": 0,
            "restraints_satisfied": 0,
            "restraints_total": len(restraints) if restraints else 0,
            "status": "warning_empty_chains",
        }

    rec_coords = np.array([a["coord"] for a in rec_atoms])
    pep_coords = np.array([a["coord"] for a in pep_atoms])

    # Distance matrix (M x N)
    # rec_coords: (M, 3), pep_coords: (N, 3)
    diff = rec_coords[:, np.newaxis, :] - pep_coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    # 1. Clashes
    clash_pairs = np.argwhere(dist_matrix < clash_cutoff)
    num_clashes = len(clash_pairs)

    # 2. Contacts
    contact_pairs = np.argwhere(dist_matrix < contact_cutoff)
    num_contacts = len(contact_pairs)

    rec_interface_residues = set()
    pep_interface_residues = set()
    for r_idx, p_idx in contact_pairs:
        rec_interface_residues.add(rec_atoms[r_idx]["residue_id"])
        pep_interface_residues.add(pep_atoms[p_idx]["residue_id"])

    # 3. Hydrogen bonds candidates (N/O pairs within cutoff)
    num_hbonds = 0
    for r_idx, p_idx in contact_pairs:
        if dist_matrix[r_idx, p_idx] <= hbond_cutoff:
            elem_r = rec_atoms[r_idx]["element"]
            elem_p = pep_atoms[p_idx]["element"]
            if elem_r in HBOND_DONORS_ACCEPTORS and elem_p in HBOND_DONORS_ACCEPTORS:
                num_hbonds += 1

    # 4. Salt bridges
    salt_bridges = 0
    checked_res_pairs = set()
    for r_idx, p_idx in contact_pairs:
        if dist_matrix[r_idx, p_idx] <= salt_bridge_cutoff:
            res_r = rec_atoms[r_idx]["resname"]
            res_p = pep_atoms[p_idx]["resname"]
            pair_key = (rec_atoms[r_idx]["residue_id"], pep_atoms[p_idx]["residue_id"])
            if pair_key not in checked_res_pairs:
                if (res_r in POS_CHARGED_RES and res_p in NEG_CHARGED_RES) or \
                   (res_r in NEG_CHARGED_RES and res_p in POS_CHARGED_RES):
                    salt_bridges += 1
                    checked_res_pairs.add(pair_key)

    # 5. Restraints satisfaction
    satisfied_count = 0
    total_restraints = len(restraints) if restraints else 0
    if restraints:
        for r_str in restraints:
            # Check if this restraint residue is in the interface
            # format could be A.TYR.45 or R A.TYR.45
            norm_r = r_str.replace("R ", "").replace("L ", "").strip()
            if norm_r in rec_interface_residues or norm_r in pep_interface_residues:
                satisfied_count += 1

    return {
        "pose_path": str(path.resolve()),
        "clashes_count": num_clashes,
        "contacts_count": num_contacts,
        "hbonds_count": num_hbonds,
        "salt_bridges_count": salt_bridges,
        "interface_receptor_residues": sorted(list(rec_interface_residues)),
        "interface_peptide_residues": sorted(list(pep_interface_residues)),
        "interface_receptor_count": len(rec_interface_residues),
        "interface_peptide_count": len(pep_interface_residues),
        "restraints_satisfied": satisfied_count,
        "restraints_total": total_restraints,
        "restraint_satisfaction_rate": (satisfied_count / total_restraints) if total_restraints > 0 else 1.0,
        "status": "valid" if num_clashes <= 20 else "severe_clashes",
    }
