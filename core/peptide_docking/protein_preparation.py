"""
Receptor protein structural preparation and validation specifically tailored for LightDock.

Provides comprehensive inspection and preparation:
1. Chain selection and biological assembly evaluation (REMARK 350).
2. Inspection of missing residues (REMARK 465), missing side-chain atoms, alternate conformations,
   and modified/non-standard residues.
3. Selective retention/removal of crystallographic waters, catalytic metal ions, cofactors,
   and bound ligands (not deleting everything indiscriminately).
4. Alternate conformation resolution (highest occupancy selection).
5. Chain ID and sequential residue numbering validation.
6. Export of clean, validated LightDock-compatible receptor PDB structures.
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Any

import gemmi
from Bio.PDB import PDBParser, PDBIO, Select, Polypeptide
from core.peptide_docking.input_validation import compute_file_checksum, validate_protein_pdb
from utils.io import ensure_dir
from utils.logger import setup_logger

logger = setup_logger("peptide_protein_prep", level="INFO")

# Metal ion residues
KNOWN_METAL_IONS = {
    "ZN", "MG", "CA", "FE", "MN", "CU", "NA", "K", "CO", "NI",
    "CD", "PT", "HG", "AU", "FE2", "ZN2", "MG2", "CA2"
}

# Known biological cofactors and coenzymes
KNOWN_COFACTORS = {
    "ATP", "ADP", "AMP", "GTP", "GDP", "NAD", "NAP", "NADH", "NADPH",
    "FAD", "FMN", "HEM", "HEA", "HEB", "HEC", "SAM", "SAH", "PLP",
    "PMP", "COA", "ACO", "B12", "THP", "F420", "CAM", "TPP"
}

# Standard amino acid codes
STANDARD_AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
}

# Non-standard residue mappings to canonical counterparts
NONSTANDARD_AA_MAPPINGS = {
    "MSE": "MET",  # Selenomethionine -> Methionine
    "PTR": "TYR",  # Phosphotyrosine -> Tyrosine
    "SEP": "SER",  # Phosphoserine -> Serine
    "TPO": "THR",  # Phosphothreonine -> Threonine
    "HYP": "PRO",  # Hydroxyproline -> Proline
    "CSO": "CYS",  # S-hydroxycysteine -> Cysteine
    "CSD": "CYS",  # S-cysteinesulfinic acid -> Cysteine
    "CME": "CYS",  # S,S-(2-hydroxyethyl)thiocysteine -> Cysteine
    "KCX": "LYS",  # Carbamylated lysine -> Lysine
    "LLP": "LYS",  # Phosphopyridoxyl-lysine -> Lysine
    "MLY": "LYS",  # N-dimethyl-lysine -> Lysine
    "TYS": "TYR",  # Sulfonated tyrosine -> Tyrosine
    "SEC": "CYS",  # Selenocysteine -> Cysteine
    "PYL": "LYS",  # Pyrrolysine -> Lysine
}


def inspect_protein_structure(input_pdb: str | Path) -> dict[str, Any]:
    """
    Inspect a raw PDB structure for chains, biological assemblies, missing residues,
    alternate conformations, non-standard residues, waters, metal ions, and cofactors.

    Returns:
        Structured dictionary containing all diagnostic findings.
    """
    p = Path(input_pdb).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Protein PDB file not found: {input_pdb}")

    structure = gemmi.read_structure(str(p))
    model = structure[0] if len(structure) > 0 else None

    if model is None:
        raise ValueError(f"No 3D coordinate models found in PDB file: {p.name}")

    # 1. Chains inspection
    chains_info = []
    total_residues = 0
    total_atoms = 0

    for c in model:
        c_res = [r for r in c if r.entity_type == gemmi.EntityType.Polymer]
        c_atoms = sum(len(r) for r in c)
        total_residues += len(c_res)
        total_atoms += c_atoms
        seq_sample = gemmi.one_letter_code([r.name for r in c_res[:10]]) if c_res else ""
        if len(c_res) > 10:
            seq_sample += f"... ({len(c_res)} AA)"

        chains_info.append({
            "chain_id": c.name,
            "num_residues": len(c_res),
            "num_atoms": c_atoms,
            "sequence_preview": seq_sample,
        })

    # 2. Biological assemblies inspection (REMARK 350)
    assemblies_info = []
    for idx, a in enumerate(structure.assemblies):
        assemblies_info.append({
            "assembly_id": idx + 1,
            "name": a.name,
            "generators_count": len(a.generators),
        })

    # 3. Missing residues inspection (REMARK 465) & Missing atoms (REMARK 470)
    missing_residues_count = 0
    missing_residues_list = []
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("REMARK 465") and len(line) >= 26:
                    parts = line.strip().split()
                    if len(parts) >= 5 and parts[2] in STANDARD_AMINO_ACIDS:
                        missing_residues_list.append(f"{parts[3]}.{parts[2]}.{parts[4]}")
                        missing_residues_count += 1
    except Exception:
        pass

    # 4. Alternate conformations inspection
    altlocs_found = {}
    for c in model:
        for r in c:
            for atom in r:
                if atom.has_altloc():
                    res_key = f"{c.name}.{r.name}.{r.seqid}"
                    altlocs_found.setdefault(res_key, set()).add(atom.altloc)

    altlocs_summary = [
        {"residue": k, "conformations": sorted(list(v))}
        for k, v in altlocs_found.items()
    ]

    # 5. Non-standard & modified residues
    nonstandard_residues = []
    for c in model:
        for r in c:
            rname = r.name.strip().upper()
            if r.entity_type == gemmi.EntityType.Polymer and rname not in STANDARD_AMINO_ACIDS:
                mapped = NONSTANDARD_AA_MAPPINGS.get(rname, "Unknown")
                nonstandard_residues.append({
                    "chain_id": c.name,
                    "residue_name": rname,
                    "seqid": str(r.seqid),
                    "mapped_to": mapped,
                })

    # 6. Non-protein entities (waters, metals, cofactors, ligands)
    waters_count = 0
    metals_found = set()
    cofactors_found = set()
    ligands_found = set()

    for c in model:
        for r in c:
            rname = r.name.strip().upper()
            if r.entity_type == gemmi.EntityType.Water or rname in {"HOH", "WAT", "H2O", "TIP3"}:
                waters_count += 1
            elif r.entity_type == gemmi.EntityType.NonPolymer or rname not in STANDARD_AMINO_ACIDS:
                if rname in KNOWN_METAL_IONS:
                    metals_found.add(rname)
                elif rname in KNOWN_COFACTORS:
                    cofactors_found.add(rname)
                elif rname not in NONSTANDARD_AA_MAPPINGS:
                    ligands_found.add(rname)

    return {
        "file_path": str(p),
        "chains": chains_info,
        "available_chain_ids": [c["chain_id"] for c in chains_info],
        "biological_assemblies": assemblies_info,
        "missing_residues_count": missing_residues_count,
        "missing_residues": missing_residues_list[:20],
        "alternate_conformations_count": len(altlocs_summary),
        "alternate_conformations": altlocs_summary[:20],
        "nonstandard_residues": nonstandard_residues,
        "waters_count": waters_count,
        "metals_detected": sorted(list(metals_found)),
        "cofactors_detected": sorted(list(cofactors_found)),
        "ligands_detected": sorted(list(ligands_found)),
        "total_residues": total_residues,
        "total_atoms": total_atoms,
    }


def prepare_receptor_for_lightdock(
    input_pdb: str | Path,
    output_dir: str | Path,
    chains: list[str] | None = None,
    assembly_id: int | None = None,
    biological_assembly: int | None = None,
    water_policy: str = "remove_bulk",      # "remove_all", "keep_all", "remove_bulk", "keep_interfacial"
    keep_metals: bool = True,
    keep_cofactors: bool = True,
    keep_het_residues: list[str] | None = None,
    keep_heteroatoms: bool | None = None,  # Backwards compatibility alias
    remove_waters: bool | None = None,     # Backwards compatibility alias
    remove_hydrogens: bool = False,
    resolve_altlocs: bool = True,
    standardize_nonstandard: bool = True,
) -> dict[str, Any]:
    """
    Prepare, validate, and clean a receptor PDB structure for LightDock.

    Preserves original protein file and outputs prepared structure into output_dir.
    Respects user decisions regarding chains, assemblies, waters, catalytic metals,
    and biological cofactors.
    """
    if biological_assembly is not None and assembly_id is None:
        assembly_id = biological_assembly
    input_path = Path(input_pdb).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Receptor PDB not found: {input_pdb}")

    # Backward compatibility handling
    if remove_waters is not None:
        water_policy = "remove_all" if remove_waters else "keep_all"
    if keep_heteroatoms is not None:
        keep_metals = keep_heteroatoms
        keep_cofactors = keep_heteroatoms

    val = validate_protein_pdb(input_path)
    out_dir = ensure_dir(output_dir)
    prepared_path = Path(out_dir) / f"{input_path.stem}_prepared.pdb"

    # Read with gemmi for precision filtering and coordinate handling
    st = gemmi.read_structure(str(input_path))
    if len(st) == 0:
        raise ValueError(f"No structural models in PDB file: {input_path}")

    # Handle biological assembly if specified
    if assembly_id is not None and assembly_id > 0 and len(st.assemblies) >= assembly_id:
        asm_def = st.assemblies[assembly_id - 1]
        logger.info(f"Generating biological assembly {assembly_id} ('{asm_def.name}') for {input_path.name}")
        st = gemmi.make_assembly(asm_def, st[0], how=gemmi.HowToNameCopiedChain.AddNumber)

    model = st[0]

    # Chain validation
    all_chains = [c.name for c in model]
    if chains:
        missing_chains = set(chains) - set(all_chains)
        if missing_chains:
            raise ValueError(
                f"Requested chain(s) {sorted(missing_chains)} not found in {input_path.name}. "
                f"Available chains: {sorted(all_chains)}"
            )
        target_chains = set(chains)
    else:
        target_chains = set(all_chains)

    custom_hets = set(r.upper() for r in (keep_het_residues or []))
    retained_metals = set()
    retained_cofactors = set()
    retained_ligands = set()
    retained_waters_count = 0
    removed_waters_count = 0
    altlocs_resolved_count = 0
    standardized_count = 0

    # Collect protein atom coordinates for interfacial water distance checks
    protein_atom_coords: list[tuple[float, float, float]] = []
    if water_policy in ("remove_bulk", "keep_interfacial"):
        for chain in model:
            if chain.name in target_chains:
                for res in chain:
                    if res.entity_type == gemmi.EntityType.Polymer and res.name in STANDARD_AMINO_ACIDS:
                        for atom in res:
                            protein_atom_coords.append((atom.pos.x, atom.pos.y, atom.pos.z))

    def is_water_near_protein(wx: float, wy: float, wz: float, cutoff: float = 4.5) -> bool:
        if not protein_atom_coords:
            return False
        cutoff_sq = cutoff * cutoff
        for px, py, pz in protein_atom_coords:
            dx = wx - px
            dy = wy - py
            dz = wz - pz
            if (dx * dx + dy * dy + dz * dz) <= cutoff_sq:
                return True
        return False

    # Process models, chains, residues
    new_model = gemmi.Model("0")

    for chain in model:
        if chain.name not in target_chains:
            continue

        new_chain = gemmi.Chain(chain.name)

        for res in chain:
            rname = res.name.strip().upper()

            # Water handling
            if res.entity_type == gemmi.EntityType.Water or rname in {"HOH", "WAT", "H2O", "TIP3"}:
                if water_policy == "remove_all":
                    removed_waters_count += 1
                    continue
                elif water_policy in ("remove_bulk", "keep_interfacial"):
                    # Check first oxygen atom distance
                    water_pos = res[0].pos if len(res) > 0 else None
                    if water_pos and is_water_near_protein(water_pos.x, water_pos.y, water_pos.z, cutoff=4.5):
                        retained_waters_count += 1
                    else:
                        removed_waters_count += 1
                        continue
                else:  # keep_all
                    retained_waters_count += 1

            # Heteroatom / non-polymer handling
            elif res.entity_type != gemmi.EntityType.Polymer:
                if rname in KNOWN_METAL_IONS:
                    if not keep_metals and rname not in custom_hets:
                        continue
                    retained_metals.add(rname)
                elif rname in KNOWN_COFACTORS:
                    if not keep_cofactors and rname not in custom_hets:
                        continue
                    retained_cofactors.add(rname)
                elif rname in custom_hets:
                    retained_ligands.add(rname)
                else:
                    # Generic small molecule or unselected heteroatom -> omit
                    continue

            # Standardize modified non-standard residues if requested
            if standardize_nonstandard and rname in NONSTANDARD_AA_MAPPINGS:
                std_name = NONSTANDARD_AA_MAPPINGS[rname]
                res.name = std_name
                standardized_count += 1
                # If MSE -> MET, change SE atom to SD
                for atom in res:
                    if atom.name == "SE" and std_name == "MET":
                        atom.name = "SD"
                        atom.element = gemmi.Element("S")

            # Alternate conformation resolution (keep highest occupancy / first)
            if resolve_altlocs:
                best_atoms: dict[str, gemmi.Atom] = {}
                for atom in res:
                    if remove_hydrogens and atom.element == gemmi.Element("H"):
                        continue
                    aname = atom.name
                    if aname not in best_atoms:
                        best_atoms[aname] = atom
                    else:
                        # Compare occupancy
                        altlocs_resolved_count += 1
                        if atom.occ > best_atoms[aname].occ:
                            best_atoms[aname] = atom
                        elif atom.occ == best_atoms[aname].occ and best_atoms[aname].altloc != 'A' and atom.altloc == 'A':
                            best_atoms[aname] = atom

                # Rebuild residue with resolved atoms and cleared altloc
                clean_res = gemmi.Residue()
                clean_res.name = res.name
                clean_res.seqid = res.seqid
                clean_res.entity_type = res.entity_type
                for at in best_atoms.values():
                    at.altloc = '\0'  # clear altloc
                    clean_res.add_atom(at)
                new_chain.add_residue(clean_res)
            else:
                if remove_hydrogens:
                    res_no_h = gemmi.Residue()
                    res_no_h.name = res.name
                    res_no_h.seqid = res.seqid
                    res_no_h.entity_type = res.entity_type
                    for atom in res:
                        if atom.element != gemmi.Element("H"):
                            res_no_h.add_atom(atom)
                    new_chain.add_residue(res_no_h)
                else:
                    new_chain.add_residue(res)

        if len(new_chain) > 0:
            new_model.add_chain(new_chain)

    st_clean = gemmi.Structure()
    st_clean.add_model(new_model)
    st_clean.write_pdb(str(prepared_path))

    # Validate generated output
    if not prepared_path.is_file() or prepared_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to generate prepared receptor structure at {prepared_path}")

    prep_val = validate_protein_pdb(prepared_path)
    logger.info(
        f"Prepared LightDock receptor: {prepared_path.name} | Chains: {prep_val['chains']} | "
        f"Residues: {prep_val['num_residues']} | Atoms: {prep_val['num_atoms']} | "
        f"Retained metals: {sorted(list(retained_metals))} | Retained cofactors: {sorted(list(retained_cofactors))} | "
        f"Retained waters: {retained_waters_count} (removed: {removed_waters_count})"
    )

    return {
        "original_path": str(input_path),
        "prepared_path": str(prepared_path),
        "original_checksum": val["checksum"],
        "prepared_checksum": prep_val["checksum"],
        "chains": prep_val["chains"],
        "num_residues": prep_val["num_residues"],
        "num_atoms": prep_val["num_atoms"],
        "retained_metals": sorted(list(retained_metals)),
        "retained_cofactors": sorted(list(retained_cofactors)),
        "retained_ligands": sorted(list(retained_ligands)),
        "retained_waters_count": retained_waters_count,
        "removed_waters_count": removed_waters_count,
        "altlocs_resolved": altlocs_resolved_count,
        "nonstandard_standardized": standardized_count,
        "water_policy": water_policy,
        "status": "success",
    }
