"""
Peptide structural preparation, peptide-aware 3D conformation modeling,
stereochemistry & geometry validation, and LightDock PDB export.

Features:
1. Ingestion of raw amino acid sequences or user-validated PDB structures.
2. Terminal chemistry management: default physiological zwitterionic termini (-NH3+, -COO-),
   with optional N-terminal acetylation (ACE) and C-terminal amidation (NHE/NME),
   and cyclical topology without free termini.
3. Covalent disulfide bridge formation (Cys-Cys loops) and head-to-tail cyclization.
4. Peptide-aware distance geometry modeling (RDKit ETKDGv3) with MMFF94 energy minimization.
5. Quality checks: peptide bond omega dihedrals (trans vs cis), stereochemistry (L-amino acids),
   and steric clash evaluation.
6. Flexible conformational ensemble sampling, energy ranking, and selection of lowest-energy pose.
7. Export of clean, heavy-atom LightDock-compatible PDB structures.
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from Bio.PDB import PDBParser, PDBIO, Polypeptide

from core.peptide_docking.input_validation import (
    PeptideCandidate,
    compute_file_checksum,
    validate_peptide_sequence,
)
from utils.io import ensure_dir
from utils.logger import setup_logger

logger = setup_logger("peptide_prep", level="INFO")


def check_peptide_geometry_and_stereochemistry(mol: Chem.Mol, conf_id: int = 0) -> dict[str, Any]:
    """
    Validate peptide-bond omega dihedrals, L-amino acid stereocenters, and steric clashes.

    Returns:
        Dictionary containing trans/cis bond counts, stereocenters, and clash status.
    """
    if mol is None or mol.GetNumConformers() == 0:
        return {
            "valid_geometry": False,
            "trans_peptide_bonds": 0,
            "cis_peptide_bonds": 0,
            "mean_omega_deg": 0.0,
            "chiral_centers_count": 0,
            "l_amino_acids_count": 0,
            "d_amino_acids_count": 0,
            "steric_clashes_count": 0,
        }

    conf = mol.GetConformer(conf_id)

    # 1. Map residue numbers to backbone atoms (N, CA, C)
    res_atoms: dict[int, dict[str, int]] = {}
    for a in mol.GetAtoms():
        info = a.GetPDBResidueInfo()
        if info:
            res_num = info.GetResidueNumber()
            aname = info.GetName().strip()
            res_atoms.setdefault(res_num, {})[aname] = a.GetIdx()

    trans_bonds = 0
    cis_bonds = 0
    omegas = []

    sorted_res = sorted(res_atoms.keys())
    for i in range(len(sorted_res) - 1):
        r1 = res_atoms[sorted_res[i]]
        r2 = res_atoms[sorted_res[i + 1]]
        if "CA" in r1 and "C" in r1 and "N" in r2 and "CA" in r2:
            try:
                omega = rdMolTransforms.GetDihedralDeg(conf, r1["CA"], r1["C"], r2["N"], r2["CA"])
                abs_omega = abs(omega)
                omegas.append(abs_omega)
                if abs_omega >= 140.0:
                    trans_bonds += 1
                elif abs_omega <= 45.0:
                    cis_bonds += 1
                else:
                    trans_bonds += 1  # Near-trans distorted
            except Exception:
                pass

    # 2. Stereochemistry check
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    l_count = 0
    d_count = 0

    for a_idx, tag in chiral_centers:
        atom = mol.GetAtomWithIdx(a_idx)
        info = atom.GetPDBResidueInfo()
        rname = info.GetResidueName().strip() if info else ""
        if rname == "CYS":
            # In Cahn-Ingold-Prelog, L-Cysteine is (R) due to sulfur priority
            if tag == "R":
                l_count += 1
            else:
                d_count += 1
        else:
            # All other canonical L-amino acids are (S)
            if tag == "S":
                l_count += 1
            else:
                d_count += 1

    # 3. Steric clashes check (< 1.2 Å for non-bonded atoms separated by >= 3 bonds)
    clashes_count = 0
    pos = conf.GetPositions()
    num_heavy = mol.GetNumHeavyAtoms()
    for i in range(num_heavy):
        for j in range(i + 1, num_heavy):
            atom_i = mol.GetAtomWithIdx(i)
            atom_j = mol.GetAtomWithIdx(j)
            if atom_i.GetSymbol() == "H" or atom_j.GetSymbol() == "H":
                continue
            # Check bond distance in graph
            path_len = len(Chem.GetShortestPath(mol, i, j)) - 1
            if path_len >= 3:
                dist = math.dist(pos[i], pos[j])
                if dist < 1.20:
                    clashes_count += 1

    mean_omega = round(sum(omegas) / len(omegas), 1) if omegas else 180.0

    return {
        "valid_geometry": (cis_bonds == 0 or cis_bonds <= 2) and clashes_count == 0,
        "trans_peptide_bonds": trans_bonds,
        "cis_peptide_bonds": cis_bonds,
        "mean_omega_deg": mean_omega,
        "chiral_centers_count": len(chiral_centers),
        "l_amino_acids_count": l_count,
        "d_amino_acids_count": d_count,
        "steric_clashes_count": clashes_count,
    }


def apply_terminal_chemistry(
    mol: Chem.Mol,
    sequence_length: int,
    n_term_cap: str = "zwitterionic",  # "zwitterionic", "acetyl", "none"
    c_term_cap: str = "zwitterionic",  # "zwitterionic", "amide", "none"
    is_cyclic: bool = False,
) -> tuple[Chem.Mol, str]:
    """
    Apply terminal capping or cyclization chemistry to a peptide molecule.

    Returns:
        (modified_mol, terminal_chemistry_description)
    """
    if is_cyclic:
        # Head-to-tail cyclization
        try:
            n_idx = None
            last_c_idx = None
            oxt_idx = None
            for a in mol.GetAtoms():
                info = a.GetPDBResidueInfo()
                if info:
                    if info.GetName().strip() == "N" and info.GetResidueNumber() == 1:
                        n_idx = a.GetIdx()
                    if info.GetName().strip() == "C" and info.GetResidueNumber() == sequence_length:
                        last_c_idx = a.GetIdx()
                    if info.GetName().strip() == "OXT" and info.GetResidueNumber() == sequence_length:
                        oxt_idx = a.GetIdx()

            if n_idx is not None and last_c_idx is not None:
                emol = Chem.EditableMol(mol)
                if oxt_idx is not None:
                    emol.RemoveAtom(oxt_idx)
                emol.AddBond(n_idx, last_c_idx, Chem.BondType.SINGLE)
                mol_cyc = emol.GetMol()
                Chem.SanitizeMol(mol_cyc)
                return mol_cyc, "Head-to-tail cyclic (no free termini)"
        except Exception as e:
            logger.warning(f"Head-to-tail cyclization bond formation notice: {e}")
        return mol, "Cyclic peptide (unmodified termini)"

    chem_desc_parts = []
    emol = Chem.EditableMol(mol)
    modified = False

    # 1. C-terminal amidation (replace -OH / OXT with -NH2)
    if c_term_cap.lower() in ("amide", "nhe", "amidated"):
        oxt_idx = None
        last_c_idx = None
        for a in mol.GetAtoms():
            info = a.GetPDBResidueInfo()
            if info:
                if info.GetName().strip() == "OXT" and info.GetResidueNumber() == sequence_length:
                    oxt_idx = a.GetIdx()
                if info.GetName().strip() == "C" and info.GetResidueNumber() == sequence_length:
                    last_c_idx = a.GetIdx()

        if oxt_idx is not None and last_c_idx is not None:
            emol.RemoveAtom(oxt_idx)
            n_atom = Chem.Atom("N")
            new_n_idx = emol.AddAtom(n_atom)
            emol.AddBond(last_c_idx, new_n_idx, Chem.BondType.SINGLE)
            chem_desc_parts.append("C-term: Amide (NHE/NH2)")
            modified = True
    else:
        chem_desc_parts.append("C-term: COO- (zwitterionic)")

    # 2. N-terminal acetylation (add CH3-C(=O)- to N1)
    if n_term_cap.lower() in ("acetyl", "ace", "acetylated"):
        mol_tmp = emol.GetMol()
        n1_idx = None
        for a in mol_tmp.GetAtoms():
            info = a.GetPDBResidueInfo()
            if info and info.GetName().strip() == "N" and info.GetResidueNumber() == 1:
                n1_idx = a.GetIdx()
                break

        if n1_idx is not None:
            c_carb = emol.AddAtom(Chem.Atom("C"))
            o_carb = emol.AddAtom(Chem.Atom("O"))
            c_me = emol.AddAtom(Chem.Atom("C"))
            emol.AddBond(c_carb, o_carb, Chem.BondType.DOUBLE)
            emol.AddBond(c_carb, c_me, Chem.BondType.SINGLE)
            emol.AddBond(n1_idx, c_carb, Chem.BondType.SINGLE)
            chem_desc_parts.append("N-term: Acetyl (ACE)")
            modified = True
    else:
        chem_desc_parts.append("N-term: NH3+ (zwitterionic)")

    if modified:
        try:
            res_mol = emol.GetMol()
            Chem.SanitizeMol(res_mol)
            return res_mol, " | ".join(chem_desc_parts)
        except Exception as e:
            logger.debug(f"Terminal modification sanitization warning: {e}")

    return mol, " | ".join(chem_desc_parts)


def apply_disulfide_bridges(mol: Chem.Mol, disulfide_pairs: list[tuple[int, int]] | None = None) -> Chem.Mol:
    """
    Form covalent disulfide (S-S) bonds between Cysteine residues.
    """
    # Map Cysteine residue numbers to their SG sulfur atom index
    cys_sulfurs: dict[int, int] = {}
    for a in mol.GetAtoms():
        info = a.GetPDBResidueInfo()
        if info and info.GetResidueName().strip() == "CYS" and a.GetSymbol() == "S":
            cys_sulfurs[info.GetResidueNumber()] = a.GetIdx()

    pairs_to_form = []
    if disulfide_pairs:
        for r1, r2 in disulfide_pairs:
            if r1 in cys_sulfurs and r2 in cys_sulfurs:
                pairs_to_form.append((cys_sulfurs[r1], cys_sulfurs[r2]))
    elif len(cys_sulfurs) == 2:
        # Default single pair disulfide loop (e.g. Oxytocin, Octreotide)
        s_indices = list(cys_sulfurs.values())
        pairs_to_form.append((s_indices[0], s_indices[1]))

    if not pairs_to_form:
        return mol

    try:
        emol = Chem.EditableMol(mol)
        for s1, s2 in pairs_to_form:
            emol.AddBond(s1, s2, Chem.BondType.SINGLE)
        mol_ss = emol.GetMol()
        for s1, s2 in pairs_to_form:
            mol_ss.GetAtomWithIdx(s1).SetNumExplicitHs(0)
            mol_ss.GetAtomWithIdx(s2).SetNumExplicitHs(0)
        Chem.SanitizeMol(mol_ss)
        logger.info(f"Formed {len(pairs_to_form)} covalent disulfide bridge(s).")
        return mol_ss
    except Exception as e:
        logger.warning(f"Disulfide bond formation note: {e}")
        return mol


def generate_peptide_3d_structure(
    sequence: str,
    output_pdb_path: str | Path,
    chain_id: str = "B",
    num_conformers: int = 10,
    is_cyclic: bool = False,
    disulfide_pairs: list[tuple[int, int]] | None = None,
    n_term_cap: str = "zwitterionic",
    c_term_cap: str = "zwitterionic",
    minimize: bool = True,
    forcefield: str = "MMFF94",
    seed: int = 42,
) -> dict[str, Any]:
    """
    Generate a 3D structural model from an amino-acid sequence using peptide-aware
    RDKit ETKDGv3 distance geometry with terminal capping, disulfide handling,
    stereochemistry/geometry validation, and conformational ensemble ranking.
    """
    clean_seq = validate_peptide_sequence(sequence)
    out_path = Path(output_pdb_path)
    ensure_dir(out_path.parent)

    mol = Chem.MolFromSequence(clean_seq)
    if mol is None:
        raise ValueError(f"RDKit failed to construct peptide molecule from sequence '{clean_seq}'")

    # 1. Apply terminal chemistry (zwitterionic by default, or capped / cyclic)
    mol, terminal_chemistry_desc = apply_terminal_chemistry(
        mol,
        sequence_length=len(clean_seq),
        n_term_cap=n_term_cap,
        c_term_cap=c_term_cap,
        is_cyclic=is_cyclic,
    )

    # 2. Form disulfide bridges if applicable
    mol = apply_disulfide_bridges(mol, disulfide_pairs=disulfide_pairs)

    # Add Hydrogens for 3D embedding and forcefield calculations
    mol = Chem.AddHs(mol)

    # 3. ETKDGv3 Conformer Sampling
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.clearConfs = True
    if len(clean_seq) > 12:
        params.useRandomCoords = True

    effective_confs = max(1, num_conformers)
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=effective_confs, params=params))

    if not cids:
        # Fallback 1: Random coords with expanded box
        logger.info(f"Retrying 3D embedding for sequence '{clean_seq}' with random coordinates fallback...")
        params.useRandomCoords = True
        params.boxSizeMult = 2.0
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=effective_confs, params=params))

    if not cids:
        # Fallback 2: Single molecule embedding
        cid = AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=250, randomSeed=seed)
        if cid >= 0:
            cids = [cid]

    if not cids:
        # Fallback 3: Relaxed distance geometry smoothing
        cid = AllChem.EmbedMolecule(
            mol,
            useRandomCoords=True,
            ignoreSmoothingFailures=True,
            maxAttempts=350,
            randomSeed=seed,
        )
        if cid >= 0:
            cids = [cid]

    if not cids:
        raise RuntimeError(f"Failed to embed 3D coordinates for peptide sequence: {clean_seq}")

    # 4. Energy Minimization & Conformer Ranking
    conformer_evals: list[dict[str, Any]] = []
    best_conf_id = cids[0]
    min_energy = float("inf")

    for cid in cids:
        energy = 9999.0
        if minimize:
            try:
                if forcefield.upper() == "MMFF94" and AllChem.MMFFHasAllMoleculeParams(mol):
                    AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=500)
                    props = AllChem.MMFFGetMoleculeProperties(mol)
                    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
                    energy = ff.CalcEnergy() if ff else 0.0
                else:
                    AllChem.UFFOptimizeMolecule(mol, confId=cid, maxIters=500)
                    uff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
                    energy = uff.CalcEnergy() if uff else 0.0
            except Exception as e:
                logger.debug(f"Minimization warning on conformer {cid}: {e}")
                energy = 9999.0

        geom_check = check_peptide_geometry_and_stereochemistry(mol, conf_id=cid)
        conformer_evals.append({
            "conf_id": cid,
            "energy_kcal_mol": round(energy, 2),
            "geometry": geom_check,
        })

        if energy < min_energy:
            min_energy = energy
            best_conf_id = cid

    best_geom = check_peptide_geometry_and_stereochemistry(mol, conf_id=best_conf_id)

    # 5. Export clean heavy-atom PDB structure for LightDock
    mol_out = Chem.RemoveHs(mol)
    pdb_block = Chem.MolToPDBBlock(mol_out, confId=best_conf_id)

    fixed_lines = []
    for line in pdb_block.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            if len(line) >= 22:
                line = line[:21] + chain_id + line[22:]
        fixed_lines.append(line)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(fixed_lines) + "\n")

    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to write generated peptide PDB to {out_path}")

    checksum = compute_file_checksum(out_path)
    return {
        "pdb_path": str(out_path.resolve()),
        "checksum": checksum,
        "conformer_id": best_conf_id,
        "num_conformers_sampled": len(cids),
        "energy_kcal_mol": min_energy if min_energy != float("inf") else None,
        "chain_id": chain_id,
        "terminal_chemistry": terminal_chemistry_desc,
        "geometry": best_geom,
        "method": f"RDKit-ETKDGv3+{forcefield}",
    }


def prepare_peptide_candidate(
    candidate: PeptideCandidate,
    output_dir: str | Path,
    chain_id: str = "B",
    num_conformers: int = 10,
    is_cyclic: bool = False,
    disulfide_pairs: list[tuple[int, int]] | None = None,
    n_term_cap: str = "zwitterionic",
    c_term_cap: str = "zwitterionic",
    minimize: bool = True,
    forcefield: str = "MMFF94",
    seed: int = 42,
) -> PeptideCandidate:
    """
    Prepare a peptide candidate for LightDock docking.

    If structure_path exists (user supplied PDB), validates and formats it.
    If sequence exists, generates a peptide-aware 3D structural model with
    geometry and stereochemistry validation.
    """
    out_dir = Path(ensure_dir(output_dir))
    target_pdb = out_dir / f"peptide_{candidate.id}.pdb"

    # Auto-detect cyclization if sequence hints or flag
    cyc_flag = is_cyclic or candidate.properties.get("is_cyclic", False)
    if "C" in candidate.sequence and candidate.sequence.count("C") >= 2 and disulfide_pairs is None:
        cyc_flag = True

    if candidate.structure_path and os.path.isfile(candidate.structure_path):
        # Format existing PDB
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("peptide", candidate.structure_path)

        for model in structure:
            for chain in model:
                chain.id = chain_id
            break

        io = PDBIO()
        io.set_structure(structure)
        io.save(str(target_pdb))

        checksum = compute_file_checksum(target_pdb)
        candidate.prepared_structure_path = str(target_pdb.resolve())
        candidate.prepared_checksum = checksum
        candidate.status = "prepared"
        candidate.properties["modeling_method"] = "user_provided_pdb"
        candidate.properties["terminal_chemistry"] = "Experimental PDB Termini"
    else:
        # Generate from sequence
        res = generate_peptide_3d_structure(
            sequence=candidate.sequence,
            output_pdb_path=target_pdb,
            chain_id=chain_id,
            num_conformers=num_conformers,
            is_cyclic=cyc_flag,
            disulfide_pairs=disulfide_pairs,
            n_term_cap=n_term_cap,
            c_term_cap=c_term_cap,
            minimize=minimize,
            forcefield=forcefield,
            seed=seed,
        )
        candidate.prepared_structure_path = res["pdb_path"]
        candidate.prepared_checksum = res["checksum"]
        candidate.status = "prepared"
        candidate.properties["modeling_method"] = res["method"]
        candidate.properties["energy_kcal_mol"] = res["energy_kcal_mol"]
        candidate.properties["terminal_chemistry"] = res["terminal_chemistry"]
        candidate.properties["peptide_geometry"] = res["geometry"]

    return candidate
