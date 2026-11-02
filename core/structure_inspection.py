"""
DrugLab Unified Structure Component Inspector.

Inspects raw and prepared protein structures (both experimental PDB and predicted
AlphaFold models) to identify components that may be removed or altered during
docking preparation:
- Crystallographic water molecules (bulk vs. interfacial)
- Non-protein ligands, substrates, and crystallization additives
- Catalytic and structural metal ions
- Biological cofactors and coenzymes
- Multiple / alternate protein chains
- Alternate conformations (altlocs)
- Non-standard / modified amino acid residues
- AlphaFold confidence distribution (pLDDT) and predictive caveats

Provides workflow-aware preparation proposals with scientific rationales,
docking consequences, and engine-specific safety constraints for both
Small-Molecule (AutoDock Vina) and Peptide (LightDock) docking.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import gemmi
from utils.logger import setup_logger

logger = setup_logger("structure_inspection", level="INFO")

# Metal ion residues
KNOWN_METAL_IONS = {
    "ZN", "MG", "CA", "FE", "MN", "CU", "NA", "K", "CO", "NI",
    "CD", "PT", "HG", "AU", "FE2", "ZN2", "MG2", "CA2", "MO", "W", "V",
}

# Known biological cofactors and coenzymes
KNOWN_COFACTORS = {
    "ATP", "ADP", "AMP", "GTP", "GDP", "NAD", "NAP", "NADH", "NADPH",
    "FAD", "FMN", "HEM", "HEA", "HEB", "HEC", "SAM", "SAH", "PLP",
    "PMP", "COA", "ACO", "B12", "THP", "F420", "CAM", "TPP", "GSH",
}

# Standard amino acid codes
STANDARD_AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}

# Standard nucleic acid codes
STANDARD_NUCLEIC_ACIDS = {
    "DA", "DC", "DG", "DT", "A", "C", "G", "U", "I",
}

# Selenomethionine and common non-standard residue mappings
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

# Common crystallographic buffer / precipitant / additive codes
COMMON_CRYSTAL_ADDITIVES = {
    "PEG", "PG4", "P6G", "1PE", "2PE", "EDO", "GOL", "DMS", "SO4", "PO4",
    "ACT", "CIT", "FMT", "TRS", "MES", "HEPES", "EPE", "BOG", "BMA", "NAG",
    "MPD", "IPA", "EOH", "MOH", "AZI", "CL", "BR", "IOD", "SCN",
}


def is_alphafold_model(file_path: str | Path, gemmi_structure: gemmi.Structure | None = None) -> bool:
    """
    Detect whether a PDB file originates from AlphaFold / AlphaFold DB.
    """
    path = Path(file_path)
    name_lower = path.name.lower()
    if name_lower.startswith("af-") or name_lower.startswith("af_") or "alphafold" in name_lower:
        return True

    if not path.is_file():
        return False

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(60):
                line = f.readline()
                if not line:
                    break
                line_u = line.upper()
                if "ALPHAFOLD" in line_u or "DEEPMIND" in line_u:
                    return True
                if line.startswith("TITLE") and "MODEL OF" in line_u:
                    return True
    except Exception:
        pass

    # Inspect B-factor column in coordinate lines:
    # AlphaFold models populate B-factor with pLDDT values between 0.00 and 100.00
    try:
        st = gemmi_structure or gemmi.read_structure(str(path))
        if len(st) > 0:
            b_factors = [a.b_iso for a in st[0] if a.b_iso is not None]
            if len(b_factors) > 50:
                min_b = min(b_factors)
                max_b = max(b_factors)
                if 0.0 <= min_b and max_b <= 100.0:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        header_chunk = f.read(2048).upper()
                        if "ALPHAFOLD" in header_chunk or "EMBL-EBI" in header_chunk:
                            return True
    except Exception:
        pass

    return False


def calculate_plddt_stats(file_path: str | Path, gemmi_structure: gemmi.Structure | None = None) -> dict[str, Any]:
    """
    Compute per-residue pLDDT confidence statistics from an AlphaFold PDB structure.
    """
    path = Path(file_path)
    st = gemmi_structure or gemmi.read_structure(str(path))
    if len(st) == 0:
        return {"has_plddt": False}

    model = st[0]
    res_plddt: list[float] = []

    for chain in model:
        for res in chain:
            rname = res.name.strip().upper()
            if res.entity_type == gemmi.EntityType.Polymer or rname in STANDARD_AMINO_ACIDS:
                # Residue pLDDT is represented in CA atom B-factor, or average across backbone
                ca = res.find_atom("CA", "*")
                if ca is not None:
                    res_plddt.append(ca.b_iso)
                elif len(res) > 0:
                    res_plddt.append(sum(a.b_iso for a in res) / len(res))

    if not res_plddt:
        return {"has_plddt": False}

    total = len(res_plddt)
    mean_val = sum(res_plddt) / total
    vh = sum(1 for v in res_plddt if v >= 90.0)
    c = sum(1 for v in res_plddt if 70.0 <= v < 90.0)
    l = sum(1 for v in res_plddt if 50.0 <= v < 70.0)
    vl = sum(1 for v in res_plddt if v < 50.0)

    return {
        "has_plddt": True,
        "mean_plddt": round(mean_val, 1),
        "total_evaluated_residues": total,
        "very_high_count": vh,
        "very_high_pct": round((vh / total) * 100, 1),
        "confident_count": c,
        "confident_pct": round((c / total) * 100, 1),
        "low_count": l,
        "low_pct": round((l / total) * 100, 1),
        "very_low_count": vl,
        "very_low_pct": round((vl / total) * 100, 1),
        "has_disordered_regions": (l + vl) > 0,
        "disordered_pct": round(((l + vl) / total) * 100, 1),
    }


def inspect_structure_components(
    input_pdb: str | Path,
    workflow_mode: Literal["small_molecule", "peptide"] = "small_molecule",
    alphafold_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Comprehensively inspect all structural components of a protein PDB file
    and formulate transparent preparation proposals with scientific rationales,
    docking consequences, and engine-specific safety constraints.

    Args:
        input_pdb: Path to the target PDB file.
        workflow_mode: Either 'small_molecule' (AutoDock Vina) or 'peptide' (LightDock).
        alphafold_metadata: Optional dictionary of AlphaFold DB API metadata.

    Returns:
        Structured inspection result containing metadata, detected components,
        AlphaFold caveats, and proposed preparation review items.
    """
    path = Path(input_pdb).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Protein file not found: {input_pdb}")

    structure = gemmi.read_structure(str(path))
    if len(structure) == 0:
        raise ValueError(f"No 3D structural models found in PDB file: {path.name}")

    model = structure[0]
    is_af = is_alphafold_model(path, structure) or (alphafold_metadata is not None and alphafold_metadata.get("is_predicted", False))
    plddt_stats = calculate_plddt_stats(path, structure) if is_af else {"has_plddt": False}

    # 1. Inspect Chains
    chains_list = []
    protein_atom_coords: list[tuple[float, float, float]] = []

    for chain in model:
        polymer_residues = [r for r in chain if r.entity_type == gemmi.EntityType.Polymer]
        total_atoms = sum(len(r) for r in chain)
        seq_sample = gemmi.one_letter_code([r.name for r in polymer_residues[:10]]) if polymer_residues else ""
        if len(polymer_residues) > 10:
            seq_sample += f"... ({len(polymer_residues)} AA)"

        for res in polymer_residues:
            if res.name in STANDARD_AMINO_ACIDS:
                for a in res:
                    protein_atom_coords.append((a.pos.x, a.pos.y, a.pos.z))

        chains_list.append({
            "chain_id": chain.name,
            "num_residues": len(polymer_residues),
            "num_atoms": total_atoms,
            "sequence_preview": seq_sample,
            "is_polymer": len(polymer_residues) > 0,
        })

    all_chain_ids = [c["chain_id"] for c in chains_list]
    primary_chain_id = all_chain_ids[0] if all_chain_ids else "A"
    has_multiple_chains = len(all_chain_ids) > 1

    # 2. Inspect Water Molecules (Classify Interfacial vs Bulk)
    def is_interfacial_water(wx: float, wy: float, wz: float, cutoff: float = 4.5) -> bool:
        if not protein_atom_coords:
            return False
        cutoff_sq = cutoff * cutoff
        # Test against nearby protein atoms
        for px, py, pz in protein_atom_coords:
            dx = wx - px
            dy = wy - py
            dz = wz - pz
            if (dx * dx + dy * dy + dz * dz) <= cutoff_sq:
                return True
        return False

    waters_all: list[str] = []
    interfacial_waters: list[str] = []
    bulk_waters: list[str] = []

    for chain in model:
        for res in chain:
            rname = res.name.strip().upper()
            if res.entity_type == gemmi.EntityType.Water or rname in {"HOH", "WAT", "H2O", "TIP3", "DOD"}:
                res_id = f"{chain.name}:{rname}:{res.seqid}"
                waters_all.append(res_id)
                w_pos = res[0].pos if len(res) > 0 else None
                if w_pos and is_interfacial_water(w_pos.x, w_pos.y, w_pos.z, cutoff=4.5):
                    interfacial_waters.append(res_id)
                else:
                    bulk_waters.append(res_id)

    # 3. Inspect Non-Protein Ligands, Metals, Cofactors
    metals_detected: list[dict[str, Any]] = []
    cofactors_detected: list[dict[str, Any]] = []
    ligands_detected: list[dict[str, Any]] = []
    crystal_additives_detected: list[dict[str, Any]] = []

    for chain in model:
        for res in chain:
            rname = res.name.strip().upper()
            if res.entity_type == gemmi.EntityType.Water or rname in {"HOH", "WAT", "H2O", "TIP3", "DOD"}:
                continue
            if res.entity_type == gemmi.EntityType.Polymer and rname in STANDARD_AMINO_ACIDS:
                continue

            item_info = {
                "chain_id": chain.name,
                "residue_name": rname,
                "seqid": str(res.seqid),
                "atom_count": len(res),
                "identifier": f"{chain.name}:{rname}:{res.seqid}",
            }

            if rname in KNOWN_METAL_IONS:
                metals_detected.append(item_info)
            elif rname in KNOWN_COFACTORS:
                cofactors_detected.append(item_info)
            elif rname in COMMON_CRYSTAL_ADDITIVES:
                crystal_additives_detected.append(item_info)
            elif rname not in NONSTANDARD_AA_MAPPINGS:
                # Bound small molecule ligand or inhibitor
                ligands_detected.append(item_info)

    # 4. Inspect Alternate Conformations (Altlocs)
    altlocs_detected: list[dict[str, Any]] = []
    altloc_res_map: dict[str, list[str]] = {}

    for chain in model:
        for res in chain:
            for atom in res:
                if atom.has_altloc():
                    r_key = f"{chain.name}:{res.name}:{res.seqid}"
                    if r_key not in altloc_res_map:
                        altloc_res_map[r_key] = []
                    if atom.altloc not in altloc_res_map[r_key]:
                        altloc_res_map[r_key].append(atom.altloc)

    for r_key, confs in altloc_res_map.items():
        altlocs_detected.append({
            "residue": r_key,
            "conformations": sorted(confs),
            "identifier": r_key,
        })

    # 5. Inspect Non-standard / Modified Amino Acids
    nonstandard_detected: list[dict[str, Any]] = []
    for chain in model:
        for res in chain:
            rname = res.name.strip().upper()
            if res.entity_type == gemmi.EntityType.Polymer and rname not in STANDARD_AMINO_ACIDS:
                mapped = NONSTANDARD_AA_MAPPINGS.get(rname, "None")
                nonstandard_detected.append({
                    "chain_id": chain.name,
                    "residue_name": rname,
                    "seqid": str(res.seqid),
                    "mapped_to": mapped,
                    "identifier": f"{chain.name}:{rname}:{res.seqid}",
                })

    # 6. Formulate Transparent Preparation Proposals
    review_items: list[dict[str, Any]] = []

    # (A) Water molecules proposal
    if waters_all:
        if workflow_mode == "small_molecule":
            prop_action = "remove"
            rationale = (
                f"AutoDock Vina scoring treats solvent implicitly. Retaining {len(waters_all)} crystallographic waters "
                f"artificially occludes binding cavities, blocks ligand entry, and distorts grid affinity potentials."
            )
            consequence = (
                "Removing waters frees pocket volume for comprehensive ligand exploration. "
                "Retaining waters without specialized water-docking models will cause severe steric clashes."
            )
            is_safe = True
            warning = "Keeping waters may cause steric collisions with small-molecule poses."
        else:  # peptide
            prop_action = "remove_bulk"  # keep interfacial if requested
            rationale = (
                f"LightDock AMBER94/DFIRE force fields do not support explicit bulk solvent waters ({len(bulk_waters)} detected). "
                f"However, interfacial waters ({len(interfacial_waters)} detected) may stabilize protein-peptide hydrogen-bond networks."
            )
            consequence = (
                "Removing bulk waters prevents scoring failures. "
                "Interfacial waters can optionally be retained if supported by the chosen scoring function."
            )
            is_safe = True
            warning = "Bulk waters will cause LightDock scoring crashes and must be removed."

        review_items.append({
            "id": "waters",
            "title": f"Crystallographic Waters ({len(waters_all)} total)",
            "component_type": "water",
            "count": len(waters_all),
            "details": {
                "total": len(waters_all),
                "interfacial": len(interfacial_waters),
                "bulk": len(bulk_waters),
            },
            "proposed_action": prop_action,
            "allowed_actions": ["remove", "remove_bulk", "keep"] if workflow_mode == "peptide" else ["remove", "keep"],
            "is_safe_to_override": workflow_mode == "small_molecule",
            "safety_warning": warning,
            "scientific_rationale": rationale,
            "docking_consequences": consequence,
        })

    # (B) Non-protein ligands proposal
    if ligands_detected:
        lig_names = sorted(list(set(l["residue_name"] for l in ligands_detected)))
        if workflow_mode == "small_molecule":
            prop_action = "remove"
            rationale = (
                f"Bound co-crystallized compounds ({', '.join(lig_names)}) occupy the native binding site. "
                f"Unless removed, they directly occlude the pocket and prevent virtual screening candidates from entering."
            )
            consequence = (
                "Removing bound ligands clears the active site for docking. "
                "Retaining a bound ligand is only appropriate when defining a reference ligand for pocket definition."
            )
            is_safe = False
            warning = "Retaining bound ligands will block the binding pocket for newly docked molecules."
        else:  # peptide
            prop_action = "remove"
            rationale = (
                f"LightDock AMBER94 / DFIRE force fields lack atom parameters for synthetic hetero-compounds ({', '.join(lig_names)}). "
                f"Leaving them in the receptor causes LightDock simulation failure (NotSupportedInScoringError)."
            )
            consequence = (
                "Removing them produces a clean, valid protein receptor. "
                "Retaining them will cause immediate simulation failure during docking setup."
            )
            is_safe = False
            warning = "LightDock scoring engines do not support synthetic small-molecule heteroatoms."

        review_items.append({
            "id": "ligands",
            "title": f"Bound Ligands / Hetero Compounds ({len(ligands_detected)} items: {', '.join(lig_names)})",
            "component_type": "ligand",
            "count": len(ligands_detected),
            "items": ligands_detected,
            "proposed_action": prop_action,
            "allowed_actions": ["remove", "keep"],
            "is_safe_to_override": False,
            "safety_warning": warning,
            "scientific_rationale": rationale,
            "docking_consequences": consequence,
        })

    # (C) Metal ions proposal
    if metals_detected:
        metal_names = sorted(list(set(m["residue_name"] for m in metals_detected)))
        if workflow_mode == "small_molecule":
            prop_action = "keep"
            rationale = (
                f"Catalytic/structural metal ions ({', '.join(metal_names)}) often coordinate active-site residues "
                f"and form critical electrostatic coordination bonds with candidate ligands (e.g. Zn²⁺ in metalloproteases, Mg²⁺ in kinases)."
            )
            consequence = (
                "Keeping metal ions maintains authentic active site architecture. "
                "Removing them will eliminate essential coordination contacts and distort docking scoring."
            )
            is_safe = True
            warning = "If kept, ensure the metal is parameterized with correct partial charges."
        else:  # peptide
            prop_action = "remove"  # LightDock AMBER94 crashes on MG/ZN unless sanitized
            rationale = (
                f"Metal ions ({', '.join(metal_names)}) participate in coordination complexes, but standard "
                f"LightDock peptide scoring functions (DFIRE/CPyDock) lack ion parameter sets and raise NotSupportedInScoringError."
            )
            consequence = (
                "LightDock simulation requires receptor coordinates without unsupported metal ions. "
                "DrugLab automatically sanitizes simulation coordinates while preserving structural reference."
            )
            is_safe = False
            warning = "LightDock scoring force fields crash on unparameterized metal ions."

        review_items.append({
            "id": "metals",
            "title": f"Metal Ions ({len(metals_detected)} ions: {', '.join(metal_names)})",
            "component_type": "metal",
            "count": len(metals_detected),
            "items": metals_detected,
            "proposed_action": prop_action,
            "allowed_actions": ["keep", "remove"],
            "is_safe_to_override": workflow_mode == "small_molecule",
            "safety_warning": warning,
            "scientific_rationale": rationale,
            "docking_consequences": consequence,
        })

    # (D) Biological cofactors proposal
    if cofactors_detected:
        cof_names = sorted(list(set(c["residue_name"] for c in cofactors_detected)))
        if workflow_mode == "small_molecule":
            prop_action = "keep"
            rationale = (
                f"Biological cofactors ({', '.join(cof_names)}) (e.g. Heme, NAD, FAD) form integral parts of the "
                f"protein catalytic pocket and contribute essential steric boundaries and hydrogen-bonding motifs."
            )
            consequence = (
                "Retaining cofactors preserves realistic pocket shape. "
                "Removing them creates an unnaturally enlarged cavity that yields false-positive binding poses."
            )
            is_safe = True
            warning = "Retained cofactors will be converted as part of the rigid receptor."
        else:  # peptide
            prop_action = "remove"
            rationale = (
                f"Cofactors ({', '.join(cof_names)}) cannot be evaluated by standard protein-peptide potential functions in LightDock."
            )
            consequence = (
                "Removing them ensures force field compatibility during docking simulation."
            )
            is_safe = False
            warning = "Cofactors lack AMBER94/DFIRE force field parameters in LightDock."

        review_items.append({
            "id": "cofactors",
            "title": f"Biological Cofactors ({len(cofactors_detected)} items: {', '.join(cof_names)})",
            "component_type": "cofactor",
            "count": len(cofactors_detected),
            "items": cofactors_detected,
            "proposed_action": prop_action,
            "allowed_actions": ["keep", "remove"],
            "is_safe_to_override": workflow_mode == "small_molecule",
            "safety_warning": warning,
            "scientific_rationale": rationale,
            "docking_consequences": consequence,
        })

    # (E) Crystallization additives proposal
    if crystal_additives_detected:
        add_names = sorted(list(set(a["residue_name"] for a in crystal_additives_detected)))
        review_items.append({
            "id": "additives",
            "title": f"Crystallization Additives ({len(crystal_additives_detected)} items: {', '.join(add_names)})",
            "component_type": "ligand",
            "count": len(crystal_additives_detected),
            "items": crystal_additives_detected,
            "proposed_action": "remove",
            "allowed_actions": ["remove", "keep"],
            "is_safe_to_override": False,
            "safety_warning": "Additives are experimental artifacts from crystallization buffers and should not be docked against.",
            "scientific_rationale": (
                f"Cryoprotectants and buffer additives ({', '.join(add_names)}) (e.g. glycerol, PEG, sulfate) "
                f"are non-physiological artifacts of crystal growth."
            ) ,
            "docking_consequences": (
                "Removing them eliminates artificial pocket obstruction. "
                "Retaining them would result in unrealistic steric barriers."
            ),
        })

    # (F) Multiple / extra chains proposal
    if has_multiple_chains:
        review_items.append({
            "id": "chains",
            "title": f"Multiple Protein Chains ({len(all_chain_ids)} chains: {', '.join(all_chain_ids)})",
            "component_type": "extra_chain",
            "count": len(all_chain_ids),
            "chains": chains_list,
            "proposed_action": "keep_all",  # or select primary chain
            "allowed_actions": ["keep_all", "select_chain"],
            "is_safe_to_override": True,
            "safety_warning": "Select a single chain if docking is targeted to an isolated monomer, or keep all chains for multimeric complexes.",
            "scientific_rationale": (
                f"The structure contains {len(all_chain_ids)} polymer chains ({', '.join(all_chain_ids)}). "
                f"If the biological receptor functions as an oligomer or complex, retaining all chains preserves the interface."
            ),
            "docking_consequences": (
                "Keeping all chains allows docking at inter-chain interfaces. "
                "Selecting a single chain isolates the monomer and reduces computational search space."
            ),
        })

    # (G) Alternate conformations (altlocs) proposal
    if altlocs_detected:
        review_items.append({
            "id": "altlocs",
            "title": f"Alternate Conformations ({len(altlocs_detected)} residues with altlocs)",
            "component_type": "altloc",
            "count": len(altlocs_detected),
            "items": altlocs_detected[:25],
            "proposed_action": "modify",  # resolve to highest occupancy
            "allowed_actions": ["modify", "keep"],
            "is_safe_to_override": False,
            "safety_warning": "Docking engines cannot accept duplicate atom coordinates with split occupancies.",
            "scientific_rationale": (
                f"Detected {len(altlocs_detected)} residues with alternate crystallographic conformations. "
                f"Docking grid calculations require a single unambiguous 3D coordinate for every atom."
            ),
            "docking_consequences": (
                "Resolving to the highest-occupancy conformation (conformer 'A') produces physically valid geometry. "
                "Failing to resolve altlocs causes topology parsing errors and overlapping atom collisions."
            ),
        })

    # (H) Non-standard amino acids proposal
    if nonstandard_detected:
        ns_names = sorted(list(set(n["residue_name"] for n in nonstandard_detected)))
        review_items.append({
            "id": "nonstandard",
            "title": f"Non-standard / Modified Residues ({len(nonstandard_detected)} residues: {', '.join(ns_names)})",
            "component_type": "nonstandard",
            "count": len(nonstandard_detected),
            "items": nonstandard_detected,
            "proposed_action": "modify",  # standardize
            "allowed_actions": ["modify", "keep", "remove"],
            "is_safe_to_override": True,
            "safety_warning": "Non-standard residues may lack force field atom types unless mapped to canonical equivalents.",
            "scientific_rationale": (
                f"Non-standard residues (e.g. MSE, PTR, SEP) can be standardized to canonical amino acids "
                f"(e.g. MSE -> MET) to ensure compatibility with standard docking parameter sets."
            ),
            "docking_consequences": (
                "Standardizing preserves backbone connectivity and assigns correct atomic partial charges. "
                "Deleting them creates unclosed chain breaks."
            ),
        })

    # (I) AlphaFold prediction safeguards & caveats
    alphafold_caveats = None
    if is_af:
        mean_p = plddt_stats.get("mean_plddt", 0.0)
        dis_pct = plddt_stats.get("disordered_pct", 0.0)
        alphafold_caveats = {
            "is_alphafold": True,
            "mean_plddt": mean_p,
            "disordered_pct": dis_pct,
            "plddt_stats": plddt_stats,
            "disclaimer": (
                "This receptor is an in silico predicted structure from the AlphaFold Database (EMBL-EBI / DeepMind). "
                "AlphaFold predictions do NOT contain co-crystallized waters, metal cofactors, or ligands. "
                "Binding pocket conformations represent an unliganded (apo) state and may require induced-fit flexibility. "
                + (f"Warning: {dis_pct}% of residues have low confidence (pLDDT < 70) and represent disordered regions." if dis_pct > 15.0 else "High structural confidence across core domains.")
            ),
            "docking_suitability": (
                "Core globular domains with pLDDT > 70 are generally suitable for rigid receptor docking. "
                "Residues with pLDDT < 50 should be excluded from binding site restraints or interpreted with extreme caution."
            ),
        }

    return {
        "file_path": str(path),
        "file_name": path.name,
        "is_alphafold": is_af,
        "workflow_mode": workflow_mode,
        "structure_source": "AlphaFold DB (Predicted)" if is_af else "Experimental PDB (X-ray / Cryo-EM / NMR)",
        "alphafold_metadata": alphafold_metadata or {},
        "alphafold_caveats": alphafold_caveats,
        "chains": chains_list,
        "chain_ids": all_chain_ids,
        "primary_chain": primary_chain_id,
        "total_residues": sum(c["num_residues"] for c in chains_list),
        "total_atoms": sum(c["num_atoms"] for c in chains_list),
        "waters_count": len(waters_all),
        "interfacial_waters_count": len(interfacial_waters),
        "bulk_waters_count": len(bulk_waters),
        "metals_count": len(metals_detected),
        "metals": [m["residue_name"] for m in metals_detected],
        "cofactors_count": len(cofactors_detected),
        "cofactors": [c["residue_name"] for c in cofactors_detected],
        "ligands_count": len(ligands_detected),
        "ligands": [l["residue_name"] for l in ligands_detected],
        "crystal_additives_count": len(crystal_additives_detected),
        "altlocs_count": len(altlocs_detected),
        "nonstandard_count": len(nonstandard_detected),
        "review_items": review_items,
    }
