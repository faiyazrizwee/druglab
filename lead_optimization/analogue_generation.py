"""
DrugLab Analogue Generation Module.

Generates structural analogues of a lead molecule using:
    - Bioisosteric replacement
    - Scaffold hopping (BRICS decomposition + recombination)
    - R-group enumeration

All strategies use RDKit and require no external services.
"""

import os
import time
import logging
import random
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    BRICS,
    Descriptors,
    rdMolDescriptors,
    rdFingerprintGenerator,
    DataStructs,
    rdmolops,
)
from rdkit.Chem.Scaffolds import MurckoScaffold

from utils.logger import setup_logger, log_step
from utils.io import ensure_dir, write_json, read_sdf, write_sdf, validate_file_exists
from utils.chemistry import (
    get_mol_name,
    set_mol_name,
    sanitize_mol,
    compute_tanimoto_similarity,
    calculate_basic_properties,
    deduplicate_mols,
)

logger = setup_logger("analogue_generation", level="INFO")

# ──────────────────────────────────────────────────────────────
# Common bioisosteric replacement SMARTS pairs
# ──────────────────────────────────────────────────────────────
BIOISOSTERE_PAIRS = [
    # (pattern, replacement, description)
    ("[CX3](=O)[OX2H1]", "[CX3](=O)[NH2]", "COOH → CONH2"),
    ("[CX3](=O)[OX2H1]", "[SX2](=O)(=O)[OH]", "COOH → SO3H"),
    ("[CX3](=O)[OX2H1]", "C(=O)O", "COOH → tetrazole placeholder"),
    ("[OX2H]", "[SX2H]", "OH → SH"),
    ("[OX2H]", "[NH2]", "OH → NH2"),
    ("[F]", "[Cl]", "F → Cl"),
    ("[Cl]", "[F]", "Cl → F"),
    ("[Cl]", "[CF3]", "Cl → CF3"),
    ("[c]1[c][c][c][c][c]1", "[c]1[c][n][c][c][c]1", "phenyl → pyridine"),
    ("[NH]C(=O)", "[NH]S(=O)(=O)", "amide → sulfonamide"),
    ("[CH3]", "[CH2F]", "CH3 → CH2F"),
    ("[CH3]", "[CF3]", "CH3 → CF3"),
]


def generate_analogues(
    lead_sdf: str,
    output_dir: str,
    strategies: list[str] | None = None,
    max_analogues: int = 100,
    similarity_range: tuple[float, float] = (0.4, 0.9),
    mw_range: tuple[float, float] = (150.0, 600.0),
    custom_fragments: str | None = None,
) -> dict:
    """
    Generate structural analogues of a lead molecule.

    Args:
        lead_sdf: Path to the lead molecule SDF file.
        output_dir: Path to the output directory.
        strategies: List of strategies to use: "bioisostere", "scaffold_hop", "r_group".
        max_analogues: Maximum total analogues to generate.
        similarity_range: (min, max) Tanimoto similarity to lead.
        mw_range: (min, max) MW filter for generated analogues.
        custom_fragments: Optional path to a file of custom SMILES fragments.

    Returns:
        Dict containing generation results and output file paths.
    """
    start_time = time.time()
    log_step(logger, "Analogue Generation", "started", {
        "lead": lead_sdf, "strategies": strategies,
    })

    validate_file_exists(lead_sdf, "Lead molecule SDF")

    if strategies is None:
        strategies = ["bioisostere", "scaffold_hop", "r_group"]

    # Output paths
    gen_dir = os.path.join(output_dir, "analogue_generation")
    ensure_dir(gen_dir)

    analogues_sdf_path = os.path.join(gen_dir, "generated_analogues.sdf")
    report_path = os.path.join(gen_dir, "analogue_report.json")

    # Load lead molecule
    lead_mols = read_sdf(lead_sdf)
    if not lead_mols:
        raise ValueError(f"No valid molecules found in lead SDF: {lead_sdf}")
    lead_mol = lead_mols[0]
    lead_name = get_mol_name(lead_mol, 0)
    logger.info(f"Lead molecule: {lead_name} ({Chem.MolToSmiles(lead_mol)})")

    # Remove explicit H for processing
    lead_mol_noH = Chem.RemoveHs(lead_mol)

    all_analogues = []
    strategy_counts = {}

    # Run each strategy
    per_strategy_limit = max(max_analogues // len(strategies), 10)

    if "bioisostere" in strategies:
        bio_mols = _bioisosteric_replacement(lead_mol_noH, per_strategy_limit)
        strategy_counts["bioisostere"] = len(bio_mols)
        all_analogues.extend(bio_mols)
        logger.info(f"Bioisosteric replacement: {len(bio_mols)} analogues")

    if "scaffold_hop" in strategies:
        hop_mols = _scaffold_hop(lead_mol_noH, per_strategy_limit)
        strategy_counts["scaffold_hop"] = len(hop_mols)
        all_analogues.extend(hop_mols)
        logger.info(f"Scaffold hopping: {len(hop_mols)} analogues")

    if "r_group" in strategies:
        rg_mols = _r_group_enumeration(lead_mol_noH, per_strategy_limit, custom_fragments)
        strategy_counts["r_group"] = len(rg_mols)
        all_analogues.extend(rg_mols)
        logger.info(f"R-group enumeration: {len(rg_mols)} analogues")

    # Deduplicate
    all_analogues = deduplicate_mols(all_analogues)
    logger.info(f"After deduplication: {len(all_analogues)} unique analogues")

    # Filter by similarity and MW
    filtered = []
    for mol in all_analogues:
        if mol is None:
            continue
        try:
            sim = compute_tanimoto_similarity(mol, lead_mol_noH)
            mw = Descriptors.MolWt(mol)
            if similarity_range[0] <= sim <= similarity_range[1] and \
               mw_range[0] <= mw <= mw_range[1]:
                mol.SetProp("similarity_to_lead", f"{sim:.3f}")
                mol.SetProp("MW", f"{mw:.2f}")
                filtered.append(mol)
        except Exception:
            pass

    logger.info(f"After similarity/MW filter: {len(filtered)} analogues")

    # Cap at max_analogues
    if len(filtered) > max_analogues:
        filtered = filtered[:max_analogues]

    # Add 3D coordinates and properties
    final_mols = []
    for idx, mol in enumerate(filtered):
        name = f"analogue_{idx+1}"
        set_mol_name(mol, name)

        # Add Hs and embed 3D
        mol_h = Chem.AddHs(mol)
        try:
            AllChem.EmbedMolecule(mol_h, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol_h)
        except Exception:
            pass

        # Calculate properties
        props = calculate_basic_properties(mol)
        for pk, pv in props.items():
            mol_h.SetProp(pk, str(pv))

        # Carry over similarity
        if mol.HasProp("similarity_to_lead"):
            mol_h.SetProp("similarity_to_lead", mol.GetProp("similarity_to_lead"))

        final_mols.append(mol_h)

    # Write outputs
    if final_mols:
        write_sdf(final_mols, analogues_sdf_path)

    duration = time.time() - start_time
    metrics = {
        "lead_name": lead_name,
        "lead_smiles": Chem.MolToSmiles(lead_mol_noH),
        "strategies_used": strategies,
        "strategy_counts": strategy_counts,
        "total_before_filter": len(all_analogues),
        "total_generated": len(final_mols),
        "analogues_sdf": analogues_sdf_path,
        "duration": round(duration, 2),
        "status": "completed" if final_mols else "no_analogues",
    }

    write_json(metrics, report_path)
    log_step(logger, "Analogue Generation", "completed", {
        "analogues": len(final_mols),
        "strategies": strategies,
        "duration": f"{duration:.1f}s",
    })

    return metrics


# ──────────────────────────────────────────────────────────────
# Strategy 1: Bioisosteric Replacement
# ──────────────────────────────────────────────────────────────

def _bioisosteric_replacement(lead_mol, max_count: int) -> list:
    """Generate analogues by replacing functional groups with bioisosteres."""
    analogues = []
    lead_smiles = Chem.MolToSmiles(lead_mol)

    for pattern_sma, replacement_sma, desc in BIOISOSTERE_PAIRS:
        try:
            pattern = Chem.MolFromSmarts(pattern_sma)
            if pattern is None:
                continue
            if not lead_mol.HasSubstructMatch(pattern):
                continue

            replacement = Chem.MolFromSmiles(replacement_sma)
            if replacement is None:
                replacement = Chem.MolFromSmarts(replacement_sma)
            if replacement is None:
                continue

            products = AllChem.ReplaceSubstructs(lead_mol, pattern, replacement)
            for prod in products:
                prod = sanitize_mol(prod)
                if prod is not None:
                    prod_smiles = Chem.MolToSmiles(prod)
                    if prod_smiles != lead_smiles:
                        set_mol_name(prod, f"bio_{desc.replace(' ', '_')}")
                        analogues.append(prod)

        except Exception as e:
            logger.debug(f"Bioisostere {desc} failed: {e}")

        if len(analogues) >= max_count:
            break

    return analogues[:max_count]


# ──────────────────────────────────────────────────────────────
# Strategy 2: Scaffold Hopping (BRICS)
# ──────────────────────────────────────────────────────────────

def _scaffold_hop(lead_mol, max_count: int) -> list:
    """
    Generate analogues via BRICS decomposition and recombination.

    Breaks the lead into fragments, then recombines them in new ways.
    """
    analogues = []
    lead_smiles = Chem.MolToSmiles(lead_mol)

    try:
        # Decompose lead into BRICS fragments
        frags = list(BRICS.BRICSDecompose(lead_mol))
        if len(frags) < 2:
            logger.debug("Lead has fewer than 2 BRICS fragments; skipping scaffold hop.")
            return []

        logger.debug(f"BRICS fragments: {len(frags)}")

        # Convert fragment SMILES to Mol objects
        frag_mols = []
        for fs in frags:
            fm = Chem.MolFromSmiles(fs)
            if fm is not None:
                frag_mols.append(fm)

        if len(frag_mols) < 2:
            return []

        # Build new molecules from fragment combinations
        # Use BRICS build — limit to avoid combinatorial explosion
        built = set()
        try:
            for product in BRICS.BRICSBuild(frag_mols):
                product = sanitize_mol(product)
                if product is None:
                    continue
                smi = Chem.MolToSmiles(product)
                if smi == lead_smiles or smi in built:
                    continue
                built.add(smi)
                set_mol_name(product, f"scaffold_hop_{len(analogues)}")
                analogues.append(product)
                if len(analogues) >= max_count:
                    break
        except Exception as e:
            logger.debug(f"BRICS build error: {e}")

    except Exception as e:
        logger.debug(f"Scaffold hop failed: {e}")

    return analogues[:max_count]


# ──────────────────────────────────────────────────────────────
# Strategy 3: R-Group Enumeration
# ──────────────────────────────────────────────────────────────

# Common small substituent fragments for R-group decoration
DEFAULT_R_GROUPS = [
    "C",       # methyl
    "CC",      # ethyl
    "C(C)C",   # isopropyl
    "C1CC1",   # cyclopropyl
    "F",       # fluorine
    "Cl",      # chlorine
    "O",       # hydroxyl
    "OC",      # methoxy
    "N",       # amino
    "NC",      # methylamino
    "C#N",     # nitrile
    "C(F)(F)F",  # trifluoromethyl
    "S(=O)(=O)N", # sulfonamide
    "C(=O)O",     # carboxyl
    "C(=O)N",     # amide
]


def _r_group_enumeration(lead_mol, max_count: int, custom_fragments: str | None = None) -> list:
    """
    Generate analogues by enumerating R-group substitutions on the lead scaffold.

    Identifies hydrogen attachment points on the Murcko scaffold and
    substitutes them with small functional groups.
    """
    analogues = []
    lead_smiles = Chem.MolToSmiles(lead_mol)

    # Load custom fragments if provided
    r_groups = list(DEFAULT_R_GROUPS)
    if custom_fragments and os.path.isfile(custom_fragments):
        try:
            with open(custom_fragments, "r") as f:
                for line in f:
                    smi = line.strip()
                    if smi and not smi.startswith("#"):
                        r_groups.append(smi)
        except Exception as e:
            logger.warning(f"Failed to load custom fragments: {e}")

    # Get scaffold
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(lead_mol)
    except Exception:
        scaffold = lead_mol

    # Find attachment points: atoms with implicit Hs
    attachment_atoms = []
    for atom in lead_mol.GetAtoms():
        if atom.GetTotalNumHs() > 0 and atom.GetAtomicNum() in (6, 7):  # C or N
            attachment_atoms.append(atom.GetIdx())

    if not attachment_atoms:
        logger.debug("No suitable attachment points found for R-group enumeration.")
        return []

    # Limit attachment points to avoid explosion
    if len(attachment_atoms) > 10:
        attachment_atoms = random.sample(attachment_atoms, 10)

    seen = {lead_smiles}

    for atom_idx in attachment_atoms:
        for r_smi in r_groups:
            try:
                r_mol = Chem.MolFromSmiles(r_smi)
                if r_mol is None:
                    continue

                # Create editable molecule
                combo = Chem.RWMol(Chem.CombineMols(lead_mol, r_mol))

                # Find the first atom of the R-group in the combined molecule
                r_start = lead_mol.GetNumAtoms()  # First atom of r_mol in combo

                # Add bond between attachment atom and R-group
                combo.AddBond(atom_idx, r_start, Chem.BondType.SINGLE)

                # Sanitize
                try:
                    Chem.SanitizeMol(combo)
                except Exception:
                    continue

                smi = Chem.MolToSmiles(combo)
                if smi in seen:
                    continue
                seen.add(smi)

                product = Chem.MolFromSmiles(smi)
                if product is not None:
                    set_mol_name(product, f"rgroup_{len(analogues)}")
                    analogues.append(product)

            except Exception:
                pass

            if len(analogues) >= max_count:
                return analogues[:max_count]

    return analogues[:max_count]
