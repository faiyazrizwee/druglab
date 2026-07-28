"""
DrugLab Shared Chemistry Utilities.

Common cheminformatics operations used across multiple modules:
SMILES/Mol conversions, sanitization, fingerprints, and basic property calculation.

Usage:
    from utils.chemistry import smiles_to_mol, calculate_basic_properties
    mol = smiles_to_mol("c1ccccc1")
    props = calculate_basic_properties(mol)
"""

from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Descriptors,
    rdMolDescriptors,
    DataStructs,
    rdFingerprintGenerator,
)

from utils.logger import setup_logger

logger = setup_logger("chemistry", level="INFO")


# ──────────────────────────────────────────────────────────────
# SMILES ↔ Mol Conversions
# ──────────────────────────────────────────────────────────────

def smiles_to_mol(smiles: str) -> Chem.Mol:
    """
    Convert a SMILES string to an RDKit Mol object.

    Args:
        smiles: SMILES string.

    Returns:
        RDKit Mol object.

    Raises:
        ValueError: If SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: '{smiles}'")
    return mol


def mol_to_smiles(mol: Chem.Mol, canonical: bool = True) -> str:
    """
    Convert an RDKit Mol to a canonical SMILES string.

    Args:
        mol: RDKit Mol object.
        canonical: If True, return canonical SMILES.

    Returns:
        SMILES string.
    """
    return Chem.MolToSmiles(mol, canonical=canonical)


def mol_to_smarts(mol: Chem.Mol) -> str:
    """
    Convert an RDKit Mol to a SMARTS pattern.

    Args:
        mol: RDKit Mol object.

    Returns:
        SMARTS string.
    """
    return Chem.MolToSmarts(mol)


# ──────────────────────────────────────────────────────────────
# Molecule Sanitization & Processing
# ──────────────────────────────────────────────────────────────

def sanitize_mol(mol: Chem.Mol) -> Chem.Mol | None:
    """
    Sanitize a molecule: kekulize, add hydrogens, compute 2D coords if missing.

    Args:
        mol: RDKit Mol object.

    Returns:
        Sanitized Mol object, or None if sanitization fails.
    """
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception as e:
        logger.warning(f"Failed to sanitize molecule: {e}")
        return None


def add_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    """
    Add explicit hydrogens to a molecule.

    Args:
        mol: RDKit Mol object.

    Returns:
        Mol with explicit hydrogens.
    """
    return Chem.AddHs(mol)


def remove_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    """
    Remove explicit hydrogens from a molecule.

    Args:
        mol: RDKit Mol object.

    Returns:
        Mol without explicit hydrogens.
    """
    return Chem.RemoveHs(mol)


def remove_salts(mol: Chem.Mol) -> Chem.Mol:
    """
    Remove salt fragments from a molecule, keeping the largest fragment.

    Args:
        mol: RDKit Mol object (may be a salt / multi-fragment).

    Returns:
        Largest fragment as Mol.
    """
    from rdkit.Chem import SaltRemover

    remover = SaltRemover.SaltRemover()
    stripped = remover.StripMol(mol)

    # Always ensure we keep only the largest fragment if multiple fragments exist
    frags = Chem.GetMolFrags(stripped, asMols=True)
    if len(frags) > 1:
        # Keep the largest fragment by heavy atom count
        stripped = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        logger.info(f"Molecule has {len(frags)} fragments. Keeping the largest by heavy atom count ({stripped.GetNumHeavyAtoms()} heavy atoms).")
    elif len(frags) == 0:
        # If everything was stripped, fallback to the original mol's largest fragment
        orig_frags = Chem.GetMolFrags(mol, asMols=True)
        if orig_frags:
            stripped = max(orig_frags, key=lambda m: m.GetNumHeavyAtoms())
        else:
            stripped = mol

    return stripped


# ──────────────────────────────────────────────────────────────
# Fingerprints
# ──────────────────────────────────────────────────────────────

def compute_fingerprint(
    mol: Chem.Mol,
    fp_type: str = "morgan",
    radius: int = 2,
    n_bits: int = 2048,
) -> DataStructs.ExplicitBitVect:
    """
    Compute a molecular fingerprint.

    Args:
        mol: RDKit Mol object.
        fp_type: Fingerprint type — "morgan", "rdkit", "topological".
        radius: Radius for Morgan fingerprints.
        n_bits: Number of bits in the fingerprint.

    Returns:
        Fingerprint as ExplicitBitVect.
    """
    if fp_type == "morgan":
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )
    elif fp_type == "rdkit":
        generator = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=n_bits)
    elif fp_type == "topological":
        generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=n_bits)
    else:
        raise ValueError(f"Unknown fingerprint type: '{fp_type}'. Use 'morgan', 'rdkit', or 'topological'.")

    return generator.GetFingerprint(mol)


def compute_tanimoto_similarity(mol1: Chem.Mol, mol2: Chem.Mol, fp_type: str = "morgan") -> float:
    """
    Compute Tanimoto similarity between two molecules.

    Args:
        mol1: First molecule.
        mol2: Second molecule.
        fp_type: Fingerprint type for comparison.

    Returns:
        Tanimoto similarity coefficient (0.0–1.0).
    """
    fp1 = compute_fingerprint(mol1, fp_type=fp_type)
    fp2 = compute_fingerprint(mol2, fp_type=fp_type)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


# ──────────────────────────────────────────────────────────────
# Molecular Property Calculation
# ──────────────────────────────────────────────────────────────

def calculate_basic_properties(mol: Chem.Mol) -> dict:
    """
    Calculate basic molecular properties relevant to drug-likeness.

    Args:
        mol: RDKit Mol object.

    Returns:
        Dict with keys:
            MW, LogP, HBD, HBA, TPSA, RotBonds, RingCount,
            AromaticRings, FractionCSP3, MolarRefractivity,
            NumHeavyAtoms, NumAtoms, SMILES
    """
    # Count carbon atoms
    num_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)

    return {
        "MW": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "TPSA": round(Descriptors.TPSA(mol), 2),
        "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "FractionCSP3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        "MolarRefractivity": round(Descriptors.MolMR(mol), 2),
        "NumHeavyAtoms": mol.GetNumHeavyAtoms(),
        "NumAtoms": mol.GetNumAtoms(),
        "NumHeteroatoms": rdMolDescriptors.CalcNumHeteroatoms(mol),
        "NumCarbons": num_carbons,
        "FormalCharge": Chem.rdmolops.GetFormalCharge(mol),
        "SMILES": Chem.MolToSmiles(mol),
    }


# ──────────────────────────────────────────────────────────────
# Molecule Naming
# ──────────────────────────────────────────────────────────────

import re

def sanitize_molecule_name(name: str) -> str:
    """
    Sanitize molecule name for filenames.
    - Convert to lowercase.
    - Replace spaces and non-alphanumeric chars with underscores.
    - Remove consecutive underscores.
    - Strip leading/trailing underscores.
    """
    if not name:
        return "unknown"
    s = name.lower()
    s = re.sub(r'[^a-z0-9_]', '_', s)
    s = re.sub(r'_+', '_', s)
    s = s.strip('_')
    return s if s else "unknown"


def get_mol_cid(mol: Chem.Mol) -> str:
    """Extract CID if available, else return N/A."""
    for prop in ["PUBCHEM_COMPOUND_CID", "PUBCHEM_CID", "CID", "cid"]:
        if mol.HasProp(prop):
            val = mol.GetProp(prop).strip()
            if val:
                return val
    if mol.HasProp("_Name"):
        val = mol.GetProp("_Name").strip()
        if val.isdigit():
            return val
    return "N/A"


def get_mol_name(mol: Chem.Mol, fallback_index: int | None = None) -> str:
    """
    Extract the molecule name from its properties, filename, or fallback.
    Priority:
    1. Filename (if format is pdb, or if other name properties are generic/missing)
    2. Molecule property (_Name) if not generic
    3. Metadata properties (Compound_Name, COMPND, HEADER, TITLE, etc.)
    4. Fallback index/UUID
    """
    # Extract source format and filename if available
    source_format = mol.GetProp("_source_format").strip().lower() if mol.HasProp("_source_format") else None
    source_filename = mol.GetProp("_source_filename").strip() if mol.HasProp("_source_filename") else None

    # Helper check for generic name
    def is_generic_name(n: str | None) -> bool:
        if not n:
            return True
        n_lower = n.lower()
        return "noname" in n_lower or n_lower in ("", "unknown", "n/a", "none")

    # 1. Filename priority:
    # Always prioritize filename for PDB ligands, or if the internal name properties are generic
    if source_format == "pdb" and source_filename and not is_generic_name(source_filename):
        return source_filename

    # Check internal properties
    name_prop = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else None

    # If the _Name property is a generic "NONAME", we should fall back to filename if available
    if is_generic_name(name_prop) and source_filename and not is_generic_name(source_filename):
        return source_filename

    # 2. Molecule property (_Name) if not generic
    if name_prop and not is_generic_name(name_prop):
        # If name is a simple digit, it could be a CID; prioritize filename if we have it
        if name_prop.isdigit() and source_filename and not is_generic_name(source_filename) and not source_filename.isdigit():
            return source_filename
        elif name_prop.isdigit():
            return f"CID_{name_prop}"
        return name_prop

    # 3. Known metadata properties
    possible_props = [
        "Compound_Name", "COMPOUND_NAME", "compound_name", "compound_Name",
        "Name", "NAME", "name", 
        "Title", "TITLE", "title",
        "PUBCHEM_COMPOUND_NAME", "PubChem Compound Name",
        "COMPND", "compnd", "HEADER", "header"
    ]
    for prop in possible_props:
        if mol.HasProp(prop):
            val = mol.GetProp(prop).strip()
            if val and not is_generic_name(val):
                return val

    # 4. Fallback to filename (for non-PDB formats) if we have it
    if source_filename and not is_generic_name(source_filename):
        return source_filename

    # 5. Fallback index or UUID/SMILES hash
    if fallback_index is not None:
        return f"mol_{fallback_index}"

    try:
        smiles = Chem.MolToSmiles(mol)
        return f"mol_{abs(hash(smiles)) % 100000}"
    except Exception:
        import uuid
        return f"mol_{str(uuid.uuid4())[:8]}"


def set_mol_name(mol: Chem.Mol, name: str) -> Chem.Mol:
    """
    Set the molecule name (_Name property).

    Args:
        mol: RDKit Mol object.
        name: Name to set.

    Returns:
        The mol object (modified in place).
    """
    mol.SetProp("_Name", name)
    return mol


# ──────────────────────────────────────────────────────────────
# Bulk Utilities
# ──────────────────────────────────────────────────────────────

def mols_to_smiles_list(mols: list[Chem.Mol]) -> list[str]:
    """
    Convert a list of Mol objects to a list of canonical SMILES strings.

    Args:
        mols: List of RDKit Mol objects.

    Returns:
        List of SMILES strings.
    """
    return [Chem.MolToSmiles(mol) for mol in mols if mol is not None]


def deduplicate_mols(mols: list[Chem.Mol]) -> list[Chem.Mol]:
    """
    Remove duplicate molecules based on canonical SMILES.

    Preserves the first occurrence of each unique structure.

    Args:
        mols: List of RDKit Mol objects.

    Returns:
        Deduplicated list of Mol objects.
    """
    seen_smiles: set[str] = set()
    unique_mols: list[Chem.Mol] = []

    for mol in mols:
        if mol is None:
            continue
        smiles = Chem.MolToSmiles(mol)
        if smiles not in seen_smiles:
            seen_smiles.add(smiles)
            unique_mols.append(mol)

    removed = len(mols) - len(unique_mols)
    if removed > 0:
        logger.info(f"Deduplicated: removed {removed} duplicates, {len(unique_mols)} unique remain")

    return unique_mols
