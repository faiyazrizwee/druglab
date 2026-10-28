"""
DrugLab Synthetic Accessibility Scorer (SAScore).

Estimates synthetic accessibility of small molecules on a scale from 1 (easy to synthesize)
to 10 (very difficult to synthesize), based on the methodology of Ertl & Schuffenhauer
(J. Cheminform. 2009, 1, 8).

Attempts to import RDKit's Contrib sascorer if available, and provides a robust,
self-contained analytical fallback based on structural complexity, ring topology,
bridgeheads, spiro systems, chiral centers, and molecular size.
"""

import math
from typing import Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski

# Check for RDKit Contrib sascorer
_HAS_RDKIT_SASCORER = False
try:
    from rdkit.Contrib.SA_Score import sascorer as rdkit_sascorer
    _HAS_RDKIT_SASCORER = True
except Exception:
    _HAS_RDKIT_SASCORER = False


def _heuristic_sascore(mol: Chem.Mol) -> float:
    """
    Compute synthetic accessibility score based on structural complexity metrics.
    
    Returns a score calibrated from 1.0 (easy) to 10.0 (difficult).
    """
    if mol is None:
        return 10.0

    num_heavy = mol.GetNumHeavyAtoms()
    if num_heavy == 0:
        return 10.0

    # 1. Size penalty
    mw = Descriptors.MolWt(mol)
    size_penalty = 0.0
    if num_heavy > 50:
        size_penalty += math.log10(num_heavy / 40.0) * 2.0
    elif num_heavy > 35:
        size_penalty += 0.5

    # 2. Chiral / Stereocenter complexity
    chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    stereo_penalty = math.log1p(chiral_centers) * 1.2

    # 3. Ring topology complexity
    ring_info = mol.GetRingInfo()
    num_rings = ring_info.NumRings()
    
    # Macrocycle penalty (rings > 8 atoms)
    macrocycles = 0
    small_rings = 0
    for ring in ring_info.AtomRings():
        if len(ring) > 8:
            macrocycles += 1
        elif len(ring) in (3, 4):
            small_rings += 1

    ring_penalty = 0.0
    if macrocycles > 0:
        ring_penalty += macrocycles * 1.5
    if small_rings > 0:
        ring_penalty += small_rings * 0.8
    if num_rings > 4:
        ring_penalty += (num_rings - 4) * 0.4

    # Spiro atoms
    spiro_atoms = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    spiro_penalty = spiro_atoms * 1.0

    # Bridgehead atoms
    bridgehead_atoms = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    bridgehead_penalty = bridgehead_atoms * 1.2

    # 4. Rotatable bond flexibility penalty (extreme flexibility is hard to crystallize/purify)
    rot_bonds = Lipinski.NumRotatableBonds(mol)
    flex_penalty = 0.0
    if rot_bonds > 12:
        flex_penalty += (rot_bonds - 12) * 0.15

    # 5. Base score starting from standard drug fragment baseline (approx 2.2 for common scaffolds)
    base_score = 2.2
    
    # Heteroatom ratio penalty (very high or very low heteroatom fraction)
    hetero_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6))
    hetero_ratio = hetero_count / float(num_heavy)
    hetero_penalty = 0.0
    if hetero_ratio > 0.4:
        hetero_penalty += (hetero_ratio - 0.4) * 2.5

    raw_score = (
        base_score
        + size_penalty
        + stereo_penalty
        + ring_penalty
        + spiro_penalty
        + bridgehead_penalty
        + flex_penalty
        + hetero_penalty
    )

    # Clamp to [1.0, 10.0]
    return max(1.0, min(10.0, round(raw_score, 2)))


def calculate_sascore(mol: Chem.Mol) -> float:
    """
    Calculate the Synthetic Accessibility Score (SAScore) for an RDKit Mol.

    Args:
        mol: RDKit Chem.Mol object.

    Returns:
        float between 1.0 (easily synthesizable) and 10.0 (difficult).
    """
    if mol is None:
        return 10.0

    if _HAS_RDKIT_SASCORER:
        try:
            score = rdkit_sascorer.calculateScore(mol)
            return round(max(1.0, min(10.0, float(score))), 2)
        except Exception:
            pass

    return _heuristic_sascore(mol)
