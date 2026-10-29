"""
Unit tests for peptide ADMET, permeability, and safety property predictions.
"""

import pytest
from core.peptide_docking.config import AdmetConfig
from core.peptide_docking.input_validation import PeptideCandidate
from core.peptide_docking.peptide_admet import (
    LocalPeptideADMETProvider,
    LocalPhysicochemicalProvider,
    PepAdmetProvider,
    PepLifeProvider,
    get_admet_provider,
    predict_peptide_admet,
    display_peptide_admet_summary,
    estimate_peptide_half_life,
    predict_peptide_permeability,
    predict_peptide_bbb_penetration,
    assess_peptide_toxicity,
    compute_maccs_similarity,
    is_peptide_cyclic,
)
from rdkit import Chem


def test_local_peptide_admet_provider_linear():
    provider = LocalPeptideADMETProvider()
    res = provider.predict(["ACDEFGH", "WYVIK"])
    assert len(res) == 2

    r1 = res[0]
    assert r1["status"] == "success"
    assert r1["molecular_weight_da"] > 500
    assert isinstance(r1["logp"], float)
    assert isinstance(r1["hbd"], int)
    assert isinstance(r1["hba"], int)
    assert isinstance(r1["tpsa_angstrom2"], float)
    assert "isoelectric_point_pi" in r1
    assert "net_charge_ph74" in r1
    assert "gravy_hydropathicity" in r1
    assert "instability_index" in r1
    assert "boman_index_kcal_mol" in r1

    # MACCS similarity
    assert r1["maccs_ref_match"] != "N/A"
    assert 0.0 <= float(r1["maccs_similarity"]) <= 1.0

    # Half life
    assert "half_life" in r1
    assert r1["predicted_stability"] in ("stable", "unstable")

    # Permeability for linear peptide
    assert "cm/s" in r1["caco2_linear_papp"]
    assert r1["caco2_cyclic_papp"] == "N/A (linear sequence)"
    assert r1["pampa_cyclic_papp"] == "N/A (cyclic-only)"
    assert r1["rrck_cyclic_papp"] == "N/A (cyclic-only)"
    assert "Low" in r1["caco2_all_peptide"] or "Moderate" in r1["caco2_all_peptide"]

    # BBB
    assert r1["bbb_penetration"] in ("BBB+ (Penetrant)", "BBB- (Non-penetrant)")

    # Toxicity
    assert "risk" in r1["peptide_toxicity_risk"].lower() or "safe" in r1["peptide_toxicity_risk"].lower()


def test_cyclic_peptide_permeability():
    # Oxytocin has 2 cysteines (CYIQNCPLG) forming a disulfide loop
    provider = LocalPeptideADMETProvider()
    res = provider.predict(["CYIQNCPLG"])
    assert len(res) == 1
    r = res[0]

    assert is_peptide_cyclic("CYIQNCPLG") is True
    assert "cm/s" in r["caco2_cyclic_papp"]
    assert "cm/s" in r["pampa_cyclic_papp"]
    assert "cm/s" in r["rrck_cyclic_papp"]


def test_peptide_toxicity_assessment():
    # 1. Benign/safe therapeutic-like peptide
    risk1, notes1, src1 = assess_peptide_toxicity("ACDEFGH", net_charge=-0.5, gravy=-0.3, boman_index=1.2)
    assert "Low risk" in risk1
    assert "Peptide Hemolytic Model" in src1

    # 2. Strongly cationic and hydrophobic peptide (membrane disruptor)
    risk2, notes2, src2 = assess_peptide_toxicity("RRRRWWWWLLLL", net_charge=4.5, gravy=0.8, boman_index=3.1)
    assert "High risk" in risk2

    # 3. Non-canonical / out-of-domain sequence
    risk3, notes3, src3 = assess_peptide_toxicity("ACD123ZX", net_charge=1.0, gravy=0.0, boman_index=1.0)
    assert risk3 == "toxicity undetermined"
    assert "outside" in notes3.lower()


def test_maccs_similarity_calculation():
    mol = Chem.MolFromSequence("FCWKTCT")
    best_ref, sim = compute_maccs_similarity(mol)
    assert "Octreotide" in best_ref
    assert sim == 1.0  # Exact sequence match


def test_admet_provider_factory():
    # Default is local
    p_def = get_admet_provider()
    assert isinstance(p_def, LocalPeptideADMETProvider)

    p_loc = get_admet_provider("local")
    assert isinstance(p_loc, LocalPeptideADMETProvider)

    # pepadmet redirects to local engine with warning
    p_pep = get_admet_provider("pepadmet")
    assert isinstance(p_pep, (LocalPeptideADMETProvider, PepAdmetProvider))

    p_life = get_admet_provider("peplife")
    assert isinstance(p_life, PepLifeProvider)


def test_peplife_provider_with_local_enrichment():
    provider = PepLifeProvider(endpoint_url="http://127.0.0.1:54321/nonexistent", timeout_sec=1)
    res = provider.predict(["MAGV", "RYVIK"])
    assert len(res) == 2

    r_magv = res[0]
    assert r_magv["sequence"] == "MAGV"
    assert "half_life" in r_magv
    assert "20 hours" in r_magv["half_life"]
    # Check that it enriched with all local descriptors
    assert "molecular_weight_da" in r_magv
    assert "logp" in r_magv
    assert "caco2_linear_papp" in r_magv
    assert "peptide_toxicity_risk" in r_magv
    assert "maccs_ref_match" in r_magv


def test_predict_peptide_admet_pipeline(tmp_path):
    cands = [
        PeptideCandidate(id="p1", sequence="ACDEFGH", source="sequence"),
        PeptideCandidate(id="p2", sequence="CYIQNCPLG", source="sequence"),
    ]
    cfg = AdmetConfig(enabled=True, provider="local")
    df = predict_peptide_admet(cands, config=cfg, output_dir=tmp_path)

    assert len(df) == 2
    tsv_path = tmp_path / "admet_predictions.tsv"
    assert tsv_path.is_file()

    # Check that required columns are in TSV
    expected_cols = [
        "molecular_weight_da", "logp", "hbd", "hba", "tpsa_angstrom2",
        "half_life", "maccs_ref_match", "maccs_similarity",
        "caco2_linear_papp", "caco2_cyclic_papp", "caco2_all_peptide",
        "pampa_cyclic_papp", "rrck_cyclic_papp", "bbb_penetration",
        "peptide_toxicity_risk", "toxicity_source",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_display_peptide_admet_summary():
    cands = [
        PeptideCandidate(id="p1", sequence="ACDEFGH", source="sequence"),
    ]
    cfg = AdmetConfig(enabled=True, provider="local")
    df = predict_peptide_admet(cands, config=cfg)
    display_peptide_admet_summary(df, provider_name="local")
    assert not df.empty
