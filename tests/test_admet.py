"""
Tests for ADMET Screening Module.
"""

from core.admet import (
    evaluate_bbb,
    detect_target_type,
    calculate_admet_risk_score,
    _predict_rdkit_fallback,
)


def test_evaluate_bbb():
    # CNS target with high BBB score -> positive impact and PASS status
    cns_res = evaluate_bbb(0.85, target_type="cns")
    assert cns_res["BBB_status"] == "PASS"
    assert cns_res["BBB_impact"] == "positive"

    # CNS target with low BBB score -> negative impact and FAIL status
    cns_low = evaluate_bbb(0.2, target_type="cns")
    assert cns_low["BBB_status"] == "FAIL"
    assert cns_low["BBB_impact"] == "negative"

    # Peripheral target with high BBB score -> cautionary impact
    periph = evaluate_bbb(0.85, target_type="peripheral")
    assert periph["BBB_impact"] == "cautionary"


def test_detect_target_type():
    assert detect_target_type(project_name="gaba_a_receptor") == "cns"
    assert detect_target_type(project_name="alzheimer_bace1") == "cns"
    assert detect_target_type(project_name="covid_mpro") == "general"
    # Config explicit override
    assert detect_target_type(config={"target_type": "peripheral"}) == "peripheral"


def test_admet_risk_score():
    preds = {
        "hERG": 0.1,
        "AMES": 0.05,
        "DILI": 0.1,
        "ClinTox": 0.1,
        "HIA_Hou": 0.9,
        "BBB_Martins": 0.7,
    }
    risk = calculate_admet_risk_score(preds, target_type="general")
    assert "admet_risk_score" in risk
    assert "admet_risk_breakdown" in risk
    assert 0 <= risk["admet_risk_score"] <= 100


def test_predict_rdkit_fallback(sample_mols):
    df = _predict_rdkit_fallback(sample_mols)
    assert not df.empty
    assert "MW" in df.columns
    assert "LogP" in df.columns
    assert "hERG" in df.columns
    assert "BBB_Martins" in df.columns
    assert "toxicity_proxy" in df.columns
