"""
Tests for Drug-Likeness Screening Module.
"""

from core.druglikeness import (
    check_lipinski,
    check_veber,
    check_ghose,
    check_egan,
    check_lead_like,
    check_rule_of_3,
    resolve_profile_rules,
)


def test_resolve_profile_rules():
    assert "lipinski" in resolve_profile_rules("recommended")
    assert "veber" in resolve_profile_rules("recommended")
    assert "pains" in resolve_profile_rules("recommended")
    assert "rule_of_3" in resolve_profile_rules("fragment")
    assert resolve_profile_rules("custom", ["lipinski", "ghose"]) == ["lipinski", "ghose"]


def test_check_lipinski_pass():
    desc = {"MW": 180.16, "LogP": 1.19, "HBD": 1, "HBA": 3}
    passed, violations = check_lipinski(desc, max_violations=1)
    assert passed is True
    assert len(violations) == 0


def test_check_lipinski_fail():
    desc = {"MW": 650.0, "LogP": 6.5, "HBD": 8, "HBA": 15}
    passed, violations = check_lipinski(desc, max_violations=1)
    assert passed is False
    assert len(violations) >= 2


def test_check_veber():
    desc_pass = {"TPSA": 60.0, "RotBonds": 3}
    passed, _ = check_veber(desc_pass)
    assert passed is True

    desc_fail = {"TPSA": 160.0, "RotBonds": 12}
    passed, violations = check_veber(desc_fail)
    assert passed is False
    assert len(violations) == 2


def test_check_lead_like():
    desc = {"MW": 250.0, "LogP": 2.0, "RotBonds": 4}
    passed, _ = check_lead_like(desc)
    assert passed is True


def test_check_rule_of_3():
    desc = {"MW": 200.0, "LogP": 1.5, "HBD": 2, "HBA": 2, "RotBonds": 2}
    passed, _ = check_rule_of_3(desc)
    assert passed is True
