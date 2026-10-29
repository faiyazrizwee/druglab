"""
Unit tests for peptide size, length warnings, and filtering.
"""

from core.peptide_docking.config import FilteringConfig
from core.peptide_docking.input_validation import PeptideCandidate
from core.peptide_docking.peptide_filtering import filter_peptides


def test_filter_peptides_length_removed_with_warning(tmp_path):
    """
    Ensure length filter does NOT drop peptide candidates, but emits advisory warnings.
    """
    c1 = PeptideCandidate(id="short", sequence="ACD", source="sequence")      # len 3
    c2 = PeptideCandidate(id="good", sequence="ACDEFGH", source="sequence")   # len 7
    c3 = PeptideCandidate(id="long", sequence="A" * 35, source="sequence")    # len 35

    cfg = FilteringConfig(enabled=True, min_length_aa=5, max_length_aa=30)
    passed, filtered, df = filter_peptides([c1, c2, c3], config=cfg, output_dir=tmp_path)

    # All candidates pass to docking — length filter is bypassed
    assert len(passed) == 3
    assert len(filtered) == 0
    assert (tmp_path / "peptide_filtering.tsv").is_file()

    # Verify warnings are present in candidate properties
    assert "length_warning" in c1.properties
    assert "minimum" in c1.properties["length_warning"]
    assert "length_warning" in c3.properties
    assert "maximum" in c3.properties["length_warning"]
    assert "length_warning" not in c2.properties

    # Check reason strings in dataframe
    reasons = {r["candidate_id"]: r["filter_reason"] for _, r in df.iterrows()}
    assert "minimum" in reasons["short"]
    assert "maximum" in reasons["long"]
    assert reasons["good"] == ""


def test_filter_molecular_weight(tmp_path):
    """Ensure molecular weight filter can still filter out excessively heavy candidates."""
    c_heavy = PeptideCandidate(id="heavy", sequence="W" * 30, source="sequence")  # MW ~5600 Da > 4000 Da
    cfg = FilteringConfig(enabled=True, max_mw=4000.0)
    passed, filtered, df = filter_peptides([c_heavy], config=cfg, output_dir=tmp_path)

    assert len(passed) == 0
    assert len(filtered) == 1
    assert "Molecular weight" in filtered[0].failure_reason


def test_filter_disabled():
    c1 = PeptideCandidate(id="short", sequence="AC", source="sequence")
    cfg = FilteringConfig(enabled=False, min_length_aa=5)
    passed, filtered, df = filter_peptides([c1], config=cfg)
    assert len(passed) == 1
    assert len(filtered) == 0
