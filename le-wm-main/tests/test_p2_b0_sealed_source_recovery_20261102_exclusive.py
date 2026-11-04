import pytest

from experiments.spectral_reachability import p2_b0_sealed_source_recovery_20261102_exclusive as m


def test_recovers_only_exact_historical_target(tmp_path):
    target, other = tmp_path / "target.py", tmp_path / "other.py"
    target.write_bytes(b"new")
    other.write_bytes(b"unchanged")
    bindings = {str(target): m.digest(b"old"), str(other): m.digest(b"unchanged")}
    result = m.census(bindings, target, b"old")
    assert len(result["current_mismatches"]) == 1
    assert result["all_binding_bytes_match_with_target_substitution"]
    assert target.read_bytes() == b"new"
    other.write_bytes(b"changed")
    assert not m.census(bindings, target, b"old")["all_binding_bytes_match_with_target_substitution"]
    with pytest.raises(ValueError, match="historical"):
        m.census(bindings, target, b"wrong")


def test_missing_and_symlink_are_not_accepted(tmp_path):
    target, missing, link = tmp_path / "target", tmp_path / "missing", tmp_path / "link"
    target.write_bytes(b"old")
    link.symlink_to(target)
    result = m.census({str(target): m.digest(b"old"), str(missing): m.digest(b"x"),
                       str(link): m.digest(b"old")}, target, b"old")
    assert len(result["remaining_mismatches_if_target_source_used"]) == 2
    assert not result["all_binding_bytes_match_with_target_substitution"]


def test_target_must_be_bound(tmp_path):
    with pytest.raises(ValueError, match="target"):
        m.census({}, tmp_path / "missing", b"old")
