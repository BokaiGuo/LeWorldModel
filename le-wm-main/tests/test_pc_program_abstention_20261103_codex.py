import pytest

from experiments.spectral_reachability import pc_program_abstention_20261103_codex as entry
from experiments.spectral_reachability.pc_program_abstention_20261103_codex import (
    select_with_abstention,
)


def program(name, score):
    value = 0. if name == "zero" else .1
    return {"name": name, "prefix": [[value, 0.]] * 5,
            "suffix": [[0., 0.]] * 10, "terminal_score": score}


def test_abstain_uses_actual_zero_fallback_when_gain_is_insufficient():
    result = select_with_abstention([program("zero", 1), program("a", .95)], .1)
    assert result["abstained"]
    assert result["selected"]["name"] == "zero"
    assert result["fallback"] == "optimized_zero_prefix"


def test_accept_only_strictly_better_and_validate_control():
    result = select_with_abstention([program("zero", 1), program("a", .8)], .1)
    assert not result["abstained"]
    assert result["selected"]["name"] == "a"
    with pytest.raises(ValueError):
        select_with_abstention([program("a", .8)], .1)
    bad = program("zero", 1)
    bad["prefix"][0][0] = .5
    with pytest.raises(ValueError):
        select_with_abstention([bad], .1)


def test_compiler_wrapper_replaces_actual_selection(monkeypatch):
    zero, candidate = program("zero", 1), program("a", .95)
    monkeypatch.setattr(entry, "compile_program", lambda *a, **k: {
        "selected": candidate, "programs": [zero, candidate],
        "qualified_against_zero_only": True,
    })
    result = entry.compile_with_abstention(minimum_gain=.1)
    assert result["abstention"]["abstained"]
    assert result["selected"] is zero
    assert "qualified_against_zero_only" not in result
