import copy
import importlib.util
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/spectral_reachability/codex_sra_composition_timestep_20261030.py"
spec = importlib.util.spec_from_file_location("sra_timestep", PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


@pytest.fixture
def data():
    return mod.load()


def test_actual(data):
    result = mod.audit(data)
    assert result["fixed_denominator"] == result["observed_pairs"] == 44
    assert result["full_d2p_comparison"] is None
    assert all(p["conditional_minus_sra"]["joint_xy"] == 0 for p in result["pairs"] if p["step"] == 0)
    assert all(b["paired_unique_predict"] == 252 and b["predict_difference_conditional_minus_sra"] == 80 for b in result["budget_by_root"])


def test_order_independent(data):
    expected = mod.audit(data)
    data["physical"]["rows"].reverse()
    data["planning"]["rows"].reverse()
    data["prior_outcomes"]["paired"].reverse()
    assert mod.audit(data) == expected


def test_missing_step_is_null(data):
    data["physical"]["rows"][0]["trace"].pop(4)
    result = mod.audit(data)
    pair = next(p for p in result["pairs"] if p["root_sha256"] == mod.ROOTS[0] and p["step"] == 5)
    assert result["fixed_denominator"] == 44 and result["observed_pairs"] == 43
    assert pair["conditional_minus_sra"] is None and pair["missing_arms"] == [mod.ARMS[0]]


@pytest.mark.parametrize("error", ["goal", "root", "duplicate", "beyond", "nan", "horizon", "budget", "prior", "missing_row"])
def test_bad_evidence(data, error):
    row = data["physical"]["rows"][0]
    if error == "goal":
        row["context_identity"]["goal"][0] += 1
    elif error == "root":
        row["root_sha256"] = "wrong"
    elif error == "duplicate":
        row["trace"].append(copy.deepcopy(row["trace"][0]))
    elif error == "beyond":
        row["completed_commands"] = row["physical_env_step_returns"] = 5
        row["status"] = "failed"
    elif error == "nan":
        row["trace"][0]["state"][0] = float("nan")
    elif error == "horizon":
        row["planned_commands"] = 15
    elif error == "budget":
        data["planning"]["rows"][0]["completed_operations"]["jacobian_calls"] = 1
    elif error == "prior":
        data["prior_outcomes"]["paired"][0]["composition_minus_no_composition"]["agent_position_error"] += 1
    else:
        data["physical"]["rows"].pop()
    with pytest.raises(ValueError):
        mod.audit(data)


def test_hash_rejects(tmp_path, monkeypatch):
    (tmp_path / "x").write_text("{}")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "FILES", {"x": ("x", "0"*64)})
    with pytest.raises(ValueError, match="source hash"):
        mod.load()


def test_exclusive_and_verify(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    monkeypatch.setattr(mod, "OUT", output)
    monkeypatch.setattr(sys, "argv", ["audit"])
    mod.main()
    with pytest.raises(FileExistsError):
        mod.main()
    monkeypatch.setattr(sys, "argv", ["audit", "--verify"])
    mod.main()
    output.write_text("{}")
    with pytest.raises(ValueError, match="output mismatch"):
        mod.main()
