import copy
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/spectral_reachability/codex_beam_mppi_physical_pair_20261030.py"
spec = importlib.util.spec_from_file_location("physical_pair", PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


@pytest.fixture
def data():
    return mod.load_sources()


def row(data, arm, root=0):
    report = data["beam" if arm == mod.BEAM else "mppi"]
    return next(r for r in report["rows"] if r["arm"] == arm and r["root_slot"] == root)


def summary(data, arm, root=0):
    return next(r for r in data["comparison"]["rows"] if r["arm"] == arm and r["root_slot"] == root)


def test_actual_data(data):
    out = mod.audit(data)
    assert out["fixed_pair_denominator"] == out["observed_pairs"] == 6
    assert all(p["mppi_minus_beam"]["all_xy_error"] == 0 for p in out["pairs"] if p["step"] == 0)


@pytest.mark.parametrize("status", ["native_success", "native_stopped", "failed"])
def test_stopped_and_failed_missing_is_null(data, status):
    r = row(data, mod.MPPI)
    r["status"] = summary(data, mod.MPPI)["physical_status"] = status
    r["own_returned"] = summary(data, mod.MPPI)["own_returned"] = 5
    r["checkpoints"] = [cp for cp in r["checkpoints"] if cp["step"] <= 5]
    out = mod.audit(data)
    p = next(x for x in out["pairs"] if x["root_slot"] == 0 and x["step"] == 10)
    assert out["observed_pairs"] == 5 and out["fixed_pair_denominator"] == 6
    assert p["mppi_minus_beam"] is None and p["values"][mod.MPPI] is None
    assert p["missing"][mod.MPPI]["status"] == status


def test_checkpoint_beyond_returned_rejected(data):
    row(data, mod.MPPI)["own_returned"] = summary(data, mod.MPPI)["own_returned"] = 5
    row(data, mod.MPPI)["status"] = summary(data, mod.MPPI)["physical_status"] = "failed"
    with pytest.raises(ValueError, match="outside returned"):
        mod.audit(data)


@pytest.mark.parametrize("mutation", ["goal", "identity", "clock", "duplicate", "nan", "length", "hash", "selection", "root"])
def test_bad_evidence_rejected(data, mutation):
    r = row(data, mod.MPPI)
    s = summary(data, mod.MPPI)
    if mutation == "goal":
        r["goal"][0] += 1
    elif mutation == "identity":
        s["identity_sha256"] = "different"
    elif mutation == "clock":
        # Change both identities so equality alone cannot pass.
        summary(data, mod.BEAM)["identity"]["packet_steps"] = 4
        s["identity"]["packet_steps"] = 4
    elif mutation == "duplicate":
        r["checkpoints"].append(copy.deepcopy(r["checkpoints"][0]))
    elif mutation == "nan":
        r["checkpoints"][0]["state"][0] = float("nan")
    elif mutation == "length":
        r["goal"].pop()
    elif mutation == "hash":
        r["checkpoints"][0]["file_sha256"] = "wrong"
    elif mutation == "selection":
        r["selection_digest"] = "wrong"
    else:
        r["root_slot"] = 1
    with pytest.raises(ValueError):
        mod.audit(data)


def test_source_hash_tampering(tmp_path, monkeypatch):
    (tmp_path / "input.json").write_text("{}")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "SOURCES", {"x": ("input.json", "0" * 64)})
    with pytest.raises(ValueError, match="source hash mismatch"):
        mod.load_sources()


def test_exclusive_write_and_verify(tmp_path, monkeypatch):
    import sys
    output = tmp_path / "output.json"
    monkeypatch.setattr(mod, "OUTPUT", output)
    monkeypatch.setattr(sys, "argv", ["audit"])
    mod.main()
    with pytest.raises(FileExistsError):
        mod.main()
    monkeypatch.setattr(sys, "argv", ["audit", "--verify"])
    mod.main()
    output.write_text("{}")
    with pytest.raises(ValueError, match="output mismatch"):
        mod.main()
