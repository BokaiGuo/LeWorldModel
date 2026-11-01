import importlib.util
import json
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/spectral_reachability/codex_full_d2p_physical_preflight_20261030.py"
spec = importlib.util.spec_from_file_location("d2p_preflight", PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


@pytest.fixture
def inputs():
    return mod.read_pinned()


def test_real_preflight(inputs):
    out = mod.build(inputs)
    assert out["actual_common"]["physical_goal7_equal"]
    assert out["existing_assets"]["sra"]["roots"] == 4
    assert out["existing_assets"]["multiround"]["roots"] == 1
    assert out["full_d2p_comparison"] is None and not out["dispatch_authorized"]
    assert all(x["status"] == "blocked" for x in out["required_before_execution"])
    assert out["static_entrypoints"]["multiround"]["arm_loop"] > 0


@pytest.mark.parametrize("case", ["root", "goal", "schedule", "budget", "duplicate", "missing", "admission", "symbol"])
def test_tampered_evidence_rejected(inputs, case):
    if case == "symbol":
        inputs["multiround_source"] = inputs["multiround_source"].replace("def arm_loop(", "def broken_loop(")
    else:
        key = "multiround_protocol" if case in ("root", "schedule", "admission") else "sra_physical"
        obj = json.loads(inputs[key])
        if case == "root":
            obj["root_sha256"] = "wrong"
        elif case == "goal":
            next(r for r in obj["rows"] if r["root_sha256"] == mod.MATCHED_ROOT)["context_identity"]["goal"][0] += 1
        elif case == "schedule":
            obj["contract"]["boundaries"] = [5, 10]
        elif case == "budget":
            obj["planning_accounting"]["per_root"]["total_predict_samples"] = 84
        elif case == "duplicate":
            obj["rows"][0] = obj["rows"][1]
        elif case == "missing":
            obj["rows"].pop()
        else:
            obj["contract"]["P1_admission"] = True
        inputs[key] = json.dumps(obj)
    with pytest.raises(ValueError):
        mod.build(inputs)


def test_hash_rejection(tmp_path, monkeypatch):
    (tmp_path / "x").write_text("{}")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "FILES", {"x": ("x", "0"*64)})
    with pytest.raises(ValueError, match="source hash"):
        mod.read_pinned()


def test_exclusive_and_reconstruction(tmp_path, monkeypatch):
    output = tmp_path / "out.json"
    monkeypatch.setattr(mod, "OUT", output)
    monkeypatch.setattr(sys, "argv", ["preflight"])
    mod.main()
    with pytest.raises(FileExistsError):
        mod.main()
    monkeypatch.setattr(sys, "argv", ["preflight", "--verify"])
    mod.main()
    output.write_text("{}")
    with pytest.raises(ValueError, match="output mismatch"):
        mod.main()
